# backtest.py
# Final cleaned backtester (RSI + Stochastic + MACD + BB + ATR + EMA)
# - Grid-search
# - Top-3 results
# - Equity plots
# Requirements:
#   pip install pandas numpy matplotlib scipy python-binance
# Usage:
#   python backtest.py

import math
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import argrelextrema

# Optional: python-binance client for fetching futures klines
try:
    from binance.client import Client
except Exception:
    Client = None
    print("Warning: python-binance not installed. Install it to fetch data from Binance.")


# ---------------------------
# 1) Data fetcher (Binance Futures)
# ---------------------------
def fetch_klines_binance(symbol='BTCUSDT', interval='15m', total_bars=3000, client=None):
    """
    Fetch historical futures klines from Binance (futures_klines).
    If client is None, attempt to create a Client() without keys (public endpoints work).
    Returns DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if Client is None and client is None:
        raise RuntimeError("python-binance Client not available. Install python-binance or pass client.")
    if client is None:
        client = Client()

    limit = 1000
    data = []
    end_time = None

    while len(data) < total_bars:
        need = min(limit, total_bars - len(data))
        try:
            klines = client.futures_klines(symbol=symbol, interval=interval, limit=need, endTime=end_time)
        except Exception as e:
            print("Binance API error:", e)
            break
        if not klines:
            break
        # Prepend chunk so result is chronological at the end
        data = klines + data
        end_time = klines[0][0] - 1
        time.sleep(0.08)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        'timestamp','open','high','low','close','volume',
        'close_time','quote_asset_volume','number_of_trades',
        'taker_buy_base','taker_buy_quote','ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    df = df[['timestamp','open','high','low','close','volume']].drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    return df


# ---------------------------
# 2) Indicators
# ---------------------------
def add_basic_indicators(df, rsi_period=14, stoch_k=14, stoch_d=3, stoch_smooth_k=3,
                         macd_fast=12, macd_slow=26, macd_signal=9,
                         atr_period=14, ema_period=50, vol_period=20):
    df = df.copy()
    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(rsi_period).mean() / down.rolling(rsi_period).mean().replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Stochastic %K and %D
    low_n = df['low'].rolling(stoch_k).min()
    high_n = df['high'].rolling(stoch_k).max()
    df['stoch_k_raw'] = (df['close'] - low_n) / (high_n - low_n)
    df['stoch_k'] = df['stoch_k_raw'].rolling(stoch_smooth_k).mean() * 100
    df['stoch_d'] = df['stoch_k'].rolling(stoch_d).mean()

    # MACD
    ema_fast = df['close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=macd_slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_sig'] = df['macd'].ewm(span=macd_signal, adjust=False).mean()

    # ATR
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum((df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()))
    df['atr'] = tr.rolling(atr_period).mean()

    # EMA
    df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()

    # Volume avg
    df['vol_avg'] = df['volume'].rolling(vol_period).mean()

    return df


def compute_bollinger(df, bb_period=20, bb_std=2.0):
    df = df.copy()
    df['bb_mid'] = df['close'].rolling(bb_period).mean()
    df['bb_std'] = df['close'].rolling(bb_period).std()
    df['bb_upper'] = df['bb_mid'] + bb_std * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - bb_std * df['bb_std']
    df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_mid']) * 100
    return df


# ---------------------------
# 3) Utils: drawdown, sharpe
# ---------------------------
def max_drawdown(equity):
    arr = np.array(equity)
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / peak
    return float(dd.min() * 100)  # percent (negative)


def sharpe_ratio(equity):
    arr = np.array(equity)
    if arr.size < 2:
        return 0.0
    rets = np.diff(arr) / arr[:-1]
    if rets.std() == 0:
        return 0.0
    return float((rets.mean() / rets.std()) * math.sqrt(252))


# ---------------------------
# 4) Signal helpers
# ---------------------------
def macd_cross_up(df, i):
    return (df['macd'].iat[i] > df['macd_sig'].iat[i]) and (df['macd'].iat[i-1] <= df['macd_sig'].iat[i-1])

def macd_cross_down(df, i):
    return (df['macd'].iat[i] < df['macd_sig'].iat[i]) and (df['macd'].iat[i-1] >= df['macd_sig'].iat[i-1])


# ---------------------------
# 5) Backtest strategy (lenient, realistic signals)
# ---------------------------
def backtest_strategy(df_raw, params, daily_limit=False):
    """
    Combined strategy:
      - LONG: (RSI oversold & stoch rising) OR MACD bullish,
              price generally above EMA (soft)
      - SHORT: (RSI overbought & stoch falling) OR MACD bearish,
              price below EMA (soft)
    Returns metrics dict with equity/trades.
    """
    df = df_raw.copy()
    # compute indicators
    df = add_basic_indicators(df,
                              rsi_period=params['rsi_period'],
                              stoch_k=params['stoch_k'],
                              stoch_d=params['stoch_d'],
                              stoch_smooth_k=params['stoch_smooth_k'],
                              macd_fast=params['macd_fast'],
                              macd_slow=params['macd_slow'],
                              macd_signal=params['macd_sig'],
                              atr_period=params['atr_period'],
                              ema_period=params['ema_period'],
                              vol_period=params.get('vol_period', 20))
    df = compute_bollinger(df, bb_period=params['bb_period'], bb_std=params['bb_std'])

    # remove NaNs introduced by rolling calculations
    df = df.dropna().reset_index(drop=True)
    if len(df) < 30:
        return None

    trades = []
    in_pos = False
    pos = None
    daily_trade_flag = False

    # iterate bars
    for i in range(1, len(df)):
        # reset daily flag on new day
        if daily_limit and df['timestamp'].iat[i].date() != df['timestamp'].iat[i-1].date():
            daily_trade_flag = False

        # if already traded today, skip (daily_limit)
        if daily_limit and daily_trade_flag:
            continue

        price = df['close'].iat[i]

        # ENTRY (lenient combination)
        if not in_pos:
            # RHS: compute simple booleans
            rsi = df['rsi'].iat[i]
            stoch_k = df['stoch_k'].iat[i]
            stoch_d = df['stoch_d'].iat[i]
            macd_bull = df['macd'].iat[i] > df['macd_sig'].iat[i]
            macd_bear = df['macd'].iat[i] < df['macd_sig'].iat[i]
            price_above_ema = price > df['ema'].iat[i] * 0.99  # soft bias
            price_below_ema = price < df['ema'].iat[i] * 1.01

            # long conditions (either combo OR MACD)
            cond_rsi_long = rsi <= params['rsi_os']
            cond_stoch_long = (stoch_k > stoch_d) and (df['stoch_k'].iat[i-1] <= df['stoch_d'].iat[i-1])
            long_signal = (cond_rsi_long and cond_stoch_long and price_above_ema) or (macd_bull and price_above_ema)

            # short conditions
            cond_rsi_short = rsi >= params['rsi_ob']
            cond_stoch_short = (stoch_k < stoch_d) and (df['stoch_k'].iat[i-1] >= df['stoch_d'].iat[i-1])
            short_signal = (cond_rsi_short and cond_stoch_short and price_below_ema) or (macd_bear and price_below_ema)

            # volume/squeeze optional filters (we keep light)
            vol_ok = True
            bb_width_ok = True

            if long_signal and vol_ok and bb_width_ok:
                entry = price
                sl = entry - params['atr_mult'] * df['atr'].iat[i]
                tp = entry + params['rr'] * (entry - sl)
                pos = {'type': 'long', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_idx': i}
                in_pos = True
                daily_trade_flag = True
                continue

            if short_signal and vol_ok and bb_width_ok:
                entry = price
                sl = entry + params['atr_mult'] * df['atr'].iat[i]
                tp = entry - params['rr'] * (sl - entry)
                pos = {'type': 'short', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_idx': i}
                in_pos = True
                daily_trade_flag = True
                continue

        else:
            # Manage exits
            if pos['type'] == 'long':
                hit_sl = df['low'].iat[i] <= pos['sl']
                hit_tp = df['high'].iat[i] >= pos['tp']
                macd_rev = macd_cross_down(df, i)
                bb_upper_hit = df['high'].iat[i] >= df['bb_upper'].iat[i]

                if hit_sl or hit_tp or macd_rev or bb_upper_hit:
                    if hit_sl:
                        exit_price = pos['sl']
                    elif hit_tp:
                        exit_price = pos['tp']
                    elif bb_upper_hit:
                        exit_price = min(df['high'].iat[i], df['bb_upper'].iat[i])
                    else:
                        exit_price = df['close'].iat[i]
                    pnl_pct = (exit_price - pos['entry']) / pos['entry'] * 100
                    trades.append({**pos, 'exit_idx': i, 'exit_price': exit_price, 'pnl_pct': pnl_pct})
                    in_pos = False
                    pos = None
            else:
                hit_sl = df['high'].iat[i] >= pos['sl']
                hit_tp = df['low'].iat[i] <= pos['tp']
                macd_rev = macd_cross_up(df, i)
                bb_lower_hit = df['low'].iat[i] <= df['bb_lower'].iat[i]

                if hit_sl or hit_tp or macd_rev or bb_lower_hit:
                    if hit_sl:
                        exit_price = pos['sl']
                    elif hit_tp:
                        exit_price = pos['tp']
                    elif bb_lower_hit:
                        exit_price = max(df['low'].iat[i], df['bb_lower'].iat[i])
                    else:
                        exit_price = df['close'].iat[i]
                    pnl_pct = (pos['entry'] - exit_price) / pos['entry'] * 100
                    trades.append({**pos, 'exit_idx': i, 'exit_price': exit_price, 'pnl_pct': pnl_pct})
                    in_pos = False
                    pos = None

    # Build metrics
    trades_df = pd.DataFrame(trades)
    num_trades = int(len(trades_df)) if not trades_df.empty else 0
    total_pnl = float(trades_df['pnl_pct'].sum()) if not trades_df.empty else 0.0
    win_rate = float((trades_df['pnl_pct'] > 0).mean() * 100) if not trades_df.empty else 0.0
    avg_pnl = float(trades_df['pnl_pct'].mean()) if not trades_df.empty else 0.0

    # equity multiplicative
    equity = [1.0]
    cur = 1.0
    for _, r in trades_df.iterrows():
        cur *= (1 + r['pnl_pct'] / 100)
        equity.append(cur)
    if len(equity) < len(df):
        repeats = len(df) // len(equity) if len(equity) > 0 else 0
        remainder = len(df) % len(equity) if len(equity) > 0 else 0
        if repeats > 0:
            equity = equity * repeats + equity[:remainder]
        else:
            equity = equity + [equity[-1]] * (len(df) - len(equity))
    equity = equity[:len(df)]

    metrics = {
        'num_trades': num_trades,
        'total_pnl_pct': total_pnl,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'equity': equity,
        'trades_df': trades_df,
        'max_drawdown': max_drawdown(equity),
        'sharpe': sharpe_ratio(equity)
    }
    return metrics


# ---------------------------
# 6) Grid search and selection
# ---------------------------
def grid_search_and_select(symbol='BTCUSDT', interval='15m', total_bars=3000, grid=None,
                           min_trades_per_days_divisor=3, require_winrate=25, require_avg_pnl=-2.0):
    if grid is None:
        grid = {
            'rsi_period': [10, 14],
            'rsi_ob': [70, 75],
            'rsi_os': [30, 25],
            'stoch_k': [14],
            'stoch_d': [3],
            'stoch_smooth_k': [3],
            'macd_fast': [12],
            'macd_slow': [26],
            'macd_sig': [9],
            'bb_period': [20],
            'bb_std': [2.0],
            'atr_period': [14],
            'atr_mult': [1.0, 1.5],
            'ema_period': [50, 100],
            'rr': [1.5, 2.0]
        }

    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    results = []

    print(f"Grid search combos: {len(combos)} — fetching data ({symbol} {interval} {total_bars} bars)...")
    df_full = fetch_klines_binance(symbol=symbol, interval=interval, total_bars=total_bars)
    if df_full.empty:
        raise RuntimeError("No data received from Binance.")

    # run combos
    for combo in combos:
        p = dict(zip(keys, combo))
        params = {
            'rsi_period': p['rsi_period'],
            'rsi_ob': p['rsi_ob'],
            'rsi_os': p['rsi_os'],
            'stoch_k': p['stoch_k'],
            'stoch_d': p['stoch_d'],
            'stoch_smooth_k': p['stoch_smooth_k'],
            'macd_fast': p['macd_fast'],
            'macd_slow': p['macd_slow'],
            'macd_sig': p['macd_sig'],
            'bb_period': p['bb_period'],
            'bb_std': p['bb_std'],
            'atr_period': p['atr_period'],
            'atr_mult': p['atr_mult'],
            'ema_period': p['ema_period'],
            'rr': p['rr']
        }

        try:
            metrics = backtest_strategy(df_full, params)
        except Exception as e:
            print("Error in combo:", p, "->", e)
            continue

        if metrics is None:
            continue

        # days and min trades
        days = max(1, (df_full['timestamp'].iloc[-1].date() - df_full['timestamp'].iloc[0].date()).days)
        min_trades = max(1, days // min_trades_per_days_divisor)

        result_entry = {**params, **metrics, 'days': days, 'min_trades_required': min_trades}
        results.append(result_entry)

    if not results:
        raise RuntimeError("Grid produced no results.")

    df_res = pd.DataFrame(results)

    # apply filters
    filtered = df_res[
        (df_res['num_trades'] >= df_res['min_trades_required']) &
        (df_res['win_rate'] >= require_winrate) &
        (df_res['avg_pnl'] >= require_avg_pnl)
    ].copy()

    if filtered.empty:
        print("No combos passed strict filters — selecting top-3 by total_pnl (fallback).")
        top = df_res.sort_values('total_pnl_pct', ascending=False).head(3)
    else:
        top = filtered.sort_values('total_pnl_pct', ascending=False).head(3)

    top_list = []
    for _, row in top.iterrows():
        top_list.append({
            'params': {k: row[k] for k in grid.keys()},
            'num_trades': int(row['num_trades']),
            'total_pnl_pct': float(row['total_pnl_pct']),
            'win_rate': float(row['win_rate']),
            'avg_pnl': float(row['avg_pnl']),
            'max_drawdown': float(row['max_drawdown']),
            'sharpe': float(row['sharpe']),
            'equity': row['equity'],
            'trades_df': row['trades_df']
        })

    return df_res, top_list, df_full

# ---------------------------
# 8) Run workflow and print top-3
# ---------------------------
def run_backtest_workflow(symbol='BTCUSDT', interval='15m', total_bars=3000):
    print(f"Loading {symbol} {interval} ({total_bars} bars)...")
    t0 = time.time()

    try:
        df_res, top_list, df_full = grid_search_and_select(symbol=symbol, interval=interval, total_bars=total_bars)
    except Exception as e:
        print("ERROR during grid search:", e)
        return

    print("\n=== Grid Search Completed ===")
    print("Best 3 results:\n")

    if not top_list:
        print("No top strategies found.")
        return

    for i, item in enumerate(top_list):
        print(f"--- TOP {i+1} ---")
        print("Parameters:", item["params"])
        print(f"Trades: {item['num_trades']}")
        print(f"Total PnL %: {item['total_pnl_pct']:.2f}")
        print(f"Win rate: {item['win_rate']:.1f}%")
        print(f"Avg PnL: {item['avg_pnl']:.3f}")
        print(f"Max drawdown: {item['max_drawdown']:.2f}%")
        print(f"Sharpe: {item['sharpe']:.2f}")
        print()



    print(f"\nExecution time: {round(time.time() - t0, 2)} sec")
    return df_res, top_list, df_full


# ---------------------------
# 9) Entry
# ---------------------------
if __name__ == "__main__":
    # user settings
    SYMBOL = "ETHUSDT"
    INTERVAL = "15m"
    TOTAL_BARS = 3000

    run_backtest_workflow(symbol=SYMBOL, interval=INTERVAL, total_bars=TOTAL_BARS)
