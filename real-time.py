import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import os
from pydantic import BaseModel, ConfigDict
import pandas_ta as ta
from apscheduler.schedulers.background import BackgroundScheduler
from websocket import create_connection
from threading import Thread, Timer
from bingxclient import BingxClient

# IMPORTANT: To fix the WebSocket error, run in your terminal:
# pip uninstall websocket
# pip install websocket-client
# This ensures the correct client library is installed, as 'websocket' is a different package.

load_dotenv()

class Config(BaseModel):
    model_config = ConfigDict(extra='allow')
    api_key: str
    api_secret: str
    symbols: list[str] = ["BTC-USDT"]
    quantity: float = 0.001
    interval: str = "1h"
    fee_rate: float = 0.0005
    leverage: int = 20
    history_lookback: int = 200
    rsi_period: int = 10
    rsi_ob: int = 70
    rsi_os: int = 25
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_smooth_k: int = 3
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    atr_mult: float = 1.5
    ema_period: int = 100
    rr: float = 1.5
    risk_percent: float = 0.01

class RealTimeTrader:
    def __init__(self, config_path: str = 'config.json'):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        handler = RotatingFileHandler('trader.log', maxBytes=5*1024*1024, backupCount=5)
        logging.getLogger().addHandler(handler)

        logging.info("[INIT] Starting RealTimeTrader...")

        # Load config from json, override keys with env if set
        try:
            with open(config_path, 'r') as f:
                json_config = json.load(f)
        except Exception as e:
            logging.error(f"[INIT] Failed to load config.json: {e}")
            raise ValueError("Config file required")

        api_key = os.getenv('BINGX_API_KEY') or json_config.get('api_key')
        api_secret = os.getenv('BINGX_API_SECRET') or json_config.get('api_secret')

        if not api_key or not api_secret:
            raise ValueError("API keys are required")

        json_config['api_key'] = api_key
        json_config['api_secret'] = api_secret

        self.config = Config(**json_config)

        self.symbols = self.config.symbols
        self.quantity = self.config.quantity
        self.interval = self.config.interval
        self.fee_rate = self.config.fee_rate
        self.leverage = self.config.leverage
        self.history_lookback = self.config.history_lookback
        self.poll_interval = self._interval_to_seconds(self.interval)

        # Strategy params
        self.rsi_period = self.config.rsi_period
        self.rsi_ob = self.config.rsi_ob
        self.rsi_os = self.config.rsi_os
        self.stoch_k = self.config.stoch_k
        self.stoch_d = self.config.stoch_d
        self.stoch_smooth_k = self.config.stoch_smooth_k
        self.macd_fast = self.config.macd_fast
        self.macd_slow = self.config.macd_slow
        self.macd_signal = self.config.macd_signal
        self.bb_period = self.config.bb_period
        self.bb_std = self.config.bb_std
        self.atr_period = self.config.atr_period
        self.atr_mult = self.config.atr_mult
        self.ema_period = self.config.ema_period
        self.rr = self.config.rr
        self.risk_percent = self.config.risk_percent

        # Client init
        self.client = BingxClient(self.config.api_key, self.config.api_secret)
        self.extend_client()

        # Data structures
        self.data: dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in self.symbols}
        self.positions: dict[str, str | None] = {s: None for s in self.symbols}

        # Set leverage
        for s in self.symbols:
            try:
                self.client.set_leverage(s, 'LONG', self.leverage)
                self.client.set_leverage(s, 'SHORT', self.leverage)
                logging.info(f"[INIT] Leverage set {s}: {self.leverage}x")
            except Exception as e:
                logging.error(f"[INIT][ERROR] leverage {s}: {e}")

        # WebSocket handler
        #self.ws_handler = WebSocketHandler(self)
        #Thread(target=self.ws_handler.start).start()

    def _interval_to_seconds(self, interval: str) -> int:
        return {'1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400}.get(interval, 3600)

    def extend_client(self):
        def safe_to_numeric(x):
            try:
                return pd.to_numeric(x, errors='coerce')
            except:
                return np.nan

        def get_klines(symbol: str, interval: str = '5m', startTime: int | None = None, endTime: int | None = None, limit: int = 1000) -> pd.DataFrame:
            path = "/openApi/swap/v2/quote/klines"
            params = {'symbol': symbol, 'interval': interval, 'limit': limit}
            if startTime: params['startTime'] = startTime
            if endTime: params['endTime'] = endTime
            data = self.client._public_request(path, params)
            if data.get('code') != 0:
                raise Exception(data.get('msg'))
            df = pd.DataFrame(data.get('data', []), columns=['ts','open','high','low','close','vol','ct','qv','t','tb','tq','i'])
            if df.empty: return pd.DataFrame(columns=['ts','open','high','low','close','vol'])
            df['ts'] = safe_to_numeric(df['ts'])
            df['ts'] = df['ts'].round().astype('Int64')
            df['ts'] = pd.to_datetime(df['ts'], unit='ms', errors='coerce')
            for c in ['open','high','low','close','vol']:
                df[c] = safe_to_numeric(df[c])
            return df[['ts','open','high','low','close','vol']]

        self.client.get_klines = get_klines

    def initialize_data(self, symbol: str) -> None:
        end = int(time.time() * 1000)
        start = max(0, end - self.history_lookback * self.poll_interval * 1000)
        df = self.client.get_klines(symbol, self.interval, startTime=start, endTime=end)
        self.data[symbol] = df
        logging.info(f"[DATA] {symbol} loaded {len(df)} candles")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_period)
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=self.stoch_k, d=self.stoch_d, smooth_k=self.stoch_smooth_k)
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']
        macd = ta.macd(df['close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        df['macd'] = macd['MACD_12_26_9']
        df['macd_sig'] = macd['MACDs_12_26_9']
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
        df['ema'] = ta.ema(df['close'], length=self.ema_period)
        return df

    def macd_cross_up(self, df: pd.DataFrame, i: int) -> bool:
        return df['macd'].iat[i] > df['macd_sig'].iat[i] and df['macd'].iat[i-1] <= df['macd_sig'].iat[i-1]

    def macd_cross_down(self, df: pd.DataFrame, i: int) -> bool:
        return df['macd'].iat[i] < df['macd_sig'].iat[i] and df['macd'].iat[i-1] >= df['macd_sig'].iat[i-1]

    def update_positions(self, symbol: str) -> None:
        try:
            positions = self.client.get_positions(symbol)
            position_open = any(float(p['positionAmt']) != 0 for p in positions)
            if not position_open and self.positions.get(symbol) is not None:
                logging.info(f"[{symbol}] Position closed (TP/SL or manually).")
                self.positions[symbol] = None
        except Exception as e:
            logging.error(f"[ERROR] {symbol} update_positions failed: {e}")

    def trade_logic(self, symbol: str) -> None:
        df = self.data[symbol].copy()
        if len(df) < 100:
            logging.info(f"[{symbol}] Not enough data for indicators")
            return
        df = self.calculate_indicators(df)
        i = len(df) - 1
        price = df['close'].iat[i]

        if self.positions.get(symbol) is not None:
            logging.info(f"[{symbol}] Already in position ({self.positions[symbol]}), skipping")
            return

        try:
            balance_data = self.client.get_balance()
            balance = float(balance_data.get('balance', {}).get('availableMargin', 0))
        except Exception as e:
            logging.error(f"[ERROR] Failed to get balance: {e}")
            return
        if balance <= 0:
            logging.warning(f"[{symbol}] No available margin, skipping")
            return

        atr = df['atr'].iat[i]
        if np.isnan(atr):
            logging.info(f"[{symbol}] ATR not ready, skipping")
            return

        risk_amount = balance * self.risk_percent
        stop_distance = self.atr_mult * atr
        qty = risk_amount / stop_distance
        min_qty_map = {'BTC-USDT': 0.001, 'ETH-USDT': 0.01, 'SOL-USDT': 1, 'BNB-USDT': 0.01}
        min_qty = min_qty_map.get(symbol, self.quantity)
        qty = max(min_qty, qty)

        required_margin = qty * price / self.leverage
        if required_margin > balance:
            logging.warning(f"[{symbol}] Qty too large for margin, reducing")
            qty = (balance * self.leverage / price) * 0.95

        rsi = df['rsi'].iat[i]
        st_k = df['stoch_k'].iat[i]
        st_d = df['stoch_d'].iat[i]
        macd_up = self.macd_cross_up(df, i)
        macd_dn = self.macd_cross_down(df, i)
        above_ema = price > df['ema'].iat[i]
        below_ema = price < df['ema'].iat[i]

        long_signal = ((rsi <= self.rsi_os and st_k > st_d and above_ema) or macd_up)
        short_signal = ((rsi >= self.rsi_ob and st_k < st_d and below_ema) or macd_dn)

        if long_signal or short_signal:
            tp = price + self.rr * atr if long_signal else price - self.rr * atr
            sl = price - self.atr_mult * atr if long_signal else price + self.atr_mult * atr
            side = 'long' if long_signal else 'short'

            try:
                resp = self.client.place_market_order(side, qty, symbol, stop=sl, tp=tp)
                self.positions[symbol] = side.upper()
                logging.info(f"[{symbol}] Enter {side.upper()} at {price:.2f} | QTY={qty:.4f} | ATR={atr:.2f} | TP={tp:.2f} | SL={sl:.2f} | Resp={resp}")
            except Exception as e:
                logging.error(f"[ERROR] {symbol} trade failed: {e}")
        else:
            logging.info(f"[{symbol}] No signal | RSI={rsi:.2f} | Stoch={st_k:.2f}/{st_d:.2f} | EMA={df['ema'].iat[i]:.2f} | Price={price:.2f}")

    def run(self):
        logging.info("[RUN] Initializing data for symbols...")
        for s in self.symbols:
            self.initialize_data(s)

        logging.info("[RUN] Entering main loop...")

        scheduler = BackgroundScheduler()

        for s in self.symbols:
            scheduler.add_job(
                lambda sym=s: self.update_positions(sym),
                'interval',
                seconds=60
            )
            scheduler.add_job(
                lambda sym=s: self.trade_logic(sym),
                'interval',
                seconds=self.poll_interval
            )

        scheduler.start()

        # ⬇⬇⬇ ВАЖНО ⬇⬇⬇
        while True:
            time.sleep(1)



from websocket import create_connection


class WebSocketHandler:
    def __init__(self, trader: RealTimeTrader):
        self.trader = trader
        self.ws = None
        self.ping_timer = None

    def start(self):
        url = "wss://open-api-swap.bingx.com/swap/ws"
        while True:
            try:
                self.ws = create_connection(url)
                self.on_open()

                while True:
                    message = self.ws.recv()
                    if message:
                        self.on_message(message)

            except Exception as e:
                logging.error(f"WS error: {e}")

                if self.ping_timer:
                    self.ping_timer.cancel()

                if self.ws:
                    try:
                        self.ws.close()
                    except:
                        pass

                time.sleep(5)

    def on_open(self):
        for symbol in self.trader.symbols:
            sub = {"id": "1", "reqType": "sub", "dataType": f"{symbol}@kline_{self.trader.interval}"}
            self.ws.send(json.dumps(sub))
        self.start_ping()

    def start_ping(self):
        def ping():
            try:
                self.ws.send(json.dumps({"ping": int(time.time()*1000)}))
            except Exception as e:
                logging.error(f"Ping error: {e}")
                return
            self.ping_timer = Timer(30, ping)
            self.ping_timer.start()
        ping()

    def on_message(self, message: str):
        try:
            data = json.loads(message)
            if 'dataType' in data and 'kline' in data['dataType']:
                symbol = data['dataType'].split('@')[0]
                kline = data['data'][0]
                new_row = pd.Series({
                    'ts': pd.to_datetime(kline['time'], unit='ms'),
                    'open': float(kline['open']),
                    'high': float(kline['high']),
                    'low': float(kline['low']),
                    'close': float(kline['close']),
                    'vol': float(kline['volume'])
                })
                self.trader.data[symbol] = pd.concat([self.trader.data[symbol], new_row.to_frame().T]).tail(self.trader.history_lookback)
                logging.info(f"[DATA] {symbol} new candle {new_row['ts']}")
                self.trader.trade_logic(symbol)
        except Exception as e:
            logging.error(f"WS message error: {e}")

if __name__ == '__main__':
    trader = RealTimeTrader()
    trader.run()