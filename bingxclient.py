import requests
import hmac
import hashlib
import time
import json

class BingxClient:
    BASE_URL = "https://open-api.bingx.com"

    def __init__(self, api_key: str, api_secret: str, symbol: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = self._to_bingx_symbol(symbol) if symbol else None
        self.time_offset = self.get_server_time_offset()

    def _to_bingx_symbol(self, symbol: str) -> str:
        return symbol.replace("USDT", "-USDT")

    def _sign(self, query: str) -> str:
        return hmac.new(self.api_secret.encode("utf-8"),
                        query.encode("utf-8"),
                        hashlib.sha256).hexdigest()

    def _request(self, method: str, path: str, params=None):
        if params is None:
            params = {}
        sorted_keys = sorted(params)
        query = "&".join([f"{k}={params[k]}" for k in sorted_keys])
        signature = self._sign(query)
        url = f"{self.BASE_URL}{path}?{query}&signature={signature}"
        headers = {"X-BX-APIKEY": self.api_key}
        r = requests.request(method, url, headers=headers)
        r.raise_for_status()
        return r.json()

    def _public_request(self, path: str, params=None, timeout: int = 10):
        url = f"{self.BASE_URL}{path}"
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def get_server_time_offset(self):
        url = f"{self.BASE_URL}/openApi/swap/v2/server/time"
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        if data.get("code") == 0:
            server_time = int(data["data"]["serverTime"])
            local_time = int(time.time() * 1000)
            return server_time - local_time
        return 0

    # ============= Полезные методы =============

    def get_mark_price(self, symbol=None):
        path = "/openApi/swap/v2/quote/premiumIndex"
        s = symbol or self.symbol
        params = {'symbol': s}
        try:
            data = self._public_request(path, params)
            if data.get('code') == 0 and 'data' in data:
                # проверяем, что data это список
                if isinstance(data['data'], list) and len(data['data']) > 0:
                    mark_price = data['data'][0].get('markPrice')
                    return float(mark_price) if mark_price is not None else None
                elif isinstance(data['data'], dict):
                    mark_price = data['data'].get('markPrice')
                    return float(mark_price) if mark_price is not None else None
            return None
        except Exception as e:
            print(f"Ошибка при получении mark price: {e}")
            return None

    def get_positions(self, symbol=None):
        s = symbol or self.symbol
        return self._request("GET", "/openApi/swap/v2/user/positions", {"symbol": s}).get("data", [])

    def place_market_order(self, side: str, qty: float, symbol: str = None, stop: float = None, tp: float = None):
        side_param = "BUY" if side == "long" else "SELL"
        s = symbol or self.symbol

        params = {
            "symbol": s,
            "side": side_param,
            "positionSide": "LONG" if side == "long" else "SHORT",
            "type": "MARKET",
            "timestamp": int(time.time()*1000) + self.get_server_time_offset(),
            "quantity": qty,
            "recvWindow": 5000,
            "timeInForce": "GTC",
        }

        # добавляем стоп, если указан
        if stop is not None:
            stopLoss_param = {
                "type": "STOP_MARKET",
                "stopPrice": stop,
                "price": stop,
                "workingType": "MARK_PRICE"
            }
            params["stopLoss"] = json.dumps(stopLoss_param)

        # добавляем тейк, если указан
        if tp is not None:
            takeProfit_param = {
                "type": "TAKE_PROFIT_MARKET",
                "stopPrice": tp,
                "price": tp,
                "workingType": "MARK_PRICE"
            }
            params["takeProfit"] = json.dumps(takeProfit_param)

        return self._request("POST", "/openApi/swap/v2/trade/order", params)


    def set_leverage(self, symbol, side, leverage):
        path = "/openApi/swap/v2/trade/leverage"
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "leverage": leverage,
            "timestamp": int(time.time()*1000) + self.time_offset
        }
        return self._request("POST", path, params)

    def get_balance(self):
        path = "/openApi/swap/v2/user/balance"
        params = {
            "timestamp": int(time.time()*1000) + self.time_offset
        }
        data = self._request("GET", path, params)
        return data.get("data", {})
