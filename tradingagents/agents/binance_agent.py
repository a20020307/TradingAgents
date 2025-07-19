from binance.client import Client

class BinanceTradingAgent:
    def __init__(self, api_key, api_secret, symbol="BTCUSDT", testnet=True):
        self.symbol = symbol
        self.client = Client(api_key, api_secret)
        if testnet:
            self.client.API_URL = 'https://testnet.binancefuture.com/fapi'
        super().__init__()

    def get_price(self):
        ticker = self.client.futures_symbol_ticker(symbol=self.symbol)
        return float(ticker['price'])

    def place_order(self, side='BUY', quantity=0.001):
        order = self.client.futures_create_order(
            symbol=self.symbol,
            side=side,
            type='MARKET',
            quantity=quantity
        )
        return order

