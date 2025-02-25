# import yfinance as yf

# def fetch_stock_data(stock_symbol):
#     """Fetch latest stock data from Yahoo Finance."""
#     stock = yf.Ticker(stock_symbol)
#     hist = stock.history(period="5d")

#     if hist.empty:
#         return None
    
#     last_close_price = hist["Close"].iloc[-1]
#     return {
#         "symbol": stock_symbol,
#         "last_close_price": round(last_close_price, 2)
#     }

import yfinance as yf

def fetch_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1d")  # Fetch today's data

        if hist.empty:
            print(f"⚠️ No stock data found for {symbol}")
            return None

        return {
            "open": hist["Open"].values[-1],
            "high": hist["High"].values[-1],
            "low": hist["Low"].values[-1],
            "close": hist["Close"].values[-1],
            "volume": hist["Volume"].values[-1],
            "last_close_price": hist["Close"].values[-1],
        }
    except Exception as e:
        print(f"❌ Error fetching stock data for {symbol}: {e}")
        return None
