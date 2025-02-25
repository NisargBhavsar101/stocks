from flask import Flask, render_template, request
import numpy as np
import requests
import warnings
from utils.model_loader import load_model
import pandas as pd
from datetime import datetime, timedelta
import time
from sklearn.metrics import mean_absolute_percentage_error


# Suppress Warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Define stock symbols
stock_names = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL','META', 'TSLA', 'AVGO', 'BRK.B', 'TSM', 'WMT', 'LLY', 'JPM', 'V', 
'MA', 'ORCL', 'UNH', 'XOM', 'COST', 'NFLX', 'HD', 'PG', 'NVO', 'JNJ', 'BAC', 'ABBV', 'SAP', 'CRM', 'ASML', 'TMUS', 'KO', 
'CVX', 'WFC', 'PLTR', 'CSCO', 'TM', 'ACN', 'BABA', 'IBM', 'MS', 'PM', 'ABT', 'AZN', 'AXP', 'GS', 'MRK', 'GE', 'TMO', 'LIN',
'MCD', 'NVS', 'ISRG', 'NOW', 'BX', 'SHEL', 'DIS', 'PEP', 'ADBE', 'HSBC', 'QCOM', 'T', 'CAT', 'AMD', 'ARM', 'RTX', 'RY', 
'VZ', 'TXN', 'INTU', 'BKNG', 'SPGI', 'PDD', 'AMGN', 'UBER', 'BSX', 'BLK', 'C', 'SCHW', 'SHOP', 'HDB', 'ANET', 'SYK', 'DHR',
'PGR', 'AMAT', 'UNP', 'MUFG', 'PFE', 'UL', 'LOW', 'NEE', 'TJX', 'TTE', 'SONY', 'BA', 'HON', 'SNY', 'KKR', 'FI', 'CMCSA']

# Finnhub API Key
FINNHUB_API_KEY = "cuu52apr01qv6ijlpde0cuu52apr01qv6ijlpdeg"  # Replace with your valid key
BASE_URL = "https://finnhub.io/api/v1"

# Get Unix Timestamp
def get_unix_timestamp(days_ago=0):
    date = datetime.now() - timedelta(days=days_ago)
    return int(time.mktime(date.timetuple()))

# Fetch Stock Data
def fetch_stock_data(stock_symbol):
    try:
        headers = {'X-Finnhub-Token': FINNHUB_API_KEY}
        
        # Get real-time quote
        quote_url = f"{BASE_URL}/quote?symbol={stock_symbol}"
        quote_response = requests.get(quote_url, headers=headers)
        quote_data = quote_response.json()

        if 'c' not in quote_data:
            print(f"❌ No data found for {stock_symbol}")
            return None

        # Get company profile
        profile_url = f"{BASE_URL}/stock/profile2?symbol={stock_symbol}"
        profile_response = requests.get(profile_url, headers=headers)
        profile_data = profile_response.json()

        company_name = profile_data.get('name', stock_symbol)

        stock_data = {
            "open": float(quote_data.get('o', 0.0)),
            "high": float(quote_data.get('h', 0.0)),
            "low": float(quote_data.get('l', 0.0)),
            "close": float(quote_data.get('c', 0.0)),
            "volume": int(quote_data.get('v', 0)),
            "company_name": company_name
        }
        return stock_data

    except Exception as e:
        print(f"❌ Error fetching data for {stock_symbol}: {e}")
        return None

# Fetch Historical Data
def fetch_historical_data(symbol, days=180):
    try:
        end_timestamp = get_unix_timestamp()
        start_timestamp = get_unix_timestamp(days)

        headers = {'X-Finnhub-Token': FINNHUB_API_KEY}
        candle_url = f"{BASE_URL}/stock/candle?symbol={symbol}&resolution=D&from={start_timestamp}&to={end_timestamp}"
        response = requests.get(candle_url, headers=headers)
        data = response.json()

        if data.get('s') != 'ok':
            return None, None

        dates = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in data['t']]
        prices = data['c']
        return dates, prices
    except Exception as e:
        print(f"❌ Error fetching historical data for {symbol}: {e}")
        return None, None

def calculate_accuracy(predicted_price, actual_price):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    if actual_price == 0:
        return "N/A"
    error = abs((actual_price - predicted_price) / actual_price) * 100
    return round(100 - error, 2)  # Accuracy as percentage

@app.route("/")
def index():
    stock_predictions = []
    stocks_per_page = 10
    page = request.args.get("page", 1, type=int)

    for stock_symbol in stock_names:
        print(f"\n🔍 Processing {stock_symbol}")
        time.sleep(1)  # Respect API rate limit

        stock_data = fetch_stock_data(stock_symbol)
        if not stock_data:
            continue

        last_close_price = stock_data.get("close", 0.0)
        features = np.array([
            stock_data.get("open", 0.0),
            stock_data.get("high", 0.0),
            stock_data.get("low", 0.0),
            stock_data.get("close", 0.0),
            stock_data.get("volume", 0.0)
        ]).reshape(1, 5)

        try:
            model, feature_scaler, target_scaler = load_model(stock_symbol)
        except Exception as e:
            print(f"❌ Error loading model for {stock_symbol}: {e}")
            continue

        predicted_price = None
        recommendation = "HOLD"
        accuracy = "N/A"

        try:
            scaled_features = feature_scaler.transform(features)
            repeated_data = np.tile(scaled_features, (60, 1))
            reshaped_input = np.array([repeated_data])

            scaled_pred = model.predict(reshaped_input)[0][0]
            predicted_price = target_scaler.inverse_transform([[scaled_pred]])[0][0]

            if predicted_price > last_close_price * 1.02:
                recommendation = "BUY"
            elif predicted_price < last_close_price * 0.98:
                recommendation = "SELL"

           # Fetch actual price for next day (if available)
            actual_price = fetch_stock_data(stock_symbol).get("close", None)

            # Compute accuracy if actual price is available
            accuracy = "N/A"
            if actual_price:
                accuracy = calculate_accuracy(predicted_price, actual_price)
        except Exception as e:
            print(f"❌ Error predicting {stock_symbol}: {e}")

        stock_predictions.append({
            "symbol": stock_symbol,
            "company_name": stock_data["company_name"],
            "last_close_price": round(last_close_price, 2),
            "predicted_price": round(predicted_price, 2) if predicted_price is not None else "N/A",
            "recommendation": recommendation,
            "accuracy": f"{accuracy}%" if accuracy != "N/A" else "N/A"
        })

    total_stocks = len(stock_predictions)
    start = (page - 1) * stocks_per_page
    end = start + stocks_per_page
    paginated_stocks = stock_predictions[start:end]
    total_pages = (total_stocks + stocks_per_page - 1) // stocks_per_page

    return render_template("index.html", stocks=paginated_stocks, page=page, total_pages=total_pages)

@app.route("/stock/<symbol>")
def stock_detail(symbol):
    dates, prices = fetch_historical_data(symbol)
    if not dates or not prices:
        return "Stock data not found", 404

    model, feature_scaler, target_scaler = load_model(symbol)
    if model is None:
        return "Model or scalers not found for this stock", 404

    stock_data = {
        "symbol": symbol,
        "company_name": symbol,  
        "last_close_price": round(prices[-1], 2),
        "dates": dates,
        "prices": prices
    }

    return render_template("stock_detail.html", stock=stock_data)

if __name__ == "__main__":
    app.run(debug=True)
