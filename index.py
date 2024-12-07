from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import datetime
from prophet import Prophet
import yfinance as yf
import re

load_dotenv()

app = Flask(__name__)
if app.config["NODE_ENV"] == "production":
    CORS(app, resources={
        r"/predict": {
            "origins": ["https://crystalballin.org"],
            "methods": ["POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
            "expose_headers": ["Content-Type"],
            "supports_credentials": True
        }
    })
else:
    CORS(app)  # Open CORS policy for local development

def validate_period(period: str) -> bool:
    return bool(re.match(r'^\d+[ydwh]$', period))

def validate_date(date_str: str) -> bool:
    try:
        pd.to_datetime(date_str)
        return True
    except:
        return False

def predict_crypto_movement(date: str, crypto: str = 'BTC-USD', period: str = '5y'):
    try:
        # Input validation
        if not validate_date(date):
            raise ValueError("Invalid date format. Please use YYYY-MM-DD format.")
        if not validate_period(period):
            raise ValueError("Invalid period format. Use format like '5y', '1y', '7d', etc.")
        if not isinstance(crypto, str) or not re.match(r'^[A-Z0-9-]+$', crypto):
            raise ValueError("Invalid crypto symbol format.")

        # Fetch historical data using Ticker object instead of download
        ticker = yf.Ticker(crypto)
        df = ticker.history(period=period)
        
        if df.empty:
            raise ValueError("No data found for the given parameters.")

        # Prepare data for Prophet
        df = df.copy()
        df.reset_index(inplace=True)
        
        # Replace inf values with NaN
        df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        
        data = pd.DataFrame({
            'ds': pd.to_datetime(df['Date']).dt.tz_localize(None),
            'y': df['Close'],
            'volume': df['Volume']
        })

        # Initialize and fit model
        model = Prophet(
            changepoint_prior_scale=0.05,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative',
        )
        model.add_regressor('volume')
        model.fit(data)

        # Prepare future dataframe for prediction
        future_date = pd.to_datetime(date, utc=True)
        future = pd.DataFrame({'ds': [future_date]})
        future['ds'] = future['ds'].dt.tz_localize(None)
        future['volume'] = df['Volume'].iloc[-1]

        # Generate prediction
        forecast = model.predict(future)
        prediction = float(forecast['yhat'].values[0])
        print(f"Prediction for {date}: {prediction}")
        return prediction

    except Exception as e:
        raise ValueError(f"Prediction error: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "https://crystalballin.org")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST")

    try:
        data = request.get_json()

        print("==================")
        print(data)
        date = data.get('date')
        if not date:
            return jsonify({"error": "Date parameter is required"}), 400
        
        crypto = data.get('crypto', 'BTC-USD')
        period = data.get('period', '5y')
        
        predicted_price = predict_crypto_movement(date, crypto, period)
        return jsonify({
            "crypto": crypto,
            "predicted_price": round(predicted_price, 2),
            "date": date,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)