from dotenv import load_dotenv
from flask import Flask, request, jsonify
import pandas as pd
import datetime
from prophet import Prophet
import yfinance as yf
load_dotenv()

app = Flask(__name__)

def predict_crypto_movement(date: str, crypto: str = 'BTC-USD', period: str = '5y'):
    try:
        # Fetch historical data
        df = yf.download(crypto, period=period)
        if df.empty:
            raise ValueError(f"No data found for {crypto} with period {period}.")

        df.columns = [col.split()[0] for col in df.columns]
        df['Date'] = pd.to_datetime(df.index)
        df = df.set_index('Date')
        data = df[['Close', 'Volume']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y', 'Volume': 'volume'})
        data['ds'] = data['ds'].dt.tz_localize(None)

        model = Prophet(
            changepoint_prior_scale=0.05,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative',
        )
        model.add_regressor('volume')
        model.fit(data)

        future_date = pd.to_datetime(date, utc=True)
        future = pd.DataFrame({'ds': [future_date]})
        future['ds'] = future['ds'].dt.tz_localize(None)
        future['volume'] = df['Volume'].iloc[-1]

        forecast = model.predict(future)
        return forecast['yhat'].values[0]

    except Exception as e:
        return f"Error: {e}"

@app.route('/predict', methods=['GET'])
def predict():
    date = request.args.get('date')
    crypto = request.args.get('crypto', 'BTC-USD')
    period = request.args.get('period', '5y')
    try:
        predicted_price = predict_crypto_movement(date, crypto, period)
        return jsonify({"predicted_price": predicted_price})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
