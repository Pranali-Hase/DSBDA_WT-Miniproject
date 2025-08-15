from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load models
model_weather = joblib.load('weather_forecast_model.pkl')
model_disaster = joblib.load('disaster_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    humidity = float(request.form['humidity'])
    precipitation = float(request.form['precipitation'])

    # Predict temperature
    temp_pred = model_weather.predict([[humidity, precipitation]])[0]

    # Predict disaster (need temperature too)
    disaster_pred = model_disaster.predict([[temp_pred, humidity, precipitation]])[0]

    return render_template('index.html',
                           temperature=round(temp_pred, 2),
                           disaster='Yes' if disaster_pred == 1 else 'No')

if __name__ == '__main__':
    app.run(debug=True)
