from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Load the model and necessary preprocessing objects
model = load_model('gru_co2_model.h5')
data = pd.read_csv('climate_change_data.csv')

# Create and fit the encoders and scaler
location_encoder = LabelEncoder()
country_encoder = LabelEncoder()
scaler = StandardScaler()

data['Location_encoded'] = location_encoder.fit_transform(data['Location'])
data['Country_encoded'] = country_encoder.fit_transform(data['Country'])

# Fit the scaler on the encoded data
scaler.fit(data[['Temperature', 'Location_encoded', 'Country_encoded']])

@app.route('/')
def index():
    countries = sorted(data['Country'].unique())
    locations = sorted(data['Location'].unique())
    return render_template('index.html', countries=countries, locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    temperature = float(request.form['temperature'])
    country = request.form['country']
    location = request.form['location']

    # Encode the input
    country_encoded = country_encoder.transform([country])[0]
    location_encoded = location_encoder.transform([location])[0]

    # Scale the input
    input_data = np.array([[temperature, location_encoded, country_encoded]])
    input_scaled = scaler.transform(input_data)
    input_reshaped = np.reshape(input_scaled, (1, 1, 3))

    # Make prediction
    prediction = model.predict(input_reshaped)[0][0]

    return jsonify({
        'prediction': float(prediction)
    })
@app.route('/get_world_data')
def get_world_data():
    # This is a placeholder. You'll need to implement this function
    # to return the actual data from your dataset.
    data = {
        'countries': ['United States', 'China', 'India', 'Russia', 'Japan'],
        'emissions': [5000000, 10000000, 2500000, 1750000, 1250000]
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)