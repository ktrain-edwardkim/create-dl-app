from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained RandomForestRegressor model
model = joblib.load('models/model_sk/model_sk.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Convert data to numpy array
    np_data = np.array(data['features']).astype(float)
    # Reshape data for model prediction
    np_data = np_data.reshape(1, -1)
    # Predict the MPG value
    prediction = model.predict(np_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(port=5000, debug=True)