from flask import Flask, request, jsonify
import torch
import numpy as np
import joblib

from model_pt import LinearRegressionModel

app = Flask(__name__)

# Load model
model = LinearRegressionModel()
model.load_state_dict(torch.load('./models/model.pt'))
model.eval()

# Load the fitted scaler
scaler = joblib.load('./models/scaler.pkl')
# You should fit the scaler to your data beforehand and save the scaler

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Convert data to numpy array and scale it
    data_scaled = scaler.transform(np.array(data['features']).reshape(1, -1))
    # Convert to PyTorch tensor and predict
    data_tensor = torch.tensor(data_scaled, dtype=torch.float)
    prediction = model(data_tensor)
    return jsonify({'prediction': prediction.item()})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
