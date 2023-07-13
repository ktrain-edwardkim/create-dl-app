from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


app = Flask(__name__)

model = tf.keras.models.load_model('./models/model_tf')
model.summary()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Convert data to numpy array
    np_data = np.array(data['features']).astype(float).reshape(-1, 9)
    print(type(np_data), np_data.shape)
    # Convert to PyTorch tensor and predict

    prediction = model.predict(np_data)
    print(type(prediction), prediction)
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
