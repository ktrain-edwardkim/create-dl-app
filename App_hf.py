from flask import Flask, request, jsonify
import torch
from transformers import ViTForImageClassification, ViTImageProcessor

from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)

# Load model
model_name = "./models/vit"
model = ViTForImageClassification.from_pretrained(model_name, return_dict=False)

model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device).eval()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    image_url = data.get("url")

    if not image_url:
        return jsonify({'error': 'No image URL provided'}), 400

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))

        inputs = processor(images=image, return_tensors="pt")
        inputs.to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_class_idx = torch.argmax(torch.stack(list(outputs), dim=0), dim=-1)[0].item()
        return jsonify({'predicted_class': model.config.id2label[predicted_class_idx]})

    except requests.RequestException as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
