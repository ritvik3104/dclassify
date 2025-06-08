from flask import Flask, request, render_template, jsonify
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import base64
import cv2
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Load trained Keras model
model = tf.keras.models.load_model('best_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

def process_transparent_image(image_data):
    if isinstance(image_data, str):
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
    else:
        image_bytes = image_data

    image = Image.open(io.BytesIO(image_bytes))
    white_background = Image.new('RGBA', image.size, (255, 255, 255, 255))
    white_background.paste(image, (0, 0), image)
    image_rgb = white_background.convert('RGB')
    gray_image = image_rgb.convert('L')
    resized_image = gray_image.resize((28, 28), Image.LANCZOS)
    image_array = np.array(resized_image)
    normalized_array = image_array.astype('float32') / 255.0
    normalized_array = np.expand_dims(normalized_array, axis=-1)
    final_image = np.expand_dims(normalized_array, axis=0)
    return final_image

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']
    processed_image = process_transparent_image(image_data)
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    prediction = model.predict(processed_image)
    classes = ['banana', 'basketball', 'ladder']
    predicted_class = np.argmax(prediction, axis=1)[0]
    probabilities = (prediction[0] * 100).round(2).tolist()
    return jsonify({'prediction': classes[predicted_class], 'probabilities': probabilities})

load_dotenv()
port = os.getenv('PORT')

if __name__ == '__main__':
    app.run(host=os.getenv('HOST'), port=port)
