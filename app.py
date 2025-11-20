from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from PIL import Image
import requests

app = Flask(__name__)

# Load Models
alz_model = tf.keras.models.load_model(r"C:\Users\sailu\Downloads\best_model (4).keras")
reversibility_model = load_model(r"C:\Users\sailu\Downloads\my_model.h5")

# OLAMMA API endpoint
OLAMMA_API_URL = "http://localhost:11434/api/generate"

# Class Labels
alz_classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# ---------------- ROUTES ---------------- #

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction')
def prediction_page():
    return render_template('prediction.html')

@app.route('/reversibility')
def reversibility_page():
    return render_template('reversibility.html')

@app.route('/explain')
def explain():
    return render_template('explain.html')


# ------------- API ROUTES -------------- #

# Alzheimer's Prediction (POST)
@app.route('/predict', methods=['POST'])
def predict_alzheimers():
    try:
        file = request.files['image']
        image = Image.open(file).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = alz_model.predict(img_array)
        class_index = np.argmax(prediction[0])
        result = alz_classes[class_index]

        return render_template('prediction.html', prediction=result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Reversibility Prediction (POST)
@app.route('/predict_reversibility', methods=['POST'])
def predict_reversibility():
    try:
        file = request.files['image']
        image = Image.open(file).convert('RGB')
        image = image.resize((128, 128))

        # ⚠️ Use correct preprocessing depending on how model was trained
        img_array = np.expand_dims(np.array(image) / 255.0, axis=0)  # For CNN
        # img_array = np.array(image).flatten().reshape(1, -1)        # For flattened input

        result = reversibility_model.predict(img_array)[0]
        return jsonify({'reversibility': result.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Chatbot / Explainable AI
@app.route('/ask', methods=['POST'])
def ask_chatbot():
    try:
        data = request.json
        prompt = data.get("prompt", "")

        response = requests.post(OLAMMA_API_URL, json={
            "model": "mistral:instruct",
            "prompt": prompt,
            "stream": False
        })

        result = response.json()
        message = result.get("response") or result.get("message", {}).get("content", "No reply")
        return jsonify({'response': message})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------- MAIN ---------------- #
if __name__ == '__main__':
    app.run(debug=True)
 