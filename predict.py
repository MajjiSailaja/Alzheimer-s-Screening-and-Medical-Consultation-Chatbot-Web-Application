from flask import Flask, request, render_template
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load model once
model = tf.keras.models.load_model("best_model (4).keras")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    prediction_result = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Open image
            image = Image.open(file.stream).convert("RGB")

            # Resize to 224x224 without distortion
            image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

            # Convert to array
            image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

            # Predict
            prediction = model.predict(image_array)
            classes = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
            prediction_result = classes[np.argmax(prediction)]

    return render_template('prediction.html', prediction=prediction_result)
