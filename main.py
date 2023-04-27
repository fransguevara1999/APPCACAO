import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = None

def load_model():
    global model
    model = tf.keras.models.load_model('my_model.h5')

def preprocess_image(image):
    # Redimensionar la imagen a 505 x 505 pixeles
    image = image.resize((505, 505))
    # Convertir la imagen a un array de NumPy y normalizar los valores de píxeles
    image_array = np.asarray(image) / 255.0
    # Agregar una dimensión adicional para que el modelo pueda procesarla
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(image):
    # Preprocesar la imagen
    image_array = preprocess_image(image)
    # Obtener la predicción del modelo
    prediction = model.predict(image_array)[0]
    # Obtener el nombre de la clase predicha
    class_names = ['inmaduro', 'maduro', 'sobremaduro']
    predicted_class_name = class_names[np.argmax(prediction)]
    # Devolver la predicción en formato JSON
    response = {'class': predicted_class_name, 'confidence': float(prediction[np.argmax(prediction)])}
    return response

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener la imagen enviada en la petición POST
    file = request.files['image']
    # Cargar la imagen usando Pillow
    image = Image.open(file)
    # Obtener la predicción de la imagen
    prediction = predict_image(image)
    # Devolver la predicción en formato JSON
    return jsonify(prediction)

if __name__ == '__main__':
    load_model()
    app.run(port=5000, debug=True)