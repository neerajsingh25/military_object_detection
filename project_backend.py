import numpy as np
import cv2

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import Image

app = Flask(__name__)

# Load the model
model = load_model('m1.h5')

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input image from the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in the request'})

    image_file = request.files['image']

    # Save the uploaded image to a temporary file
    temp_file_path = 'temp_image.jpg'
    image_file.save(temp_file_path)

    # Read the image file using PIL
    #img = Image.open(temp_file_path)
    #img = img.resize((224, 224))

    img = cv2.imread(temp_file_path)
    img = cv2.resize(img, (224, 224))

    # Preprocess the image
    #x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)

    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)

    # Get the model's prediction
    preds = model.predict(x)
    preds

    ans = preds
    if ans[0][0] > ans[0][1]:
        ans = "aeroplane"
    else:
        ans = "tank"
    

    # Return the prediction as a JSON response
    response = {
        'prediction': ans
    }
    #return preds
    return jsonify(response)


@app.route('/')
def index():
    return "Welcome to the app"


if __name__ == '__main__':
    # Start the Flask app
    app.run()
