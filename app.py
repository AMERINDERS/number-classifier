from flask import Flask, render_template, request
import os
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('digits.model.keras')

# Ensure the 'images' folder exists
UPLOAD_FOLDER = './images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')



@app.route('/', methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        return render_template('index.html', error="No file uploaded")

    imagefile = request.files['imagefile']
    if imagefile.filename == '':
        return render_template('index.html', error="No file selected")

    # Save the uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, imagefile.filename)
    imagefile.save(image_path)

    try:
        # Preprocess the image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = ImageOps.invert(img)  # Invert colors (white on black -> black on white)
        img = img.resize((28, 28))  # Resize to 28x28 pixels
        img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_array = img_array.reshape(1, 28, 28)  # Reshape for the model

        

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        confidence = prediction[0][predicted_digit] * 100  # Get confidence as a percentage

        # Delete the uploaded image after processing
        os.remove(image_path)

        return render_template('index.html', success=f"Predicted Digit: {predicted_digit} with {confidence:.2f}% confidence")
    except Exception as e:
        # Ensure the file is deleted even if an error occurs
        if os.path.exists(image_path):
            os.remove(image_path)
        return render_template('index.html', error=f"Error processing image: {e}")

@app.route('/predict_canvas', methods=['POST'])
def predict_canvas():
    try:
        # Get the base64 image data from the form
        image_data = request.form['image']
        image_data = image_data.split(',')[1]  # Remove the "data:image/png;base64," prefix
        

        # Decode the base64 image
        image = Image.open(BytesIO(base64.b64decode(image_data))).convert('L')  # Convert to grayscale
        image = ImageOps.invert(image)  # Invert colors (white on black -> black on white)
        image = image.resize((28, 28))  # Resize to 28x28 pixels
        img_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
        img_array = img_array.reshape(1, 28, 28)  # Reshape for the model

        # Save the preprocessed image for verification
       # preprocessed_image_path = os.path.join(UPLOAD_FOLDER, "preprocessed_canvas_image.png")
        #Image.fromarray((img_array[0] * 255).astype(np.uint8)).save(preprocessed_image_path)

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        confidence = prediction[0][predicted_digit] * 100  # Get confidence as a percentage

        return render_template('index.html', success=f"Predicted Digit: {predicted_digit} with {confidence:.2f}% confidence")
    except Exception as e:
        return render_template('index.html', error=f"Error processing canvas input: {e}")


if __name__ == '__main__':
    app.run(port=3000, debug=True)