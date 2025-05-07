

Number Classifier
A web-based application that allows users to draw or upload images of handwritten digits and predicts the digit using a trained neural network model.

Features
Canvas Drawing: Users can draw digits directly on a canvas using their mouse.
Image Upload: Users can upload images of handwritten digits for classification.
Real-Time Predictions: The application processes the input and predicts the digit with a confidence percentage.
Preprocessing Pipeline: Images are preprocessed to match the MNIST dataset format (28x28 pixels, grayscale, black digit on a white background).
Interactive UI: A clean and responsive interface for easy interaction.
Technologies Used
Frontend:
HTML5, CSS3
JavaScript (Canvas API)
Bootstrap (for styling)
Backend:
Flask (Python web framework)
TensorFlow/Keras (for the trained neural network model)
Pillow (for image processing)
Model:
A neural network trained on the MNIST dataset to classify digits (0–9).

Drawing on Canvas:

Users draw a digit on the canvas.
The drawing is captured as a Base64-encoded image and sent to the backend for processing.
Image Upload:

Users can upload an image of a handwritten digit.
The image is preprocessed to match the MNIST dataset format.
Prediction:

The preprocessed image is fed into a trained neural network model.
The model predicts the digit and returns the result along with the confidence percentage.
