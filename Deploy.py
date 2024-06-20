from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the Keras model from the directory
model = load_model("Model Hama Sawi.h5")

def preprocess_image(image):
    image = image.resize((128, 128))  # Adjust the target size to match your model's expected input
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalizing the image if needed
    return image

@app.route("/api/hello", methods=["GET"])
def halo():
    return jsonify({"Message": "haii teman"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    print("Request received")
    print(request.files)  # Print the request files to see what is received

    if 'file' not in request.files:
        print("No file part found in the request")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({"error": "No selected file"}), 400

    try:
        print("Processing image")
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image).tolist()
        print("Prediction made")
        return jsonify({"prediction": prediction})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
