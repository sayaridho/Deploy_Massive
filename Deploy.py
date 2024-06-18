from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Memuat model Keras
model = load_model("model3.h5")

def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
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
        processed_image = prepare_image(image, target_size=(224, 224))  # Sesuaikan dengan ukuran input model Anda
        prediction = model.predict(processed_image).tolist()
        print("Prediction made")
        return jsonify({"prediction": prediction})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
