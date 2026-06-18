import base64
import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image

app = Flask(__name__)

weights = np.load("weights.npz")
W1 = weights["W1"]
b1 = weights["b1"]
W2 = weights["W2"]
b2 = weights["b2"]


def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(X):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return a2


def preprocess_canvas(data_url: str) -> np.ndarray:
    """Convert a base64 canvas PNG to a normalised 784-dim vector matching MNIST format."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)

    img = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale
    img = img.resize((28, 28), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)

    # Canvas: white background (255), dark stroke. MNIST: black bg (0), white digit.
    arr = 255.0 - arr
    arr = arr / 255.0

    return arr.reshape(1, 784)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    data_url = body.get("image")
    if not data_url:
        return jsonify({"error": "No image provided"}), 400

    X = preprocess_canvas(data_url)
    probs = forward(X)[0]  # shape (10,)

    digit = int(np.argmax(probs))
    confidence = float(probs[digit])

    return jsonify({
        "digit": digit,
        "confidence": round(confidence * 100, 1),
        "probabilities": [round(float(p) * 100, 1) for p in probs],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
