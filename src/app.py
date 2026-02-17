from flask import Flask, render_template, request
import os
import pickle
import numpy as np

app = Flask(__name__)

# ==============================
# Correct Model Path (Render Compatible)
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "phishing_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

try:
    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    print("Model and Scaler Loaded Successfully")
except Exception as e:
    print("Error loading model:", e)
    model = None
    scaler = None


# ==============================
# Home Route
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# Prediction Route
# ==============================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        url = request.form["url"]

        # Feature 1: URL Length
        url_length = len(url)

        # Feature 2: HTTPS Check
        has_https = 1 if url.startswith("https") else 0

        features = np.array([[url_length, has_https]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)

        if prediction[0] == 1:
            result = "✅ Legitimate Website"
        else:
            result = "⚠️ Phishing Website"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


# ==============================
# Run Locally
# ==============================

if __name__ == "__main__":
    app.run(debug=True)
