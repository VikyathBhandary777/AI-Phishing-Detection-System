from flask import Flask, render_template, request
import os
import pickle
import numpy as np

app = Flask(__name__)

# ==============================
# Correct Path Setup (Render + Local)
# ==============================

# Go from src/app.py → project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "phishing_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Load model safely
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
        url_length = float(request.form["url_length"])
        has_https = float(request.form["has_https"])

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
# Run App (Local Only)
# ==============================

if __name__ == "__main__":
    app.run(debug=True)
