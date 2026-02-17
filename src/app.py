from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Safe path loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models", "phishing_model.pkl")
scaler_path = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
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

if __name__ == "__main__":
    app.run(debug=True)
