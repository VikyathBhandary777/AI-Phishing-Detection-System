from flask import Flask, render_template, request
import os
import pickle
import numpy as np

app = Flask(__name__)

# ==================================
# Model Path (Render Compatible)
# ==================================

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


# ==================================
# Feature Extraction Function
# ==================================

def extract_features(url):
    # Feature 1: URL Length
    url_length = len(url)

    # Feature 2: HTTPS
    has_https = 1 if url.startswith("https") else 0

    # Feature 3: Dot count (subdomains)
    dot_count = url.count(".")

    # Feature 4: @ symbol presence
    has_at = 1 if "@" in url else 0

    # Feature 5: Suspicious keywords
    suspicious_words = ["login", "verify", "secure", "update", "free", "bank"]
    has_suspicious = 1 if any(word in url.lower() for word in suspicious_words) else 0

    return np.array([[url_length, has_https, dot_count, has_at, has_suspicious]])


# ==================================
# Routes
# ==================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        url = request.form["url"]

        if model is None or scaler is None:
            return render_template("index.html",
                                   prediction_text="Model not loaded properly.")

        # Extract Features
        features = extract_features(url)

        # Scale
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[0]

        confidence = round(max(probability) * 100, 2)

        if prediction[0] == 1:
            result = f"✅ Legitimate Website ({confidence}% confidence)"
        else:
            result = f"⚠️ Phishing Website ({confidence}% confidence)"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"Error: {str(e)}")


# ==================================
# Run Locally
# ==================================

if __name__ == "__main__":
    app.run(debug=True)
