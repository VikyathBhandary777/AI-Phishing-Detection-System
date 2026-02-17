import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ==============================
# Load Dataset
# ==============================

data = pd.read_csv("data/phishing_data.csv")

# ==============================
# Feature Engineering
# ==============================

def extract_features(url):
    url_length = len(url)
    has_https = 1 if url.startswith("https") else 0
    dot_count = url.count(".")
    has_at = 1 if "@" in url else 0
    suspicious_words = ["login", "verify", "secure", "update", "free", "bank"]
    has_suspicious = 1 if any(word in url.lower() for word in suspicious_words) else 0

    return [url_length, has_https, dot_count, has_at, has_suspicious]


# Apply feature extraction
data["features"] = data["url"].apply(extract_features)

X = np.array(data["features"].tolist())
y = data["label"]

# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# Scaling
# ==============================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# Train Model
# ==============================

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# ==============================
# Save Model
# ==============================

os.makedirs("models", exist_ok=True)

pickle.dump(model, open("models/phishing_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

print("Model and Scaler saved successfully!")
