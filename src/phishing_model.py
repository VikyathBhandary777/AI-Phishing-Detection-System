import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ==================================
# Safe Base Directory
# ==================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "phishing_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

# ==================================
# Synthetic Dataset Generator
# ==================================

def generate_dataset(n_samples=2000):
    X = []
    y = []

    for _ in range(n_samples):
        # Random URL length
        url_length = np.random.randint(10, 120)

        # HTTPS probability
        has_https = np.random.choice([0, 1], p=[0.4, 0.6])

        # Dot count
        dot_count = np.random.randint(1, 6)

        # @ symbol probability
        has_at = np.random.choice([0, 1], p=[0.9, 0.1])

        # Suspicious words probability
        has_suspicious = np.random.choice([0, 1], p=[0.7, 0.3])

        # Simple rule for labeling
        risk_score = (
            (url_length > 70) +
            (has_https == 0) +
            (dot_count > 3) +
            has_at +
            has_suspicious
        )

        label = 0 if risk_score >= 2 else 1
        # 0 = Phishing
        # 1 = Legitimate

        X.append([url_length, has_https, dot_count, has_at, has_suspicious])
        y.append(label)

    return np.array(X), np.array(y)


# ==================================
# Generate Data
# ==================================

print("Generating synthetic dataset...")
X, y = generate_dataset(3000)

print("Dataset Generated Successfully!")
print("Total Samples:", len(X))

# ==================================
# Train-Test Split
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==================================
# Scaling
# ==================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==================================
# Train Model
# ==================================

print("Training model...")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# ==================================
# Save Model
# ==================================

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

pickle.dump(model, open(model_path, "wb"))
pickle.dump(scaler, open(scaler_path, "wb"))

print("Model and Scaler saved successfully!")
print("Training Complete ✅")
