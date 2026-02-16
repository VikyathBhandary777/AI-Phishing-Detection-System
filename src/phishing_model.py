# ==========================================
# AI PHISHING DETECTION SYSTEM
# ==========================================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1️⃣ Load Dataset
data = pd.read_csv("../data/phishing_dataset.csv")

print("Dataset Loaded Successfully ✅")
print("Shape:", data.shape)
print("Column Names:", data.columns)

# 2️⃣ Select Target Automatically
target_column = data.columns[-1]
print("Target Column Detected:", target_column)

X = data.drop(target_column, axis=1)
y = data[target_column]

# 3️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5️⃣ Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6️⃣ Evaluate Model
y_pred = model.predict(X_test_scaled)

print("\n===== MODEL PERFORMANCE =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nTraining Accuracy:", model.score(X_train_scaled, y_train))
print("Testing Accuracy:", model.score(X_test_scaled, y_test))

# 7️⃣ Save Model
pickle.dump(model, open("../models/phishing_model.pkl", "wb"))
pickle.dump(scaler, open("../models/scaler.pkl", "wb"))

print("\nModel Saved Successfully ✅")

# ==========================================
# 8️⃣ Test Model on New Website
# ==========================================

print("\n===== TEST NEW WEBSITE =====")

# Example new website input
# Format: [url_length, has_https]
new_data = [[75, 1]]

new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)

if prediction[0] == 1:
    print("Prediction: Legitimate Website ✅")
else:
    print("Prediction: Phishing Website ⚠️")
