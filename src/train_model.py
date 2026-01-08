import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# STEP 1: Get correct project path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# STEP 2: Build dataset path safely
dataset_path = os.path.join(BASE_DIR, "..", "dataset", "network_traffic.csv")

# STEP 3: Load dataset
data = pd.read_csv(dataset_path)

# STEP 4: Encode protocol column
encoder = LabelEncoder()
data["protocol"] = encoder.fit_transform(data["protocol"])

# STEP 5: Split features and label
X = data.drop("label", axis=1)
y = data["label"]

# STEP 6: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# STEP 8: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# STEP 9: Predict
y_pred = model.predict(X_test)

# STEP 10: Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# STEP 11: Save model
model_path = os.path.join(BASE_DIR, "..", "results", "model", "dos_model.pkl")
joblib.dump(model, model_path)

print("\n✅ Model trained and saved successfully!")
