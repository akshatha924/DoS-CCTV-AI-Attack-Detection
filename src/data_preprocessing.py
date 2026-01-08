import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1️⃣ Get the path of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2️⃣ Build the dataset path (this works even if folder names have spaces)
dataset_path = os.path.join(BASE_DIR, "../dataset/network_traffic.csv")

# 3️⃣ Load the dataset
data = pd.read_csv(dataset_path)

# 4️⃣ Encode 'protocol' column to numeric
encoder = LabelEncoder()
data['protocol'] = encoder.fit_transform(data['protocol'])

# 5️⃣ Separate features and label
X = data.drop('label', axis=1)
y = data['label']

# 6️⃣ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Print success and first 5 rows
print("Data preprocessing completed successfully!")
print(data.head())
