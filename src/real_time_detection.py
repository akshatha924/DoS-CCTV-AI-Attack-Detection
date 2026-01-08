import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# STEP 1: Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# STEP 2: Load trained model
model_path = os.path.join(BASE_DIR, "..", "results", "model", "dos_model.pkl")
model = joblib.load(model_path)

# STEP 3: Simulated real-time CCTV network traffic
# Format: [packet_rate, byte_rate, avg_packet_size, src_ip_count, protocol]
# protocol: TCP=0, UDP=1, ICMP=2

real_time_data = np.array([[950, 260000, 1100, 60, 1]])

# STEP 4: Scale input (same as training)
scaler = StandardScaler()
real_time_data_scaled = scaler.fit_transform(real_time_data)

# STEP 5: Prediction
prediction = model.predict(real_time_data_scaled)

# STEP 6: Output result
if prediction[0] == 1:
    print("🚨 ALERT: DoS Attack Detected in CCTV Network!")
else:
    print("✅ Normal Network Traffic")
