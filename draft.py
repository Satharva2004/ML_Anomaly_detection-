import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the dataset using the full path
file_path = r"C:\Users\sawan\OneDrive\Desktop\Python\Anomaly Detections\kddcup.data_10_percent_corrected"

# Load the dataset with the correct delimiter
df = pd.read_csv(file_path, header=None, delimiter=",")

# Step 2: Assign column names
columns = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
]

df.columns = columns

# Step 3: Encode categorical variables
df = pd.get_dummies(df, columns=["protocol_type", "service", "flag"], drop_first=True)


print(df)
# Step 4: Separate features and labels
X = df.drop("label", axis=1)
y = df["label"].apply(lambda x: 1 if x == "normal." else -1)

# Step 5: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Step 7: Train the model
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 10: Visualize anomalies
# Visualizing anomalies
anomalies = X_test[model.predict(X_test) == -1]

plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], label="Normal Data", alpha=0.6)
plt.scatter(anomalies[:, 0], anomalies[:, 1], color="red", label="Anomalies", alpha=0.6)
plt.title("Isolation Forest Anomaly Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Setting custom x-axis ticks and labels
plt.xticks(np.arange(0, 60, step=1))

plt.tight_layout()
plt.legend()
plt.show()
