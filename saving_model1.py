import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import shap
import pickle
import os

# Load the dataset using the full path
file_path = r"C:\Users\sawan\OneDrive\Desktop\Python\Anomaly Detections\kddcup.data_10_percent_corrected"

# Load the dataset with the correct delimiter
df = pd.read_csv(file_path, header=None, delimiter=",")

# Assign column names
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

# Encode categorical variables
df = pd.get_dummies(df, columns=["protocol_type", "service", "flag"], drop_first=True)

# Separate features and labels
X = df.drop("label", axis=1)
y = df["label"].apply(lambda x: 1 if x == "normal." else -1)

# Select top features based on SHAP values
top_features = [53, 28, 84, 19, 38]
X_top_features = X.iloc[:, top_features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_top_features)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.3, random_state=42
)

# Parameter tuning with GridSearchCV
param_grid = {
    "n_estimators": [100, 150, 200],
    "max_samples": [0.75, 1.0],
    "contamination": [0.01, 0.05],
    "max_features": [0.5, 1.0],
    "bootstrap": [False, True],
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=IsolationForest(random_state=42),
    param_grid=param_grid,
    scoring="f1",
    n_jobs=-1,
    cv=StratifiedKFold(n_splits=5),
    verbose=1,
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get best parameters
best_params = grid_search.best_params_
print("Best parameters found by GridSearchCV:")
print(best_params)

# Train the model with best parameters
model = IsolationForest(**best_params, random_state=42)
start_time = time.time()
model.fit(X_train)
end_time = time.time()
training_time = end_time - start_time

# Save the trained model
with open("isolation_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Make predictions
start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()
prediction_time = end_time - start_time

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"Training Time: {training_time:.2f} seconds")
print(f"Prediction Time: {prediction_time:.2f} seconds")

# Feature Importance using SHAP
# explainer = shap.Explainer(model, X_train)
# shap_values = explainer(X_train)
# shap.summary_plot(shap_values, X_train, plot_type="bar")

# Visualize anomalies
anomalies = X_test[model.predict(X_test) == -1]

plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], label="Normal Data", alpha=0.6)
plt.scatter(anomalies[:, 0], anomalies[:, 1], color="red", label="Anomalies", alpha=0.6)
plt.title("Isolation Forest Anomaly Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.xticks(np.arange(0, 60, step=1))
plt.tight_layout()
plt.legend()
plt.show()
