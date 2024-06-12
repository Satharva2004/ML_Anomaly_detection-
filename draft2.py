import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# Step 1: Load the dataset using the full path
file_path = r"C:\Users\sawan\OneDrive\Desktop\Python\Anomaly Detections\kddcup.data_10_percent_corrected"
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

# Step 4: Separate features and labels
X = df.drop("label", axis=1)
y = df["label"].apply(lambda x: 1 if x == "normal." else -1)

# Step 5: Feature Selection
selector = SelectKBest(f_classif, k=20)  # Experiment with different k values
X_new = selector.fit_transform(X, y)

# Step 6: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_new)

# Step 7: Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Step 8: Hyperparameter Tuning using GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300, 400, 500],  # Expand the search space
    "max_samples": [0.8, 1.0],
    "contamination": [0.01, 0.05, 0.1],  # Add more contamination levels
    "max_features": [0.8, 1.0],
}
grid = GridSearchCV(
    IsolationForest(random_state=42), param_grid, scoring="f1", cv=3, n_jobs=-1
)
grid.fit(X_train, y_train)

# Best model
model = grid.best_estimator_

grid = GridSearchCV(
    IsolationForest(random_state=42), param_grid, scoring="f1", cv=3, n_jobs=-1
)
grid.fit(X_train, y_train)

# Best model
model = grid.best_estimator_

# Step 9: Train the model
start_time = time.time()
model.fit(X_train)
end_time = time.time()
training_time = end_time - start_time

# Step 10: Make predictions
start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()
prediction_time = end_time - start_time

# Step 11: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()

print(f"Training Time: {training_time:.2f} seconds")
print(f"Prediction Time: {prediction_time:.2f} seconds")

# Step 10: Make predictions
# start_time = time.time()
# y_pred = model.predict(X_test)
# end_time = time.time()
# prediction_time = end_time - start_time

# # Step 11: Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# print("Classification Report:")
# print(classification_report(y_test, y_pred))
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print(f"Training Time: {training_time:.2f} seconds")
# print(f"Prediction Time: {prediction_time:.2f} seconds")

# # Step 12: Visualize anomalies
# # Visualizing anomalies
anomalies = X_test[model.predict(X_test) == -1]

plt.figure(figsize=(10, 6))
plt.scatter(
    X_test[:, 0], X_test[:, 1], label="Normal Data", alpha=0.6
)  # Adding transparency
plt.scatter(
    anomalies[:, 0], anomalies[:, 1], color="red", label="Anomalies", alpha=0.6
)  # Adding transparency
plt.title("Isolation Forest Anomaly Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Setting custom x-axis ticks and labels
plt.xticks(np.arange(0, 30, step=5))

plt.legend()
plt.tight_layout()  # Adjust layout to prevent clipping labels
plt.show()
