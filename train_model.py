import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ================= LOAD DATA ================= #

df = pd.read_csv("student_data.csv")

X = df[[
    "CGPA",
    "Attendance",
    "Backlogs",
    "ParentalIncome",
    "BehaviorRating",
    "PerformanceTrend"
]]

y = df["RiskLabel"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 🔥 SPLIT FIRST (VERY IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# 🔥 SCALE AFTER SPLIT
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

input_size = 6
num_classes = len(np.unique(y_encoded))

# ================= MODEL ================= #

class AdvancedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = AdvancedMLP()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ================= TRAINING ================= #

epochs = 150

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# ================= EVALUATION ================= #

with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
f1 = f1_score(y_test.numpy(), predicted.numpy(), average="weighted")
precision = precision_score(y_test.numpy(), predicted.numpy(), average="weighted")
recall = recall_score(y_test.numpy(), predicted.numpy(), average="weighted")

print("Accuracy:", accuracy)
print("F1:", f1)
print("Precision:", precision)
print("Recall:", recall)

# ================= SAVE MODEL ================= #

torch.save(model.state_dict(), "model/mlp_model.pt")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")

# ================= SAVE METRICS ================= #

metrics = {
    "accuracy": round(accuracy * 100, 2),
    "f1_score": round(f1 * 100, 2),
    "precision": round(precision * 100, 2),
    "recall": round(recall * 100, 2)
}

with open("model/metrics.json", "w") as f:
    json.dump(metrics, f)

# ================= BAR CHART ================= #

labels = ["Accuracy", "F1 Score", "Precision", "Recall"]
values = [
    metrics["accuracy"],
    metrics["f1_score"],
    metrics["precision"],
    metrics["recall"]
]

plt.figure(figsize=(8,5))
plt.bar(labels, values)
plt.ylim(0, 100)
plt.ylabel("Percentage")
plt.title("Model Performance Metrics")

for i, v in enumerate(values):
    plt.text(i, v + 1, str(v) + "%", ha='center')

plt.savefig("static/images/model_metrics_bar.png")
plt.close()

print("Model, metrics, and chart saved successfully.")
