# parkinson_fnn_train.py
# --------------------------------------------
# Feedforward Neural Network for Parkinson’s Voice Classification
# Author: Suchismita Batabyal, Virginia Tech ECE Dept
# --------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np

# 1. Load Dataset
# Replace 'parkinsons.csv' with your dataset path (UCI PD voice dataset)
df = pd.read_csv("parkinsons.csv")

# Target column is 'status': 1 = PD, 0 = Healthy
X = df.drop(columns=["name", "status"], errors="ignore").values
y = df["status"].values

# 2. Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


# 3. Define FNN Model
class ParkinsonFNN(nn.Module):
    def __init__(self, input_dim):
        super(ParkinsonFNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


model = ParkinsonFNN(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}")

# 5. Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = (y_pred > 0.5).float()

acc = accuracy_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)
cm = confusion_matrix(y_test, y_pred_class)

print("\nModel Evaluation:")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:\n", cm)

# 6. Save Model
torch.save(model.state_dict(), "fnn_parkinson_model.pth")
print("\nModel saved as 'fnn_parkinson_model.pth'")
