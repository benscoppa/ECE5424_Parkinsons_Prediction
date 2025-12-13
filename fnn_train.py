# fnn_train.py
# --------------------------------------------
# Feedforward Neural Network for Parkinson’s Voice Classification
# Author: Suchismita Batabyal, Ben Scoppa, Virginia Tech ECE Dept
# --------------------------------------------

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


def load_and_preprocess(
    csv_path: str = "parkinsons.csv",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Loads Parkinsons dataset, standardizes features, and creates train/test split.

    Returns:
      X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler
    """
    df = pd.read_csv(csv_path)

    # Target column is 'status': 1 = PD, 0 = Healthy
    X = df.drop(columns=["name", "status"], errors="ignore").values
    y = df["status"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler


def train_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    input_dim: int,
    epochs: int = 200,
    lr: float = 0.001,
    print_every: int = 10,
):
    """
    Trains the ParkinsonFNN and returns the trained model.
    """
    model = ParkinsonFNN(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if print_every and (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}")

    return model


def evaluate_model(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor):
    """
    Evaluates a trained model and prints metrics.
    Returns: (acc, f1, cm)
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_class = (y_pred > 0.5).float()

    acc = accuracy_score(y_test.cpu().numpy(), y_pred_class.cpu().numpy())
    f1 = f1_score(y_test.cpu().numpy(), y_pred_class.cpu().numpy())
    cm = confusion_matrix(y_test.cpu().numpy(), y_pred_class.cpu().numpy())

    print("\nModel Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    return acc, f1, cm


def save_model(model: nn.Module, path: str = "fnn_parkinson_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"\nModel saved as '{path}'")


def load_model(
    input_dim: int, path: str = "fnn_parkinson_model.pth", device: str = "cpu"
):
    model = ParkinsonFNN(input_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def main():
    # 1) Load & preprocess
    X_train, y_train, X_test, y_test, _scaler = load_and_preprocess("parkinsons.csv")

    # 2) Train
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        input_dim=X_train.shape[1],
        epochs=200,
        lr=0.001,
        print_every=10,
    )

    # 3) Evaluate
    evaluate_model(model, X_test, y_test)

    # 4) Save
    save_model(model, "fnn_parkinson_model.pth")


if __name__ == "__main__":
    main()
