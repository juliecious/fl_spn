from typing import Dict

import numpy as np
import pandas as pd
from simple_einet.einet import Einet
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import torch

import logging


logger = logging.getLogger(__name__)


def load_dataset(name: str, test_size: float = 0.2, random_state: int = 42) -> Dict:
    data = fetch_openml(name, version=2, as_frame=True)
    df = data.frame

    df = df.replace("?", np.nan)
    df_clean = df.dropna()

    X = df_clean.drop("class", axis=1)
    y = df_clean["class"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["category", "object"]
    ).columns.tolist()

    # encoding based on feature types
    X_numeric = StandardScaler().fit_transform(X[numeric_features])
    X_numeric_df = pd.DataFrame(X_numeric, columns=numeric_features, index=X.index)

    X_categorical_encoded = pd.DataFrame(index=X.index)
    for col in categorical_features:
        le = LabelEncoder()
        X_categorical_encoded[col] = le.fit_transform(X[col].astype(str))

    X_processed = pd.concat([X_numeric_df, X_categorical_encoded], axis=1)
    y_encoded = LabelEncoder().fit_transform(y)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed.values,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    # wrap in torch tensor form
    X_train_tensor = torch.tensor(X_train).float()
    X_test_tensor = torch.tensor(X_test).float()
    y_train_tensor = torch.tensor(y_train).long()
    y_test_tensor = torch.tensor(y_test).long()

    return {
        "X_train": X_train_tensor,
        "X_test": X_test_tensor,
        "y_train": y_train_tensor,
        "y_test": y_test_tensor,
        "X_processed": X_processed,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }


def _accuracy(model: Einet, X: torch.tensor, y: torch.tensor):
    with torch.no_grad():
        outputs = model(X)
        predictions = outputs.argmax(-1)
        correct = (predictions == y).sum()
        total = y.shape[0]
        return (100.0 * correct / total).item()


def _f1_score(model: Einet, X: torch.tensor, y: torch.tensor, num_classes=None):
    with torch.no_grad():
        outputs = model(X)
        predictions = outputs.argmax(-1)
        # 假設 y 和 predictions 為 1D tensor
        if num_classes is None:
            num_classes = int(torch.max(y).item()) + 1
        f1_scores = []
        for c in range(num_classes):
            tp = ((predictions == c) & (y == c)).sum().item()
            fp = ((predictions == c) & (y != c)).sum().item()
            fn = ((predictions != c) & (y == c)).sum().item()
            if tp + fp + fn == 0:
                f1 = 0.0  # 防止0除
            else:
                precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)
        # 取 macro-average F1
        return 100.0 * sum(f1_scores) / len(f1_scores)


if __name__ == "__main__":
    print(load_dataset(name="adult", test_size=0.2, random_state=42).keys())
