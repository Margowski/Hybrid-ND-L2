"""Common equal-capacity ridge readouts and metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CLASS_COUNTS = {"memory": 8, "context": 4, "self_state": 3, "relation": 4}


@dataclass
class RidgeReadout:
    alpha: float = 1.0
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    weights_: dict[str, np.ndarray] | None = None

    def fit(self, x: np.ndarray, targets: dict[str, np.ndarray]) -> "RidgeReadout":
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        z = (x - self.mean_) / self.scale_
        z = np.column_stack([z, np.ones(len(z))])
        gram = z.T @ z + self.alpha * np.eye(z.shape[1])
        self.weights_ = {}
        for name, y in targets.items():
            onehot = np.eye(CLASS_COUNTS[name])[y]
            self.weights_[name] = np.linalg.solve(gram, z.T @ onehot)
        return self

    def predict(self, x: np.ndarray) -> dict[str, np.ndarray]:
        if self.mean_ is None or self.scale_ is None or self.weights_ is None:
            raise RuntimeError("readout must be fitted before predict")
        z = (x - self.mean_) / self.scale_
        z = np.column_stack([z, np.ones(len(z))])
        return {name: np.argmax(z @ w, axis=1) for name, w in self.weights_.items()}


def accuracies(
    predictions: dict[str, np.ndarray], targets: dict[str, np.ndarray]
) -> dict[str, float]:
    result = {
        name: float(np.mean(predictions[name] == target))
        for name, target in targets.items()
    }
    result["macro"] = float(np.mean(list(result.values())))
    return result

