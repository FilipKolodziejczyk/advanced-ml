import numpy as np
from optimizer import Optimizer


class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias = 0

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return self.sigmoid(z)

    @staticmethod
    def binary_cross_entropy(y, y_pred):
        # y_pred += 1e-50  # Preventing log(0) situations
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def fit(
        self,
        X,
        y,
        optimizer: Optimizer,
        max_epochs: int = 500,
        tolerance: float = 0.001,
    ):
        self.weights = np.random.normal(0, 0.01, X.shape[1])
        loss_history = [-np.inf]

        for _ in range(max_epochs):
            params = np.concatenate((self.weights, [self.bias]))
            new_params = optimizer.update(X, y, self.predict_proba(X), params)

            self.weights = new_params[:-1]
            self.bias = new_params[-1]
            loss_history.append(self.binary_cross_entropy(y, self.predict_proba(X)))

            if abs(loss_history[-1] - loss_history[-2]) <= tolerance:
                break

        return loss_history[1:]
