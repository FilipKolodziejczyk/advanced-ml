import numpy as np
from optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""

    def __init__(self, learning_rate: float = 0.001, batch_size: int = 16, regularization: float = 0.001):
        super().__init__(regularization=regularization)
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def update(self, X, y, y_pred, params):
        X, y, y_pred = self.shuffle_data(X, y, y_pred)

        # Go through the data in batches
        for i in range(0, X.shape[0], self.batch_size):
            X_batch = X[i : i + self.batch_size]
            y_batch = y[i : i + self.batch_size]
            y_pred_batch = y_pred[i : i + self.batch_size]
            grad = self.gradient(X_batch, y_batch, y_pred_batch, params)
            params -= self.learning_rate * grad

        return params
