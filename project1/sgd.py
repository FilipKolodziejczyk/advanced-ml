import numpy as np
from optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""

    def __init__(self, learning_rate: float = 0.001, batch_size: int = 1):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.regularization = 0.001

    def gradient(self, X, y, y_pred, params):
        # Check for bias and adjust if necessary
        if X.shape[1] != params.shape[0]:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        return X.T @ (y_pred - y) / X.shape[0] + self.regularization * params

    def update(self, X, y, y_pred, params):
        # Shuffle the data
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X, y, y_pred = X[indices], y[indices], y_pred[indices]

        # Perform the update
        for i in range(0, X.shape[0], self.batch_size):
            X_batch = X[i : i + self.batch_size]
            y_batch = y[i : i + self.batch_size]
            y_pred_batch = y_pred[i : i + self.batch_size]
            grad = self.gradient(X_batch, y_batch, y_pred_batch, params)
            params -= self.learning_rate * grad

        return params
