from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    """An abstract class for ML model optimizers"""

    def __init__(self, regularization: float = 0.001):
        self.regularization = regularization

    def gradient(self, X, y, y_pred, params):
        """Calculate the gradient of the loss function with respect to the model's parameters"""
        # Check for bias and adjust if necessary
        if X.shape[1] != params.shape[0]:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        return X.T @ (y_pred - y) / X.shape[0] + self.regularization * params

    def shuffle_data(self, X, y, y_pred):
        """Shuffle the data for stochastic optimizers"""
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        return X[indices], y[indices], y_pred[indices]

    @abstractmethod
    def update(self, X, y, y_pred, params):
        """Update the model's parameters based on the given data and model"""
        raise NotImplementedError("Subclasses must implement this method")
