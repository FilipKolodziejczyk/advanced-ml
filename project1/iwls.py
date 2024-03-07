import numpy as np
from optimizer import Optimizer


class IWLS(Optimizer):
    """Iteratively reweighted least squares optimizer"""

    def __init__(self, regularization: float = 0.001):
        super().__init__(regularization=regularization)

    def hessian(self, X, y_pred, params):
        """Calculate the Hessian of the loss function with respect to the model's parameters"""
        # Check for bias and adjust if necessary
        if X.shape[1] != params.shape[0]:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        W = np.diag(y_pred * (1 - y_pred))
        return X.T @ W @ X / X.shape[0] + self.regularization * np.eye(params.shape[0])

    def update(self, X, y, y_pred, params):
        grad = self.gradient(X, y, y_pred, params)
        hessian = self.hessian(X, y_pred, params)
        params -= np.linalg.inv(hessian) @ grad
        return params
