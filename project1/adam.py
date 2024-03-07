import numpy as np
from optimizer import Optimizer


class ADAM(Optimizer):
    """Adaptive Moment Estimation optimizer"""

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        batch_size: int = 16,
        regularization: float = 0.001,
    ):
        super().__init__(regularization=regularization)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.first_moment = None
        self.second_moment = None
        self.iteration = 0

    def update(self, X, y, y_pred, params):
        self.iteration += 1
        if self.first_moment is None:
            self.first_moment = np.zeros(params.shape)
            self.second_moment = np.zeros(params.shape)

        X, y, y_pred = self.shuffle_data(X, y, y_pred)

        # Go through the data in batches
        for i in range(0, X.shape[0], self.batch_size):
            X_batch = X[i : i + self.batch_size]
            y_batch = y[i : i + self.batch_size]
            y_pred_batch = y_pred[i : i + self.batch_size]

            grad = self.gradient(X_batch, y_batch, y_pred_batch, params)
            self.first_moment = self.beta1 * self.first_moment + (1 - self.beta1) * grad
            self.second_moment = self.beta2 * self.second_moment + (1 - self.beta2) * grad**2

            # Bias correction
            first_moment_corr = self.first_moment / (1 - self.beta1**self.iteration)
            second_moment_corr = self.second_moment / (1 - self.beta2**self.iteration)

            params -= self.learning_rate * first_moment_corr / (np.sqrt(second_moment_corr) + self.epsilon)

        
        return params
