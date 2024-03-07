from abc import ABC, abstractmethod


class Optimizer(ABC):
    """An abstract class for ML model optimizers"""

    @abstractmethod
    def update(self, X, y, y_pred, params):
        """Update the model's parameters based on the given data and model"""
        raise NotImplementedError("Subclasses must implement this method")
