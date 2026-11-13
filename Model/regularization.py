import numpy as np


class Regularizer:
    """
    Regularization for FullyConnected layers
    """
    def __init__(self, alpha=0.01, penalty='l2'):
        """
        penalty: type of distance measure, either "l1" or "l2"
        alpha: weight parameter for the regularization gradient

        You will not need to edit this function.
        """
        self.alpha = alpha
        self.penalty = penalty

    def grad(self, weights):
        weights = weights.copy()  # avoid direct modification
        if self.penalty == "l1":
            return self.l1_grad(weights)
        elif self.penalty == "l2":
            return self.l2_grad(weights)

    def l1_grad(self, weights):
        """
        Compute the gradient of L1 regularization for these weights.

        Note: weights[0, :] contains the intercept, and you should not apply
            regularization to those parameters.
        """
        grad = np.where(weights < 0, -1, 1) * self.alpha
        grad[weights == 0] = 0
        grad[0, :] = 0 
        return grad

    def l2_grad(self, weights):
        """
        Compute the gradient of L2 regularization for these weights.

        Note: weights[0, :] contains the intercept, and you should not apply
            regularization to those parameters.
        """
        grad = 2 * weights * self.alpha
        grad[0, :] = 0
        return grad
