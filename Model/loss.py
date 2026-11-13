import numpy as np


class BinaryCrossEntropyLoss:
    """
    Binary Cross Entropy Loss function. You do not need to edit this class.
    """
    def forward(self, y_pred, y_true):
        """
        Save the inputs to self.old_input and then compute the binary
        cross-entropy loss
        """
        assert set(np.unique(y_true)).issubset(set([0, 1]))
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)

        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)

        self.old_input = (y_pred, y_true)
        grad = np.where(y_true, -np.log(y_pred), -np.log(1 - y_pred))
        return np.sum(grad)

    def backward(self, grad=None, lr=None):
        """
        Compute the gradient of the loss function

        `grad` and `lr` are left as arguments to match the other `backward`
        functions in this codebase, but will never be passed anything.
        """
        assert grad is None
        (y_pred, y_true) = self.old_input
        ret = (- y_true + y_pred) / (y_pred - y_pred ** 2)
        return ret


class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-12, 1)
        return  - np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]

    def backward(self, grad=None, lr=None):  # never needed as CE loss is only ever used in SigmoidCEBlock
        raise NotImplementedError("should be used with softmax-CE block in Model.backward")


class SquaredLoss:
    """
    Sum of Squared Error Loss.

    Your implementation should mirror (and can copy from) the
    BinaryCrossEntropyLoss implementation.
    """

    def forward(self, y_pred, y_true):
        """
        Return the scalar sum (or mean) of squared errors.

        Note: While sometimes Squared Error is written including a factor of
        1/2 (because `d/dx (1/2 x ^ 2) = x`) please do not include that factor
        in your implementation of the `forward` function.
        """
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
            
        self.errors = y_pred - y_true
        return np.mean(self.errors**2)

    def backward(self, grad=None, lr=None):
        """
        Return the vector gradient of the derivative of sum of squared errors.

        Note: Do not divide your gradient vector by the number of data
        examples. The gradient updates will be averaged across the batch in
        FullyConnected.backward.
        """
        return 2 * self.errors
