import numpy as np


class SigmoidActivation:
    """
    The sigmoid activation function

    You should not need to edit this class
    """
    def sigmoid(self, x):
        """
        Helper function to compute sigmoid and avoid warnings
        """
        with np.errstate(over="ignore"):  # exp overflow for very negative x just gives sigmoid = 0
            return 1 / (1 + np.exp(-x))

    def forward(self, X):
        """
        First, save the input to self.old_X for gradient updates
        Then, return sigmoid(X)
        """
        self.old_X = X
        return self.sigmoid(X)

    def backward(self, grad, lr=None):
        """
        Using the saved inputs from the last `forward` call,
            compute the gradient of the sigmoid with respect to those inputs
            Multiply by `grad`, the gradient computed previously
        """
        new_grad = self.sigmoid(self.old_X) * (1 - self.sigmoid(self.old_X))
        return new_grad * grad


class TanhActivation:
    def forward(self, X):
        self.out = np.tanh(X)
        return self.out
    
    def backward(self, grad, lr=None):
        return (1 - self.out**2) * grad


class SoftmaxActivation:
    def forward(self, X, mask=None):
        if mask is not None:
            X = np.where(mask, X, -1e9)
        stable = X - np.max(X, axis=1, keepdims=True)
        e = np.exp(stable)
        return e / np.sum(e, axis=1, keepdims=True)

    def backward(self, grad, lr=None):
        raise NotImplementedError("should be used with softmax-CE block in Model.backward")


class ReluActivation:
    """
    The ReLU activation function

    Your implementation should mirror (and can copy from) the SigmoidActivation
    implementation. Make sure you are using `self.old_X` to track past inputs.
    """
    def forward(self, x):
        self.old_X = x
        return np.maximum(x, 0)

    def backward(self, grad, lr=None):
        return (self.old_X > 0) * grad


class Dropout:
    """
    Drops fraction p of weights. Only turned on during fit, controlled by self.training. 
    """
    def __init__(self, p=0.2, rng=None):
        self.p = p                          # fraction of units dropped
        self.rng = rng or np.random.default_rng()
        self.training = False

    def forward(self, X):
        if not self.training:
            return X                       
        self.mask = (self.rng.random(X.shape) >= self.p) / (1 - self.p)   # scale kept units so the expected value is unchanged
        return X * self.mask

    def backward(self, grad, lr=None):
        return grad * self.mask if self.training else grad


class FullyConnected:
    """
    A fully-connected neural network layer.

    You don't need to modify this class.
    """
    def __init__(self, input_dim, output_dim, regularizer=None, rng=None):
        """
        input_dim: the input dimension of the layer,
            *not* including the intercept that will be added
            If this is the first layer and the input is 2-dimensional,
            input_dim should be 2.
        output_dim: the output dimension of the layer
        regularizer: if not None, must implement `.grad(weights)`
            to be called in `self.backward()` to add regularization
            to this layer
        rng: optional np.random.Generator for weight initialization; pass
            np.random.default_rng(seed) for reproducible weights

        Note that self.weights will have shape [1 + input_dim, output_dim]
            because an intercept (or bias) term is added here and then
            an intercept column is to X whenever `self.forward(X)` is called
        """
        # A weight initialization strategy called "Xavier initialization"
        #     https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        #     https://arxiv.org/pdf/1704.08863
        rng = rng if rng is not None else np.random.default_rng()
        self.weights = rng.normal(0, np.sqrt(1 / input_dim), (1 + input_dim, output_dim))
        self.regularizer = regularizer

        # for adam
        self.m = np.zeros_like(self.weights)
        self.v = np.zeros_like(self.weights)
        self.t = 0


    def forward(self, X):
        """
        First, concatenate an intercept vector to the inputs
        Then save the modified input to self.old_X for gradient updates
        Then compute X @ W
        """
        X = np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)
        self.old_X = X
        return X.dot(self.weights)

    def backward(self, grad, lr=0.1):
        """
        Using saved inputs from the previous `forward` call, compute
            two gradients: d(X @ W)/dW and d(X @ W)/dX.
            The first is used to update self.weights, and the second
            is returned to be used by later `backward` calls in the network.

        Note: you should not need to edit this function.
        """

        # Make sure the gradient that's been computed so far
        #   matches up to the shape of this layer
        input_dim, output_dim = self.weights.shape
        batch_size, output_dim2 = grad.shape
        assert output_dim == output_dim2, "Shape mismatch"

        # Compute d(X @ W)/dW for each node in this layer
        update = (self.old_X.T @ grad) / self.old_X.shape[0]

        # Compute d(X @ W)/dX for the next `backward` call
        #   and multiply it by `grad`
        new_grad = grad.dot(self.weights[1:, ].T)

        # Compute Adam
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * update
        self.v = 0.999 * self.v + 0.001 * (update**2)
        m_hat = self.m / (1 - 0.9**self.t)      # correct bias from starting at 0
        v_hat = self.v / (1 - 0.999**self.t)

        # Now, we can update our weights
        self.weights -= lr * (m_hat / (np.sqrt(v_hat) + 1e-8))

        # If using regularization, perform an additional update to self.weights
        if self.regularizer is not None:
            self.weights -= lr * self.regularizer.grad(self.weights)

        return new_grad

