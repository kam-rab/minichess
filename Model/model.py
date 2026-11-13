import heapq
from Model.layers import FullyConnected, SoftmaxActivation, Dropout
import numpy as np

# training hyperparams. They're set in the function calls, but also listed here for reference.
BATCH_SIZE = 128

class Model:
    """
    A wrapper class for a neural network composed of
    layers and a loss function
    """
    def __init__(self, layers, loss=None, learning_rate=1, softmax_ce_block=False, apply_mask=False):
        """
        layers: a list of layers
            each must have a `forward` and `backward` function
        loss: the loss function to use when calling self.backward
        softmax_ce_block: set True if softmax activation + cross entropy loss 
        apply_mask: if mask should be applied to final layer

        You should not need to edit this function.
        """
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate
        self.softmax_ce_block = softmax_ce_block
        self.apply_mask = apply_mask


    def predict(self, X):
        """
        Helper function to match the scikit-learn API

        You will not need to edit this function.
        """
        return self.forward(X)

    def forward(self, X, mask=None):
        """
        Take the input and pass it forward through each layer of the network,
        using the `.forward()` function of each layer.

        Return the output of the final layer.
        """
        for layer in self.layers[:-1]:
            X = layer.forward(X)
        return self.layers[-1].forward(X, mask) if self.apply_mask else self.layers[-1].forward(X)

    def backward(self, pred=None, y=None, grad=None):
        """
        Take the predicted and target outputs and compute the loss.

        Then, beginning with `self.loss` and continuing *backwards*
        through each layer of the network, use the `.backward()`
        function of each layer to perform backpropagation.

        Note: each call to `backward()` in self.layers
            should use self.learning_rate

        Returns grad
        """
        if self.softmax_ce_block:
            assert grad is None  # must be final layer and use CE loss, bc no grad for only softmax is implemented
            if grad is None: 
                grad = pred - y
            for layer in self.layers[::-1][1:]:
                grad = layer.backward(grad, self.learning_rate)
            return grad
        else:
            if grad is None:
                self.loss.forward(pred, y)  # will error if no loss is given and try to call backward
                grad = self.loss.backward()
            for layer in self.layers[::-1]:
                grad = layer.backward(grad, self.learning_rate)
            return grad


    def fit(self, X, y, mask=None, max_iter=10000):
        """
        Train the model on the data for `max_iter` iterations.
        For each iteration, call `self.forward` and then `self.backward` to
            make a prediction and then update each layer's weights.  Each
            layer needs to have its `forward()` function called before its
            `backward()` function; this should be handled by this class's
            `self.forward()` and `self.backward()`.

        This function should always run for `max_iter` iterations; don't stop
            early even if the gradients are negligibly small.

        Returns None
        """
        for _ in range(max_iter):
            self.backward(self.forward(X, mask), y)


class Node:
    """
    For a tree of Model objects.
    Used here to create a model with split heads, but designed to be much more flexible.
    Inputs for nodes with multiple parents must be summed.
    """

    def __init__(self, parents=None, children=None, model=None, loss_weights=None):
        """
        Parents and children are tuples of Nodes. 
        model is Model object
        temp_X is a list of pointers to X vectors used in ModelTree.forward()
        temp_grad is a list of pointers to grad vectors used in ModelTree.backward() 
        loss_weights is a tuple of len(children) indicating how heavily the gradient should be weighted coming from each child during backward()
        """
        self.parents = parents if parents else ()
        self.children = children if children else ()
        self.model = model
        self.loss_weights = loss_weights

        self.temp_grad = []
        self.temp_X = []

    def add_temp_X(self, X):
        self.temp_X.append(X)
        return len(self.parents) - len(self.temp_X)

    def forward(self, mask=None):
        new_X = self.model.forward(sum(self.temp_X), mask)
        self.temp_X = []
        return new_X

    def add_temp_grad(self, grad, par=None):
        self.temp_grad.append(grad * (1 if (par is None or self.loss_weights is None) else self.loss_weights[self.children.index(par)]))
        return len(self.children) - len(self.temp_grad)

    def backward(self):
        new_grad = self.model.backward(grad=sum(self.temp_grad))
        self.temp_grad = []
        return new_grad

    def backward_loss(self, pred, y):
        return self.model.backward(pred, y)

    def __lt__(self, other):  # for heapq
        return False



class ModelTree:
    """
    A wrapper class for a graph of Models
    Inputs for nodes with multiple parents are summed.
    """
    def __init__(self, roots=(), leaves=()):
        """
        roots should be a list of all root nodes for model tree
        Tree should not have cycles. If any cycles exist, forward and backward can run forever.
        Nodes with multiple parents must take in same dim X from each parent. 
        Graph must be fully connected.
        Include leaf node pointers if your tree has multiple leaves and order of outputs matters.
        Otherwise, output will be determined by tree traversal below (perfectly if only one leaf).
        Loss weights should be tuple of the same length as leaves and in the same order. Otherwise gradients sum equally from each loss.
        """
        self.roots = roots
        # initialize pointers to leaf nodes too
        if leaves:
            self.leaves = leaves
        else:
            leaves = []
            stack = []
            seen = []
            stack.extend(roots)
            while any(stack):
                node = stack.pop()
                if node not in seen:
                    seen.append(node)
                    if node.children:
                        stack.extend(node.children)
                    else:
                        if node not in leaves:
                            leaves.append(node)
            self.leaves = tuple(leaves)


    def forward(self, X, mask=None):
        """
        Tree traversal pq implementation. 
        """
        output = [None] * len(self.leaves)

        # set up queue
        pq = []
        curr = {} # to track stale nodes in queue
        for root in self.roots:
            priority = root.add_temp_X(X)
            heapq.heappush(pq, (priority, root))
            curr[root] = priority

        # now let queue handle work:
        while any(pq):
            priority, node = heapq.heappop(pq)
            if priority == curr[node]:
                assert priority <= 0  # pq should never reach node still waiting on inputs
                X = node.forward(mask)
                if not node.children:
                    output[self.leaves.index(node)] = X
                else:
                    for child in node.children:
                        priority = child.add_temp_X(X)
                        heapq.heappush(pq, (priority, child))
                        curr[child] = priority

        return output

    
    def backward(self, pred, y, return_loss=False):
        """ 
        Tree traversal just like forward(). Nonleaf Model objects are given grads for their backward functions.
        pred and y should both be tuples of length len(self.leaves)
        """
        # set up queue
        pq = []
        curr = {} # to track stale nodes in queue
        loss = []
        for i in range(len(self.leaves)):   # becuase of loss function vs non loss function grads, have to complete first step before queueing
            if return_loss:
                loss.append(self.leaves[i].model.loss.forward(pred[i], y[i]))

            grad = self.leaves[i].backward_loss(pred[i], y[i])
            for par in self.leaves[i].parents:
                priority = par.add_temp_grad(grad, self.leaves[i])
                heapq.heappush(pq, (priority, par))
                curr[par] = priority

        # now let queue handle work:
        while any(pq):
            priority, node = heapq.heappop(pq)
            if priority == curr[node]:
                assert priority <= 0  # pq should never reach node still waiting on inputs
                grad = node.backward()
                for par in node.parents:  # node.parents is empty for root nodes
                    priority = par.add_temp_grad(grad, node)
                    heapq.heappush(pq, (priority, par))
                    curr[par] = priority

        if return_loss:
            return tuple(loss)


    def evaluate(self, X, y, mask=None):
        """
        Return loss on dataset of X, y.
        Returns a tuple with same len as y, one loss per leaf (head). 
        """
        pred = self.forward(X, mask)
        loss = []
        for i in range(len(self.leaves)):
            loss.append(self.leaves[i].model.loss.forward(pred[i], y[i]))
        return loss


    def fit(self, X, y, mask=None, batch_size=BATCH_SIZE, return_loss=False):
        """
        Train the model on data examples. 
        X: (n, 300) data examples.
        y: tuple of target np arrays, one per leaf, each with n rows.
        """
        rng = np.random.default_rng()
        n = X.shape[0]

        idx = rng.integers(0, n, batch_size)  # sample data randomly 
        m = mask[idx] if mask is not None else None
        dropouts = self.get_layer_type(Dropout)
        for dropout in dropouts: 
            dropout.training = True
        try:
            ret = self.backward(self.forward(X[idx], m), tuple(t[idx] for t in y), return_loss)
        finally:
            for dropout in dropouts: 
                dropout.training = False
        return ret


    # deterministic, so same model tree always loads weights properly.
    def get_layer_type(self, type=FullyConnected):
        """
        type=FullyConnected is used for saving model.
        """
        layers = []
        stack = []
        seen = []
        stack.extend(self.roots)
        while any(stack):
            node = stack.pop()
            if node not in seen:
                seen.append(node)
                for layer in node.model.layers:
                    if isinstance(layer, type):
                        layers.append(layer)
                if node.children:
                    stack.extend(node.children)
        return layers

    def save(self, path):
        if not path[-4:] == ".npz": 
            path += ".npz"
        np.savez(path, *[layer.weights for layer in self.get_layer_type()])   # saved as arr_0, arr_1, ...

    def load(self, path):
        if not path[-4:] == ".npz":
            path += ".npz"
        with np.load(path) as weight_data:
            layers = self.get_layer_type()
            assert len(layers) == len(weight_data.files)   # ensure same number of dense layers
            
            for i in range(len(layers)):
                weights = weight_data[f"arr_{i}"]
                assert weights.shape == layers[i].weights.shape   # ensure same shape for each layer
                layers[i].weights[...] = weights    # copy into existing array, so not pointing to file







             


