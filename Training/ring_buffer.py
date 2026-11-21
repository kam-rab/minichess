import numpy as np 

# total buffer capacity
CAPACITY = 80000

# default for current game (leela)
X_DIM = 302
PI_DIM = 625
V_DIM = 1

# Note: the buffer currently returns (q+z)/2 for v. 


class RingBuffer:
    """
    A class to hold buffers for X, pi, and v from self play games to be used for training
    """

    def __init__(self, capacity=CAPACITY, x_dim=X_DIM, pi_dim=PI_DIM, v_dim=V_DIM, q_dim=V_DIM): 
        self.X = np.empty((capacity, x_dim), dtype=np.float32)  
        self.pi = np.empty((capacity, pi_dim), dtype=np.float32) 
        self.v = np.empty((capacity, v_dim), dtype = np.float32)
        self.masks = np.empty((capacity, pi_dim), dtype=np.bool)
        self.moves_left = np.empty([capacity, v_dim], dtype=np.float32)
        self.q = np.empty([capacity, q_dim], dtype=np.float32)
        self.capacity = capacity
        self.index = 0
        self.size = 0

    def add(self, X, pi, v, masks, moves_left=None, q=None):
        num_states = X.shape[0]
        idxs = (np.arange(num_states) + self.index) % self.capacity
        self.X[idxs] = X
        self.pi[idxs] = pi
        self.v[idxs] = v
        self.masks[idxs] = masks
        if moves_left is not None:
            self.moves_left[idxs] = moves_left 
        if q is not None:
            self.q[idxs] = q
        else:
            self.q[idxs] = v
        # update indicators
        self.index = (self.index + num_states) % self.capacity
        self.size = min(self.size + num_states, self.capacity)

    def get_data(self, moves_left=False, q=False):
        """
        Return data in training data ready form:
        (X, (pi, v, moves_left), masks),  or y tuple of len 2 if moves_left=False
        with length size
        """
        value_targets = self.v[:self.size].copy()
        if q:
            value_targets += self.q[:self.size]
            value_targets /= 2
        if moves_left:
            return self.X[:self.size], (self.pi[:self.size], value_targets, self.moves_left[:self.size]), self.masks[:self.size]
        return self.X[:self.size], (self.pi[:self.size], value_targets), self.masks[:self.size]

    def save(self, path):
        n = self.size
        np.savez_compressed(path, X=self.X[:n], pi=self.pi[:n], v=self.v[:n],
                            masks=self.masks[:n], moves_left=self.moves_left[:n], q=self.q[:n], index=self.index)

    def load(self, path):
        with np.load(path) as d:
            n = d["X"].shape[0]
            shift = int(d["index"]) % n                   # oldest entry first (does nothing if  old buffer wasn't full)
            keep = min(n, self.capacity)                   # if the new buffer is smaller, keep the newest
            for name, arr in (("X", self.X), ("pi", self.pi), ("v", self.v), ("masks", self.masks), ("moves_left", self.moves_left), ("q", self.q)):
                src = d[name] if name in d.files else d["v"]     # buffers saved before q existed: q = z, so their target stays z
                arr[:keep] = np.roll(src, -shift, axis=0)[n - keep:]    # populate all arrs in ring buffer
            self.size = keep
            self.index = keep % self.capacity
