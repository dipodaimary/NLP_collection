import numpy as np
from numpy import random
from time import perf_counter

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

'''
Part 1: Feed method for vanilla RNNs and GRUs
'''
random.seed(10)             # random seed
emb = 128                   # embedding size
T = 256                     # number of variables in the sequence
h_dim = 16                  # hidden state dimension
h_0 = np.zeros((h_dim, 1))  # initial hidden state

# random initializations of weights and biases
w1 = random.standard_normal((h_dim, emb+h_dim))
w2 = random.standard_normal((h_dim, emb+h_dim))
w3 = random.standard_normal((h_dim, emb+h_dim))
b1 = random.standard_normal((h_dim, 1))
b2 = random.standard_normal((h_dim, 1))
b3 = random.standard_normal((h_dim, 1))
X = random.standard_normal((T, emb, 1))

weights = [w1, w2, w3, b1, b2, b3]

# forward methods for vanilla RNNs
def forward_V_RNN(inputs, weights): # forward propagation for a singe vanilla RNN cell

    x, h_t = inputs

    # weights
    wh, _, _, bh, _, _ = weights

    # new hidden state
    h_t = np.dot(wh, np.concatenate([h_t, x])) + bh
    h_t = sigmoid(h_t)

    return h_t, h_t


# forward method for GRUs
def forward_GRU(inputs, weights): # forward propagation for a single GRU cell

    x, h_t = inputs

    # weights
    wu, wr, wc, bu, br, bc = weights

    # update gate
    u = np.dot(wu, np.concatenate([h_t, x])) + bu
    u = sigmoid(u)

    # relevance gate
    r = np.dot(wr, np.concatenate([h_t, x])) + br
    r = sigmoid(r)

    # candidate hiddens state
    c = np.dot(wc, np.concatenate([r*h_t, x])) + bc
    c = np.tanh(c)

    # new hidden state h_t
    h_t = u*c + (1 - u)*h_t

    return h_t, h_t


#forward_GRU([X[1], h_0], weights)[0]


'''
Part 2: Implementation of the scan function
scan function is used for forward propagation in RNNs.
It takes as inputs:

1) fn: the function to be called recurrently (ie forward_GRU)
2) elems: the list of inputs for each time step(X)
3) weights: the parameters needed to compute fn
4) h_0: the initial hidden state
'''

def scan(fn, elems, weights, h_0=None): # forward propagation for RNNs

    h_t = h_0
    ys = []

    for x in elems:
        y, h_t = fn([x, h_t], weights)
        ys.append(y)

    return ys, h_t


'''
Part 3: comparison between vanilla RNNs and GRUs
'''

# vanilla RNNs
tic = perf_counter()
ys, h_T = scan(forward_V_RNN, X, weights, h_0)
toc = perf_counter()
RNN_time = (toc-tic)*1000
print(f"it took {RNN_time:.2f}ms to run the forward method for vanilla RNN")

# GRUs
tic = perf_counter()
ys, h_T = scan(forward_GRU, X, weights, h_0)
toc = perf_counter()
GRU_time = (toc - tic)*1000
print(f"It took {GRU_time:.2f}ms to run the forward method for the GRU.")



















