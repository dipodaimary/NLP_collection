import re
import nltk
import emoji
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
from utils2 import get_dict, sigmoid, get_batches, compute_pca

# download sentences tokenizer
nltk.data.path.append('.')

'''
fdist = nltk.FreqDist(word for word in data)
'''

#word2Ind, Ind2word = get_dict(data)
#V = len(word2Ind)


'''
Training the model
'''

def initialize_model(N, V, random_seed=1):
    """
    :param N: dimension of hidden vector
    :param V: dimension of vocabulary
    :param random_seed: random seed for consistent results in the unit tests
    :return: W1, W2, b1, b2: initialized weights and biases
    """

    np.random.seed(random_seed)

    # W1 has shape (N,V)
    W1 = np.random.rand(N, V)

    # W2 has shape (V, N)
    W2 = np.random.rand(V, N)

    # b1 has shape (N, 1)
    b1 = np.random.rand(N, 1)

    # b2 has shape (V, 1)
    b2 = np.random.rand(V, 1)

    return W1, W2, b1, b2

# test the initialize_model function
tmp_N = 4
tmp_V = 10
tmp_W1, tmp_W2, tmp_b1, tmp_b2 = initialize_model(tmp_N, tmp_V)
assert tmp_W1.shape == ((tmp_N, tmp_V))
assert tmp_W2.shape == ((tmp_V, tmp_N))

print(f"tmp_W1.shape: {tmp_W1.shape}")
print(f"tmp_W2.shape: {tmp_W2.shape}")
print(f"tmp_b1.shape: {tmp_b1.shape}")
print(f"tmp_b2.shape: {tmp_b2.shape}")


'''
Softmax
'''
def softmax(z):
    """
    :param z: output scores from hidden layer
    :return: yhat: prediction (estimate of y)
    """

    # calculate yhat (softmax)
    e_z = np.exp(z)
    yhat = e_z/np.sum(e_z, axis=0)

    return yhat

# test the softmax function
tmp = np.array([
    [1, 2, 3],
    [1, 1, 1]
])
tmp_sm = softmax(tmp)
print(np.matrix(tmp_sm))

'''
Forward propagation
'''
def forward_prop(x, W1, W2, b1, b2):
    """
    :param x: average one hot vector for the context
    :param W1, W2, b1, b2: matrices and biases to be learned
    :return: z: output score vector
    """

    h = np.dot(W1, x) + b1

    # apply the relu on h (store result in h)
    h = np.maximum(0, h)

    # calculate z
    z = np.dot(W2, h) + b2

    return z, h


# test the forward_prop function
tmp_N = 2
tmp_V = 3
tmp_x = np.array([[0, 1, 0]]).T

tmp_W1, tmp_W2, tmp_b1, tmp_b2 = initialize_model(N=tmp_N, V=tmp_V, random_seed=1)

print(f"x has shape: {tmp_x.shape}")
print(f"N is {tmp_N} and vocabulary size V is {tmp_V}")

# call function
tmp_z, tmp_h = forward_prop(tmp_x, tmp_W1, tmp_W2, tmp_b1, tmp_b2)
print("call forward prop")

print(f"z has shape {tmp_z.shape}")
print(tmp_z)

print(f"h has shape {tmp_h.shape}")
print(tmp_h)


'''
cost function: cross entropy
'''
def compute_cost(y, yhat, batch_size):
    """
    cost function
    :param y:
    :param yhat:
    :param batch_size:
    :return:
    """
    logprobs = np.multiply(np.log(yhat), y) + np.multiply(np.log(1 - yhat), 1 - y)
    cost = - 1/batch_size * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost


# tests
tmp_C = 2
tmp_N = 50
tmp_batch_size = 4
#tmp_word2Ind, tmp_Ind2word = get_dict(data)

'''
Training the model - back propagation
'''
def back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size):
    """
    :param x: average one hot vector for the context
    :param yhat: prediction (estimate of y)
    :param y: target vector
    :param h: hidden vector (see eq. 1)
    :param W1, W2, b1, b2: matrices and biases
    :param batch_size: batch size
    :return:
        grad_W1, grad_W2, grad_b1, grad_b2: gradients of matrices and biases
    """

    # compute l1 as W2^T(yhat - y)
    # re-use it whenever you see W2^T(yhat - y) used to compute a gradient

    l1 = np.dot(W2.T, (yhat-y))

    # apply relu to ll
    l1 = np.max(0, l1)

    # compute the gradient of W1
    grad_W1 = (1/batch_size)*np.dot(l1, x.T)

    # compute the gradient of W2
    grad_W2 = (1/batch_size)*np.dot(yhat-y, h.T)

    # compute the gradient of b1
    grad_b1 = np.sum((1/batch_size)*np.dot(l1, x.T), axis=1, keepdims=True)

    # compute the gradient of b2
    grad_b2 = np.sum((1/batch_size)*np.dot(yhat-y, h.T), axis=1, keepdims=True)

    return grad_W1, grad_W2, grad_b1, grad_b2



'''
Gradient Descent implementation
'''
def gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.03):
    """
    This is the gradient_descent function
    :param data: text
    :param word2Ind: words to Indices
    :param N: dimension of hidden vector
    :param V: dimension of vocabulary
    :param num_iters: number of iterations
    :param alpha: learning rate
    :Output:
        W1, W2, b1, b2: updated matrices and biases
    """
    W1, W2, b1, b2 = initialize_model(N, V, random_seed=282)
    batch_size = 128
    iters = 0
    C = 2

    for x, y in get_batches(data, word2Ind, V, C, batch_size):
        # get z and h
        z, h = forward_prop(x, W1, W2, b1, b2)

        # get y_hat
        yhat = softmax(z)

        # get cost
        cost = compute_cost(y, yhat, batch_size)

        if ((iters+1) % 10 == 0):
            print(f"iters: {iers + 1} cost : {cos:.6f}")

        # get gradients
        grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size)

        # update the weights and biases
        W1 -= alpha*grad_W1
        W2 -= alpha*grad_W2
        b1 -= alpha*grad_b1
        b2 -= alpha*grad_b2

        iters += 1
        if iters == num_iters:
            break
        if iters % 100 == 0:
            alpha *= 0.66

    return W1, W2, b1, b2



























