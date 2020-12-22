import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def sigmoid_gradient(x):
    return x * (1-x)



