import numpy as np

from trax import layers as tl
from trax import shapes
from trax import fastmath

'''
Layers
'''
relu = tl.Relu()

# inspecting properties
print(f"name: {relu.name}")
print(f"expected inputs: {relu.n_in}")
print(f"promised_coutputs: {relu.n_out}")
