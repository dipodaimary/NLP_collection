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

# inputs
x = np.array([-2, -1, 0, 1, 2])
print("-- Inputs --")
print(f"x: {x}")

# outputs
y = relu(x)
print("-- Outputs --")
print(f"y: {y}")

'''
Concatenate layers
'''

concat = tl.Concatenate()

# properties
print(f"name: {concat.name}")
print(f"expected inputs: {concat.n_in}")
print(f"expected outputs: {concat.n_out}")

x1 = np.array([-10, -20, -30])
x2 = x1/-10

# outputs
y = concat([x1, x2])
print(f"y: {y}")


'''
Layers are configurable
'''

concat_3 = tl.Concatenate(n_items=3) # configure the layer's expected inputs
print(f"name: {concat_3.name}")
print(f"expected inputs: {concat_3.n_in}")
print(f"promised outputs: {concat_3.n_out}")

# inputs
x1 = np.array([-10, -20, -30])
x2 = x1/-10
x3 = x2*0.99

print(f"x1: {x1}")
print(f"x2: {x2}")
print(f"x3: {x3}")

# outputs
y = concat_3([x1, x2, x3])
print(f"y: {y}")

'''
Layers can have weights
'''
# layer initialization
norm = tl.LayerNorm()

# you first must know what the input data will look like
x = np.array([0, 1, 2, 3], dtype='float')

# use the input data signature to get shape and type for initialization weights and biases
norm.init(shapes.signature(x)) # we need to convert the input datatype from usual tuple to trax ShapeDtype

print(f"Normal shape: {x.shape}, Data type: {type(x.shape)}")
print(f"Shapes Trax: {shapes.signature(x)}, Data Type: {type(shapes.signature(x))}")

# inspect properties
print(f"name: {norm.name}")
print(f"expected inputs: {norm.n_in}")
print(f"promised outputs: {norm.n_out}")

# weights and biases
print(f"weights: {norm.weights[0]}")
print(f"biases: {norm.weights[1]}")

# inputs
print(f"x: {x}")

# outputs
y = norm(x)
print(f"y: {y}")


'''
Custom layers
'''
# define a custom layer
# in this example you will create a layer to calculate the input times 2

def TimesTwo():
    layer_name = "TimesTwo" # don't forget to give your custom layer a name to identify

    # custom function for the custom layer
    def func(x):
        return x * 2

    return tl.Fn(layer_name, func)

# test it
times_two = TimesTwo()

# inspect properties
print(f"name: {times_two.name}")
print(f"expected inputs: {times_two.n_in}")
print(f"promised outputs: {times_two.n_out}")

# inputs
x = np.array([1, 2, 3])
print(f"x: {x}")

# outputs
y = times_two(x)
print(f"y: {y}")



'''
Combinators
you can combine layers to build more complex layers. 
Trax provides a set of objects named combinator to make this happen. 
Combinators are themselves layers, so behaviour commutes.
'''

# serial combinator
serial = tl.Serial(
    tl.LayerNorm(), # normalize input
    tl.Relu(),      # convert negative values to zero
    times_two       # the custom that we created above
)

# initialization
x = np.array([-2, -1, 0, 1, 2]) # input
serial.init(shapes.signature(x)) # initialising serial instance

print(serial)
print(f"name: {serial.name}")
print(f"sublayers: {serial.sublayers}")
print(f"expected inputs: {serial.n_in}")
print(f"promised outputs: {serial.n_out}")
print(f"weights & biases: {serial.weights}")

# inputs
print(f"x: {x}")

# outputs
y = serial(x)
print(f"y: {y}")


'''
JAX
'''

print(f"good old numpy: {type(np.array([1, 2, 3]))}")
print(f"jax trax numpy: {type(fastmath.numpy.array([1, 2, 3]))}")



