'''
We will explore stack semantics in trax. This will help in understanding
how to use layers like Select and Residual which gets. Stack is data
structure that follows the Last In, First Out (LIFO) principle. That
is, whatever is the latest element that is pushed into the stack will 
also be the first one to be popped out.
'''

import numpy as np              # regular numpy
from trax import layers as tl   # core building block
from trax import shapes         # data signatures: dimensionality and type
from trax import fastmath       # use jax, offers numpy on steroids

'''
1. The tl.Serial Combinator is Stack Oriented
'''
# defining addition
def Addition():
    layer_name = "Addition"

    def func(x, y):
        return x + y

    return tl.Fn(layer_name, func)

# test
add = Addition()

# inspect properties
print(f"name: {add.name}")
print(f"expected inputs: {add.n_in}")
print(f"promised outputs: {add.n_out}")

# inputs
x = np.array([3])
y = np.array([4])
print(f"x: {x}")
print(f"y: {y}")

# outputs
z = add((x, y))
print(f"z: {z}")

# defining multiplication
def Multiplication():
    layer_name = (
        "Multiplication"        # give your custom layer a name to identify
    )

    # custom function for the custom layer
    def func(x, y):
        return x*y

    return tl.Fn(layer_name, func)

# test it
mul = Multiplication()

# inspect properties
print(f"name: {mul.name}")
print(f"expected inputs: {mul.n_in}")
print(f"expected outputs: {mul.n_out}")

# inputs
x = np.array([7])
y = np.array([15])
z = mul((x, y))
print(f"z: {z}")

'''
Implementing the computations using Serial combinator
'''
serial = tl.Serial(
    Addition(), Multiplication(), Addition()    # add 3+4 # multpliply by 15 and add 3
)

# initialization
x = (np.array([3]), np.array([4]), np.array([15]), np.array([3])) # input

serial.init(shapes.signature(x))                # initializing serial instance

print(serial)
print(f"name: {serial.name}")
print(f"sublayers: {serial.sublayers}")
print(f"expected inputs: {serial.n_in}")
print(f"promised outputs: {serial.n_out}")

# outputs
y = serial(x)
print(f"y: {y}")



'''
2. The tl.Select combinator in the context of the Serial combinator
'''

serial = tl.Serial(tl.Select([0, 1, 0, 1]), Addition(), Multiplication(), Addition())

# initialization
x = (np.array([3]), np.array([4]))              # input

serial.init(shapes.signature(x))                # initializing serial instance

print("------- Serial Model -------")
print(serial)
print("------- Properties -------")
print(f"name: {serial.name}")
print(f"sublayer: {serial.sublayers}")
print(f"expected inputs: {serial.n_in}")
print(f"expected outputs: {serial.n_out}")

# inputs
print(f"x: {x}")

# outputs
y = serial(x)
print(f"y: {y}")


'''
Second example of tl.Select
'''
serial = tl.Serial(
    tl.Select([0, 1, 0, 1]),
    Addition(),
    tl.Select([0], n_in=2),
    Multiplication()
)

# initialization
x = (np.array([3]), np.array([4]))          # initializing serial instance

print(serial)
print("-- Properties --")
print(f"name: {serial.name}")
print(f"sublayers: {serial.sublayers}")
print(f"expected inputs: {serial.n_in}")
print(f"expected outputs: {serial.n_out}")

# inputs
print("-- Inputs --")
print(f"x: {x}")

# outputs
y = serial(x)
print(f"y: {y}")


'''
3. The tl.Residual combinator in the context of the Serial combinator

Residual networks are frequently used to make deep models to train and
you will be using it in the assignment as well. Trax already has a 
built in layer for this. The Residual layer computes the element-wise
sum of the stack-top input with the output of the layer series. For
example, if we wanted the cumulative sum of the following series
of computations (3+4)*3 + 4. The result can be obtained with the use
of the Residual combinator in the following manner.
'''

serial = tl.Serial(
    tl.Select([0, 1, 0, 1]),
    Addition(),
    Multiplication(),
    Addition(),
    tl.Residual()
)

# initialization
x = (np.array([3]), np.array([4]))          # input

serial.init(shapes.signature(x))            # initializing serial instance

print("-- Inputs --")
print(serial)
print("-- Properties --")
print(f"name: {serial.name}")
print(f"sublayers: {serial.sublayers}")
print(f"expected inputs: {serial.n_in}")
print(f"expected output: {serial.n_out}")

# inputs
print(f"x: {x}")

# outputs
y = serial(x)
print(f"y: {y}")


# a slightly trickier example:
serial = tl.Serial(
    tl.Select([0, 1, 0, 1]),
    Addition(),
    Multiplication(),
    tl.Residual(Addition())
)


# initialization
x = (np.array([3]), np.array([4]))

serial.init(shapes.signature(x))



















