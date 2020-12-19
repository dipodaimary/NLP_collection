import random
import numpy as np

# example of traversing a list of indexes to create a circular list

a = [1, 2, 3, 4]
b = [0]*10

a_size = len(a)
b_size = len(b)
lines_index = [*range(a_size)]

index = 0
for i in range(b_size):
    # we wrap by resetting index to 0 so the sequence circles back at the end
    # to point to the first index
    if index >= a_size:
        index = 0

    b[i] = a[lines_index[index]]
    index += 1

print(b)


'''
Shuffling the data order
'''
a = [1, 2, 3, 4]
b = []

a_size = len(a)
b_size = 10
lines_index = [*range(a_size)]
print(f"original order of index: {lines_index}")

# if we shuffle the index we can change the order of our circular list
# without modifying the order or our original data
random.shuffle(lines_index) # shuffle the order
print(f"shuffled order of index: {lines_index}")

print(f"new value order for first batch: {[a[index] for index in lines_index]}")
batch_counter = 1
index = 0

for i in range(b_size):
    # we wrap by resetting index to 0
    if index >= a_size:
        index = 0
        batch_counter += 1
        random.shuffle(lines_index) # reshuffle the order
        print(f"shuffled indexes for batch no. {batch_counter, lines_index}")
        print(f"values for batch no {batch_counter, [a[index] for index in lines_index]}")

    b.append(a[lines_index[index]])

    index += 1

print(f"final value of b: {b}")


def data_generator(batch_size, data_x, data_y, shuffle=True):
    """

    :param batch_size: integer describing the batch size
    :param data_x: list containing samples
    :param data_y: list containing labels
    :param shuffle: shuffle the data order
    :return:
        a tuple containing 2 elements
        X - list of dim (batch_size) of samples
        Y - list of dim (batch_size) of labels
    """

    data_lng = len(data_x) # len(data_x) must be equal to len(data_y)
    index_list = [*range(data_lng)] # create a list with the ordered indexes of sample data

    # if shuffle is set to true, we traverse the list in a random way
    if shuffle:
        random.shuffle(index_list) # inplace shuffle of the list

    index = 0 # start with the first element

    # fill all the None values with code taking reference of what you learned so far
    while True:
        X = [0] * batch_size # we can create a list with batch_size elements
        Y = [0] * batch_size # we can create a list with batch_size elements

        for i in range(batch_size):

            # wrap the index each time that we reach the end of the list
            if index >= data_lng:
                index = 0
                # shuffle the index_list if shuffle is true
                if shuffle:
                    random.shuffle(index_list) # re-shuffle the order

            X[i] = data_x[index_list[index]] # we can set the corresponding element in x
            Y[i] = data_y[index_list[index]] # we can set the corresponding element in y
            index += 1

        yield ((X, Y))


def test_data_generator():
    x = [1, 2, 3, 4]
    y = [xi ** 2 for xi in x]

    generator = data_generator(3, x, y, shuffle=False)

    assert np.allclose(next(generator), ([1,2,3], [1,4,9])), "first batch does not match"
    assert np.allclose(next(generator), ([4,1,2], [16,1,4])), "second batch does not match"
    assert np.allclose(next(generator), ([3,4,1], [9,16,1])), "third batch does not match"
    assert np.allclose(next(generator), ([2,3,4], [4,9,16])), "fourth batch does not match"

    print("\33[92mall tests passed!")


test_data_generator()















