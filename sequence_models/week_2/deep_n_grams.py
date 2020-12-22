'''
Overview:
Task will be to predict the next set of characters using
the previous characters.

1. start by converting a line of text into a tensor
2. create a generator to feed data into the model
3. train a neural network in order to predict the new
set of characters of defined length
4. use embeddings for each character and feed them as inputs
to your model
    a. many natural language tasks rely on using embeddings
    for predictions.
5. model will convert each character to its embeddings, run
the embeddings through embeddings through a Gated Recurrent Unit
GRU, and run it through a linear layer to predict the next set of
characters.
'''

import os
import trax
import trax.fastmath.numpy as np
import pickle
import numpy
import random as rnd
from trax import fastmath
from trax import layers as tl

# set the random seed
trax.supervised.trainer_lib.init_random_number_generators(32)
rnd.seed(32)

'''
Part 1: Importing the data
'''

# loading in the data
dirname = 'data/'
lines = [] # storing all the lines in a variable
for filename in os.listdir(dirname):
    with open(os.path.join(dirname, filename)) as files:
        for line in lines:
            # remove leading and trailing whitespace
            pure_line = line.strip()

            # if pure_line is not the empty string
            if pure_line:
                # append it to the list
                lines.append(pure_line)


n_lines = len(lines)
print(f"Number of lines: {n_lines}")
print(f"Sample line at position 0 {lines[0]}")
print(f"Sample line at position 999 {lines[999]}")

# go through each line
for i, line in enumerate(lines):
    # convert to all lowercase
    lines[i] = line.lower()

eval_lines = lines[-1000:] # create a holdout validation set
lines = lines[:-1000] # leave the rest for training

print(f"Number of lines for training: {len(lines)}")
print(f"Number of lines for validation: {len(eval_lines)}")

# 1.2 convert a line to tensor
def line_to_tensor(lines, EOS_int=1):
    """Turns a line of text into a tensor
    Args:
        lines (str): a single line of text
        EOS_int: (int, optional): end-of-sentence integer. Defaults to 1.

    :returns:
        list: a list of integers (unicode values) for the characters in the `line`
    """

    # initialize the tensor as empty list
    tensor = []

    # for each character:
    for c in line:

        # convert to unicode int
        c_int = ord(c)

        # append the unicode integer to the tensor list
        tensor.append(c_int)

    # include the end-of-sentence integer
    tensor.append(EOS_int)

    return tensor

# 1.3 batch generator
def data_generator(batch_size, max_length, data_lines, line_to_tensor=line_to_tensor, shuffle=True):
    """Generator function that yields batches of data

    Args:
        batch_size (int): number of examples (in this case, sentences) per batch.
        max_length (int): maximum length of the output tensor.
        NOTE: max_length includes the end-of-sentence character that will be added
                to the tensor.
                Keep in mind that the length of the tensor is always 1 + the length
                of the original line of characters.
        data_lines (list): list of the sentences to group into batches.
        line_to_tensor (function, optional): function that converts line to tensor. Defaults to line_to_tensor.
        shuffle (bool, optional): True if the generator should generate random batches of data. Defaults to True.

    Yields:
        tuple: two copies of the batch (jax.interpreters.xla.DeviceArray) and mask (jax.interpreters.xla.DeviceArray).
        NOTE: jax.interpreters.xla.DeviceArray is trax's version of numpy.ndarray
    """
    # initialize the index that points to the current position in the lines index array
    index = 0

    # initialize the list that will contain the current batch
    cur_batch = []

    # count the number of lines in data_lines
    num_lines = len(data_lines)

    # create an array with the indexes of data_lines that can be shuffled
    lines_index = [*range(num_lines)]

    # shuffle line indexes if shuffle is set to True
    if shuffle:
        rnd.shuffle(lines_index)

    ### START CODE HERE (Replace instances of 'None' with your code) ###
    while True:

        # if the index is greater or equal than to the number of lines in data_lines
        if index >= num_lines:
            # then reset the index to 0
            index = 0
            # shuffle line indexes if shuffle is set to True
            if shuffle:
                rnd.shuffle(lines_index)

        # get a line at the `lines_index[index]` position in data_lines
        line = data_lines[lines_index[index]]

        # if the length of the line is less than max_length
        if len(line) < max_length:
            # append the line to the current batch
            cur_batch.append(line)

        # increment the index by one
        index += 1

        # if the current batch is now equal to the desired batch size
        if len(cur_batch) == batch_size:

            batch = []
            mask = []

            # go through each line (li) in cur_batch
            for li in cur_batch:
                # convert the line (li) to a tensor of integers
                tensor = line_to_tensor(li)

                # Create a list of zeros to represent the padding
                # so that the tensor plus padding will have length `max_length`
                pad = [0] * (max_length - len(tensor))

                # combine the tensor plus pad
                tensor_pad = tensor + pad

                # append the padded tensor to the batch
                batch.append(tensor_pad)

                # A mask for  tensor_pad is 1 wherever tensor_pad is not
                # 0 and 0 wherever tensor_pad is 0, i.e. if tensor_pad is
                # [1, 2, 3, 0, 0, 0] then example_mask should be
                # [1, 1, 1, 0, 0, 0]
                # Hint: Use a list comprehension for this
                example_mask = [0 if t == 0 else 1 for t in tensor_pad]
                mask.append(example_mask)

            # convert the batch (data type list) to a trax's numpy array
            batch_np_arr = np.array(batch)
            mask_np_arr = np.array(mask)

            ### END CODE HERE ##

            # Yield two copies of the batch and mask.
            yield batch_np_arr, batch_np_arr, mask_np_arr

            # reset the current batch to an empty list
            cur_batch = []


# Try out your data generator
tmp_lines = ['12345678901', #length 11
             '123456789', # length 9
             '234567890', # length 9
             '345678901'] # length 9

# Get a batch size of 2, max length 10
tmp_data_gen = data_generator(batch_size=2,
                              max_length=10,
                              data_lines=tmp_lines,
                              shuffle=False)

# get one batch
tmp_batch = next(tmp_data_gen)

# view the batch
tmp_batch


# 1.4 repeating the batch generator
import itertools

infinite_data_generator = itertools.cycle(
    data_generator(batch_size=2, max_length=10, data_lines=tmp_lines))

ten_lines = [next(infinite_data_generator) for _ in range(10)]
print(len(ten_lines))


'''
Part 2: Defining the GRU model
'''

def GRUML(vocab_size=256, d_model=512, n_layers=2, mode='train'):
    """Returns a GRU Language Model
    Args:
        vocab_size (int, optional): size of the vocabulary. Defaults to 256
        d_model (int, optional): size of embedding (n_units in the GRU cell). Defaults to 512
        n_layers (int, optional): Number of GRU Layers. Deafults to 2
        mode (str, optional): 'train', 'eval' or 'predict', predict mode is for fast inference. Defaults to 'train'

    Returns:
        trax.layers.combinators.Serial: A GRU Language Model as a layer that maps from a tensor of tokens to activations
        over a vocab set.
    """

    model = tl.Serial(
        tl.ShiftRight(mode=mode),
        tl.Embedding(vocab_size=vocab_size, d_feature=d_model),
        [tl.GRU(n_units=d_model) for _ in range(n_layers)],
        tl.Dense(n_units=vocab_size),
        tl.LogSoftMax()
    )

    return model


'''
Part 3: Training
'''
batch_size = 32
max_length = 64

def n_used_lines(lines, max_length):
    """
    :param lines: all lines of text an array of lines
    :param max_length: max_length of a line in order to be considered an int
    :return:
        number of effective examples
    """

    n_lines = 0
    for l in lines:
        if len(l) <= max_length:
            n_lines += 1
        return n_lines


num_used_lines = n_used_lines(lines, 32)
steps_per_epoch = int(num_used_lines/batch_size)

# 3.1 training the model
from trax.supervised import training

def train_model(model, data_generator, batch_size=32, max_length=64, lines=lines, eval_lines=eval_lines, n_steps=1, output_dir='model/'):
    """Function that trains the model

    Args:
        model (trax.layers.combinators.Serial): GRU model.
        data_generator (function): Data generator function.
        batch_size (int, optional): Number of lines per batch. Defaults to 32.
        max_length (int, optional): Maximum length allowed for a line to be processed. Defaults to 64.
        lines (list, optional): List of lines to use for training. Defaults to lines.
        eval_lines (list, optional): List of lines to use for evaluation. Defaults to eval_lines.
        n_steps (int, optional): Number of steps to train. Defaults to 1.
        output_dir (str, optional): Relative path of directory to save model. Defaults to "model/".

    Returns:
        trax.supervised.training.Loop: Training loop for the model.
    """

    ### START CODE HERE (Replace instances of 'None' with your code) ###
    bare_train_generator = data_generator(batch_size, max_length, data_lines=lines)
    infinite_train_generator = itertools.cycle(bare_train_generator)

    bare_eval_generator = data_generator(batch_size, max_length, data_lines=eval_lines)
    infinite_eval_generator = itertools.cycle(bare_eval_generator)

    train_task = training.TrainTask(
        labeled_data=infinite_train_generator,  # Use infinite train data generator
        loss_layer=tl.CrossEntropyLoss(),  # Don't forget to instantiate this object
        optimizer=trax.optimizers.Adam(0.0005)  # Don't forget to add the learning rate parameter
    )

    eval_task = training.EvalTask(
        labeled_data=infinite_eval_generator,  # Use infinite eval data generator
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],  # Don't forget to instantiate these objects
        n_eval_batches=3  # For better evaluation accuracy in reasonable time
    )

    training_loop = training.Loop(model,
                                  train_task,
                                  eval_task=eval_task,
                                  output_dir=output_dir)

    training_loop.run(n_steps=n_steps)

    ### END CODE HERE ###

    # We return this because it contains a handle to the model, which has the weights etc.
    return training_loop


training_loop = train_model(GRULM(), data_generator)


'''
Part 4: evaluation
'''


def test_model(preds, target):
    """Function to test the model.

    Args:
        preds (jax.interpreters.xla.DeviceArray): Predictions of a list of batches of tensors corresponding to lines of text.
        target (jax.interpreters.xla.DeviceArray): Actual list of batches of tensors corresponding to lines of text.

    Returns:
        float: log_perplexity of the model.
    """
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    total_log_ppx = np.sum(preds * tl.one_hot(target, preds.shape[-1]),
                           axis=-1)  # HINT: tl.one_hot() should replace one of the Nones

    non_pad = 1.0 - np.equal(target, 0)  # You should check if the target equals 0
    ppx = total_log_ppx * non_pad  # Get rid of the padding

    log_ppx = np.sum(ppx) / np.sum(non_pad)
    ### END CODE HERE ###

    return -log_ppx

# Testing
model = GRULM()
model.init_from_file('model.pkl.gz')
batch = next(data_generator(batch_size, max_length, lines, shuffle=False))
preds = model(batch[0])
log_ppx = test_model(preds, batch[1])
print('The log perplexity and perplexity of your model are respectively', log_ppx, np.exp(log_ppx))


'''
Part 5: Generating the language with your own model
'''


# Run this cell to generate some news sentence
def gumbel_sample(log_probs, temperature=1.0):
    """Gumbel sampling from a categorical distribution."""
    u = numpy.random.uniform(low=1e-6, high=1.0 - 1e-6, size=log_probs.shape)
    g = -np.log(-np.log(u))
    return np.argmax(log_probs + g * temperature, axis=-1)


def predict(num_chars, prefix):
    inp = [ord(c) for c in prefix]
    result = [c for c in prefix]
    max_len = len(prefix) + num_chars
    for _ in range(num_chars):
        cur_inp = np.array(inp + [0] * (max_len - len(inp)))
        outp = model(cur_inp[None, :])  # Add batch dim.
        next_char = gumbel_sample(outp[0, len(inp)])
        inp += [int(next_char)]

        if inp[-1] == 1:
            break  # EOS
        result.append(chr(int(next_char)))

    return "".join(result)


print(predict(32, ""))































