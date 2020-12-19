import os
import random
import random as rnd
# import relevant libraries
import trax
trax.supervised.trainer_lib.init_random_number_generators(31)

# import trax.fastmath.numpy
import trax.fastmath.numpy as np

# import trax layers
from trax import layers as tl

from utils import Layer, load_tweets, process_tweet

# test fastmath array
a = np.array(5.0)
print(a)
print(type(a))


# define a function that will use trax.fastmath.numpy array

def f(x):
    return (x**2)

print(f"f(a) for a ={a} is {f(a)}")

# directly use trax.fastmath.grad to calculate the gradiate (derivative) of the function
grad_f = trax.fastmath.grad(fun=f)

print(type(grad_f))

grad_calculation = grad_f(a)
print(grad_calculation)

'''
Loading the data
'''

import numpy as np

all_positive_tweets, all_negative_tweets = load_tweets()
val_pos, train_pos = all_positive_tweets[4000:], all_positive_tweets[:4000]
val_neg, train_neg = all_negative_tweets[4000:], all_negative_tweets[:4000]

train_x = train_pos + train_neg
val_x = val_pos + val_neg

train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
val_y = np.append(np.ones(len(val_pos)), np.zeros(len(val_neg)))


'''
Building the vocabulary
'''

# include special tokens
# started with pad, end of lines and unk tokens

Vocab = {'__Pad__':0, '__</e>__':1, '__UNK__':2}

# note that we build vocab using training data
for tweet in train_x:
    processed_tweet = process_tweet(tweet)
    for word in processed_tweet:
        if word in processed_tweet:
            if word not in Vocab:
                Vocab[word] = len(Vocab)

print(f"total words in Vocab: {len(Vocab)}")

'''
Convert a tweet to a tensor
'''
def tweet_to_tensor(tweet, vocab_dict, unk_token='__UNK__', verbose=False):
    """

    :param tweet: a string containing a tweet
    :param vocab_dict: the words dictionary
    :param unk_token: the special string for unknown tokens
    :param verbose: print info during runtime
    :return: tensor_l: a python list with
    """

    word_l = process_tweet(tweet)

    if verbose:
        print(f"list of words from the processed tweet: {word_l}")

    tensor_l = []

    # get the unique integer ID of the __UNK__ token
    unk_ID = vocab_dict[unk_token]

    if verbose:
        print(f"the unique integer ID for the unk_token is {unk_ID}")

    # for each word in the list:
    for word in word_l:
        # get the unique integer ID
        # if the word doesn't exist in the vocab dictionary
        # use the unique ID for __UNK__ instead
        word_ID = vocab_dict[word] if word in vocab_dict else unk_ID

        tensor_l.append(word_ID)

    return tensor_l

print(f"actual tweet is: {val_pos[0]}")
print(f"tensor of tweet: {tweet_to_tensor(val_pos[0], vocab_dict=Vocab)}")

# test the function tweet_to_tensor
def test_tweet_to_tensor():
    test_cases = [
        {
            "name" : "simple_test_check",
            "input" : [val_pos[1], Vocab],
            "expected" : [444, 2, 304, 567, 56, 9],
            "error" : "The function gives bad output for val_pos[1]. Test failed"
        },
        {
            "name" : "datatype_check",
            "input" : [val_pos[1], Vocab],
            "expected" : type([]),
            "error" : "Datatype mismatch. Need only list not np.array"
        },
        {
            "name" : "without_unk_check",
            "input" : [val_pos[1], Vocab],
            "expected" : 6,
            "error" : "Unk word check not done- Please check if you included mapping for unknown word"
        }
    ]

    count = 0
    for test_case in test_cases:

        try:
            if test_case['name'] == "simple_test_check":
                assert test_case["expected"] == tweet_to_tensor(*test_case['input'])
                count += 1
            if  test_case['name'] == 'datatype_check':
                assert isinstance(tweet_to_tensor(*test_case['input']), test_case['expected'])
                count += 1
            if test_case['name'] == 'without_unk_check':
                assert None not in tweet_to_tensor(*test_case['input'])
                count += 1
        except:
            print(test_case['error'])

    if count == 3:
        print("\33[92mAll tests passed")
    else:
        print(count, " Tests passed out of 3")

test_tweet_to_tensor()


'''
Data generator
'''

def data_generator(data_pos, data_neg, batch_size, loop, vocab_dict, shuffle=False):
    '''
    Input:
        data_pos - Set of posstive examples
        data_neg - Set of negative examples
        batch_size - number of samples per batch. Must be even
        loop - True or False
        vocab_dict - The words dictionary
        shuffle - Shuffle the data order
    Yield:
        inputs - Subset of positive and negative examples
        targets - The corresponding labels for the subset
        example_weights - An array specifying the importance of each example

    '''
    ### START GIVEN CODE ###
    # make sure the batch size is an even number
    # to allow an equal number of positive and negative samples
    assert batch_size % 2 == 0

    # Number of positive examples in each batch is half of the batch size
    # same with number of negative examples in each batch
    n_to_take = batch_size // 2

    # Use pos_index to walk through the data_pos array
    # same with neg_index and data_neg
    pos_index = 0
    neg_index = 0

    len_data_pos = len(data_pos)
    len_data_neg = len(data_neg)

    # Get and array with the data indexes
    pos_index_lines = list(range(len_data_pos))
    neg_index_lines = list(range(len_data_neg))

    # shuffle lines if shuffle is set to True
    if shuffle:
        rnd.shuffle(pos_index_lines)
        rnd.shuffle(neg_index_lines)

    stop = False

    # Loop indefinitely
    while not stop:

        # create a batch with positive and negative examples
        batch = []

        # First part: Pack n_to_take positive examples

        # Start from pos_index and increment i up to n_to_take
        for i in range(n_to_take):

            # If the positive index goes past the positive dataset lenght,
            if pos_index >= len_data_pos:

                # If loop is set to False, break once we reach the end of the dataset
                if not loop:
                    stop = True;
                    break;

                # If user wants to keep re-using the data, reset the index
                pos_index = 0

                if shuffle:
                    # Shuffle the index of the positive sample
                    rnd.shuffle(pos_index_lines)

            # get the tweet as pos_index
            tweet = data_pos[pos_index_lines[pos_index]]

            # convert the tweet into tensors of integers representing the processed words
            tensor = tweet_to_tensor(tweet, vocab_dict)

            # append the tensor to the batch list
            batch.append(tensor)

            # Increment pos_index by one
            pos_index = pos_index + 1

        ### END GIVEN CODE ###

        ### START CODE HERE (Replace instances of 'None' with your code) ###

        # Second part: Pack n_to_take negative examples

        # Using the same batch list, start from neg_index and increment i up to n_to_take
        for i in range(n_to_take):

            # If the negative index goes past the negative dataset length,
            if neg_index >= len_data_neg:

                # If loop is set to False, break once we reach the end of the dataset
                if not loop:
                    stop = True;
                    break;

                # If user wants to keep re-using the data, reset the index
                neg_index = 0

                if shuffle:
                    # Shuffle the index of the negative sample
                    rnd.shuffle(neg_index_lines)
            # get the tweet as pos_index
            tweet = data_neg[neg_index_lines[neg_index]]

            # convert the tweet into tensors of integers representing the processed words
            tensor = tweet_to_tensor(tweet, vocab_dict)

            # append the tensor to the batch list
            batch.append(tensor)

            # Increment neg_index by one
            neg_index += 1

        ### END CODE HERE ###

        ### START GIVEN CODE ###
        if stop:
            break;

        # Update the start index for positive data
        # so that it's n_to_take positions after the current pos_index
        pos_index += n_to_take

        # Update the start index for negative data
        # so that it's n_to_take positions after the current neg_index
        neg_index += n_to_take

        # Get the max tweet length (the length of the longest tweet)
        # (you will pad all shorter tweets to have this length)
        max_len = max([len(t) for t in batch])

        # Initialize the input_l, which will
        # store the padded versions of the tensors
        tensor_pad_l = []
        # Pad shorter tweets with zeros
        for tensor in batch:
            ### END GIVEN CODE ###

            ### START CODE HERE (Replace instances of 'None' with your code) ###
            # Get the number of positions to pad for this tensor so that it will be max_len long
            n_pad = max_len - len(tensor)

            # Generate a list of zeros, with length n_pad
            pad_l = [0] * n_pad

            # concatenate the tensor and the list of padded zeros
            tensor_pad = tensor + pad_l

            # append the padded tensor to the list of padded tensors
            tensor_pad_l.append(tensor_pad)

        # convert the list of padded tensors to a numpy array
        # and store this as the model inputs
        inputs = np.array(tensor_pad_l)

        # Generate the list of targets for the positive examples (a list of ones)
        # The length is the number of positive examples in the batch
        target_pos = [1] * n_to_take

        # Generate the list of targets for the negative examples (a list of ones)
        # The length is the number of negative examples in the batch
        target_neg = [0] * n_to_take

        # Concatenate the positve and negative targets
        target_l = target_pos + target_neg

        # Convert the target list into a numpy array
        targets = np.array(target_l)

        # Example weights: Treat all examples equally importantly.
        example_weights = np.ones_like(targets)

        ### END CODE HERE ###

        ### GIVEN CODE ###
        # note we use yield and not return
        yield inputs, targets, example_weights


random.seed(30)

# create the training data generator
def train_generator(batch_size, shuffle=False):
    return data_generator(train_pos, train_neg, batch_size, True, Vocab, shuffle)

# create the validation data generator
def val_generator(batch_size, shuffle=False):
    return data_generator(val_pos, val_neg, batch_size, True, Vocab, shuffle)

# create the Validation data generator
def test_generator(batch_size, shuffle=False):
    return data_generator(val_pos, val_neg, batch_size, False, Vocab, shuffle)

# get a batch from the train generator and inspect
inputs, targets, example_weights = next(train_generator(4, shuffle=True))

# this will print a list of 4 tensors padded with zeros
print(f"inputs: {inputs}")
print(f"targets: {targets}")
print(f"example weights: {example_weights}")

# test the train_generator

# create a data_generator for training data,
# which produces batches of size 4 (for tensors and their respective targets)
tmp_data_gen = train_generator(batch_size=4)

# call the data generator to get one batch and its targets
tmp_inputs, tmp_targets, tmp_example_weights = next(tmp_data_gen)

print(f"The input shape is {tmp_inputs.shape}")
print(f"The targets shape is {tmp_targets.shape}")
print(f"The example weights shape is {tmp_example_weights.shape}")

for i,t in enumerate(tmp_inputs):
    print(f"input tensor: {t}; target {tmp_targets[i]}; example weights {tmp_example_weights[i]}")


'''
Defining Classes
'''
class Layer(object):
    """Base class for layers"""

    # constructor
    def __init__(self):
        # set weights to None
        self.weights = None

    # the forward propagation should be implemented
    # by subclass of this Layer class
    def forward(self, x):
        raise NotImplementedError

    # this function initializes the weights
    # based on the input signature and random key
    # should be implemented by subclasses of this Layer class
    def init_weights_and_state(self, input_signature, random_key):
        pass

    # this initializes and returns the weights, do not override
    def init(self, input_signature, random_key):
        self.init_weights_and_state(input_signature, random_key)
        return self.weights

    # __call__ allows an object of this class
    # to be called like it's a function
    def __call__(self, x):
        # when this layer object is called,
        # it calls its forward propagation function
        return self.forward(x)


'''
ReLU class
'''

class Relu(Layer):
    """Relu activation function implementation"""
    def forward(self, x):
        """

        :param x: (a numpy array): the input
        :return: activation (numpy array): all positive or 0 version of x
        """

        activation = np.maximum(x, 0)

        return activation


# test the relu layer
x = np.array([[-2.0, -1.0, 0.0], [0.0, 1.0, 2.0]], dtype=float)
relu_layer = Relu()
print(f"test data is: {x}")
print(f"output of relu is: {relu_layer(x)}")


from trax import fastmath

np = fastmath.numpy
random = fastmath.random

# see how the fastmath.trax.random.normal function works
tmp_key = random.get_prng(seed=1)
print(f"The random seed generated by random.get_prng {tmp_key}")

tmp_shape=(2,3)
tmp_weight = trax.fastmath.random.normal(key=tmp_key, shape=tmp_shape)

print(tmp_weight)


'''
Dense class
'''
class Dense(Layer):
    """
    A dense (fully-connected) layer
    """
    def __init__(self, n_units, init_stdev=0.1):
        # set the number of units in this layer
        self._n_units = n_units
        self._init_stdev = init_stdev


    # implementation of 'forward()'
    def forward(self, x):
        # matrix multiply x and the weights matrix
        dense = np.dot(x, self.weights)

        return dense

    def init_weights_and_state(self, input_signature, random_key):
        # the input_signature has a .shep attribute that gives the shape as a tuple
        input_shape = input_signature.shape

        # generate the weight matrix from a normal distribution
        # and standard deviation of 'stdev'
        w = self._init_stdev * random.normal(key=random_key, shape=(input_shape[-1], self._n_units))

        self.weights = w

        return self.weights

# test your dense layer
dense_layer = Dense(n_units=10) # sets the number of units in dense layer
random_key = random.get_prng(seed=0) # sets the random seed
z = np.array([[2.0, 7.0, 25.0]]) # input array

dense_layer.init(z, random_key)
print(f"Weights are {dense_layer.weights}") # returns randomly generated weights
print(f"Forward function output is : {dense_layer(z)}") # returns multiplied values of units and weights

'''
Model
'''

def classifier(vocab_size=len(Vocab),
               embedding_dim=256,
               output_dim=2,
               mode='train'
               ):
    # create embedding layer
    embed_layer = tl.Embedding(
        vocab_size=vocab_size, # size of the vocabulary
        d_feature=embedding_dim # embedding dimension
    )

    # create a mean layer, to create an "average" word embedding
    mean_layer = tl.Mean(axis=1)

    # create a dense layer, one unit for each output
    dense_output_layer = tl.Dense(n_units=output_dim)

    # create the log softmax layer (no parameters needed)
    log_softmax_layer = tl.LogSoftmax()

    # use tl.Serial to combine all layers
    # and create the classifier
    # of type trax.layers.combinators.Serial

    model = tl.Serial(
        embed_layer, # embedding layer
        mean_layer, # mean layer
        dense_output_layer, # dense output layer
        log_softmax_layer # log softmax layer
    )

    return model


tmp_model = classifier()
print(type(tmp_model))
print(tmp_model)

'''
Training
Let's define TrainTask, EvalTask and Loop in preparation to train the model
'''

from trax.supervised import training
batch_size = 16
rnd.seed(271)

train_task = training.TrainTask(
    labeled_data=train_generator(batch_size=batch_size, shuffle=True),
    loss_layer=tl.CrossEntropyLoss(),
    optimizer=trax.optimizers.Adam(0.01),
    n_steps_per_checkpoint=10,
)

eval_task = training.EvalTask(
    labeled_data=val_generator(batch_size=batch_size, shuffle=True),
    metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
)

model = classifier()

output_dir = 'model/'
output_dir_expand = os.path.expanduser(output_dir)
print(output_dir_expand)

def train_model(classifier, train_task, eval_task, n_steps, output_dir):
    """

    :param classifier: the model you are building
    :param train_task: training task
    :param eval_task: evaluation task
    :param n_steps: the evaluation steps
    :param output_dir: folder to save your files
    :return: trainer: trax trainer
    """
    training_loop = training.Loop(classifier, # the learning model
                                  train_task, # the training task
                                  eval_task=eval_task, # the evaluation task
                                  output_dir=output_dir # the output directory
                                  )
    training_loop.run(n_steps=n_steps)

    return training_loop

training_loop = train_model(model, train_task, eval_task, 100, output_dir_expand)



'''
Practice making a prediction
'''
# create a generator object
tmp_train_generator = train_generator(16)

# get one batch
tmp_batch = next(tmp_train_generator)

# position 0 has the model inputs (tweets vs tensors)
# position 1 has the targets (the actual labels)
tmp_inputs, tmp_targets, tmp_example_weights = tmp_batch

print(f"The batch is a tuple of length {len(tmp_batch)} because position 0 contains the tweets, and position on 1 contains the targets.")
print(f"The shape of the tweet tensors is {tmp_inputs.shape} (num of examples, lenght of tweet tensors).")
print(f"The shape of the labels is {tmp_targets.shape}, which is the batch size.")
print(f"The shape of the example_weights is {tmp_example_weights.shape}, which is the same as inputs/targets size.")


# feed the tweet tensors into the model to get a prediction
tmp_pred = training_loop.eval_model(tmp_inputs)
print(f"The prediction shape is {tmp_pred.shape}, num of tensor_tweets as rows")
print("Column 0 is the probability of a negative sentiment (class 0)")
print("Column 1 is the probability of a positive sentiment (class 1)")
print()
print("View the prediction array")
tmp_pred

tmp_is_positive = tmp_pred[:,1] > tmp_pred[:,0]
for i,p in enumerate(tmp_is_positive):
    print(f"Neg log_prob {tmp_pred[i, 0]:.4f}\tPos log_prob {tmp_pred[i,1]:.4f}\tis positive? {p}\tactual {tmp_targets[i]}")


'''
Evaluation
'''
def compute_accuracy(preds, y, y_weights):
    """

    :param preds: a tensor of shape (dim_batch, output_dim)
    :param y: a tensor of shape (dim_batch, output_dim) with true labels
    :param y_weights: a n.ndarray with the a weight for each example
    :return:
    accuracy: a float between 0-1
    weighted_num_correct (np.float): Sum of the weighted correct predictions
    sum_weights (np.float32): Sum of the weights
    """

    is_pos = preds[:, 1] > preds[:, 0]

    # convert the array of booleans into an array of np.int32
    is_pos_int = is_pos.astype(np.int32)

    # compare the array of predictions (as int32) with the target (labels) of type int32
    correct = is_pos_int == y

    # count the sum of the weights
    sum_weights = np.sum(y_weights)

    # convert the array of correct predictions (boolean) into an array of np.float32
    correct_float = correct.astype(np.float32)

    # multiply each prediction with its corresponding weight
    weighted_correct_float = correct_float * y_weights

    # sum up the weighted correct predictions (of type np.float32), to go in the
    # denominator
    weighted_num_correct = np.sum(weighted_correct_float)

    # divide the number of weighted correct predictions by fthe sum of the
    # weights
    accuracy = weighted_num_correct/ sum_weights

    return accuracy, weighted_num_correct, sum_weights


'''
Testing on your own inputs
'''
def predict(sentence):
    inputs = np.array(tweet_to_tensor(sentence, vocab_dict=Vocab))

    # batch size 1, add dimension for batch, to work with the model
    inputs = inputs[None, :]

    # predict with the model
    pred_probs = model(inputs)

    # turn probabilities into categories
    preds = int(pred_probs[0, 1] > pred_probs[0, 0])

    sentiment = "negative"
    if preds == 1:
        sentiment = 'positive'

    return preds, sentiment































