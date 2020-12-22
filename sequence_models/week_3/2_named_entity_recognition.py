import trax
from trax import layers as tl
import os
import numpy as np
import pandas as pd

from utils import get_params, get_vocab
import random as rnd

def data_generator(batch_size, x, y, pad, shuffle=False, verbose=False):

    """
    :param batch_size: integer describing the batch size
    :param x: list containing sentences where words are represented as integers
    :param y: list containing tags associated with the sentences
    :param pad: an integer representing a pad character
    :param shuffle:
    :param verbose:
    :return:
        a tuple containing 2 elements:
            X - np.ndarray of dim (batch_size, max_len) of padded sentences
            Y - np.ndarray of dim (batch_size, max_len) of tags associated with the sentences in X
    """

    # count the number of lines in data_lines
    num_lines = len(x)

    # create an array with the indexes of data_lines that can be shuffled
    lines_index = [*range(num_lines)]

    # shuffle the indexes if shuffle is set to True
    pass


def NER(vocab_size=35181, d_model=50, tags=tag_map):

    """
    :param vocab_size: integer containing the size of the vocabulary
    :param d_model: integer describing the embedding size
    :param tags:
    :return:
        model: a trax serial model
    """
    model = tl.Serial(
        tl.Embedding(vocab_size, d_model), # embedding layer
        tl.LSTM(d_model), # lstm layer
        tl.Dense(len(tags)), # dense layer with len(tags) units
        tl.LogSoftmax() # logsoftmax layer
    )

    return model

model = NER()
print(model)

'''
Part 3: Train the model
'''
from trax.supervised import training

rnd.seed(33)
batch_size = 64

# crate training data, mask pad id=35180 for training
train_generator = trax.supervised.inputs.add_loss_weights(
    data_generator(batch_size, t_sentences, t_labels, vocab['<PAD>'], True),
    id_to_mask=vocab['<PAD>']
)


eval_generator = trax.supervised.inputs.add_loss_weights(
    data_generator(batch_size, v_sentences, v_labels, vocab['<PAD>'], True),
    id_to_mask=vocab['<PAD>']
)

# 3.1 training the model

def train_model(NER, train_generator, eval_generator, train_steps=1, output_dir='model'):
    """
    :param NER: the model you are building
    :param train_generator: the data generator for training example
    :param eval_generator: the data generator for validation example
    :param train_steps: number of training steps
    :param output_dir: folder to save your model
    :return:
        training_loop - a trax supervised training loop
    """

    train_task = training.TrainTask(
        train_generator,
        loss_layer=tl.CrossEntropy(), # a cross-entropy loss function
        optimizer=trax.optimizers.Adam(0.01), # the adam optimizer
    )

    eval_task = training.EvalTask(
        labeled_data = eval_generator,
        metrics = [tl.CrossEntropyLoss(), t1.Accuracy()], # evaluate with cross-entropy loss and accuracy
        n_eval_batches =10 # number of batches to use on each evaluation
    )

    training_loop = training.Loop(
        NER,
        train_task,
        eval_task,
        output_dir=output_dir
    )

    training_loop.run(n_steps=train_steps)

    return training_loop


'''
Part 4: Compute Accuracy
'''

def evaluate_prediction(pred, labels, pad):
    """
    :param pred: prediction array with shape
        (num examples, max sentence length in batch, num of classes)
    :param labels: array of size (batch_size, seq_len)
    :param pad: integer representing pad character
    :return:
        accuracy: float
    """

    outputs = np.argmax(pred, axis=1)

    mask = labels != pad

    accuracy = np.sum(outputs == labels)/ float(np.sum(mask))

    return accuracy


def predict(sentence, model, vocab, tag_map):
    s = [vocab[token] if token in vocab else vocab['UNK'] for token in sentence.slpit(' ')]
    batch_data = np.ones(1, len(s))
    batch_data[0][:] = s
    sentence = np.array(batch_data).astype(int)
    output = model(sentence)
    outputs = np.argmax(output, axis=2)
    labels = list(tag_map.keys())
    pred = []
    for i in range(len(outputs[0])):
        idx=outputs[0][i]
        pred_label=labels[idx]
        pred.append(pred_label)
    return pred



























