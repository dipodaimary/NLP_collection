import os
import nltk
import trax
from trax import layers as tl
from trax.supervised import training
from trax.fastmatch import numpy as fastnp
import numpy as np
import pandas as pd
import random as rnd

from collections import defaultdict

vocab = defaultdict(lambda: 0)
vocab['<PAD>'] = 1

for idx in range(len(Q1_train_words)):
    Q1_train[idx] = nltk.word_tokenize(Q1_train_words[idx])
    Q2_train[idx] = nltk.word_tokenize(Q2_train_words[idx])
    q = Q1_train[idx] + Q2_train[idx]
    for word in q:
        if word not in vocab:
            vocab[word] = len(vocab) + 1




def Siamese(vocab_size=len(vocab), d_model=128, mode='train'):
    """Returns a Siamese model
    Args:
        vocab_size (int, optional): length of the vocabulary. Defaults to len(vocab)
        d_model (int, optional): deep of the model. defaults to 128
        mode (str, optional): 'train', 'eval', or 'predict', predict mode is for fast inference. Defaults to 'train'
    Returns:
          trax.layers.combinators.Praallel: A Siamese model
    """
    def normalize(x): # normalize the vectors to have L2 norm 1
        return x/ fastnp.sqrt(fastnp.sum(x*x, axis=-1, keepdims=True))


    q_processor = tl.Serial(
        tl.Embedding(vocab_size, d_model), # embedding layer
        tl.LSTM(d_model),
        tl.Mean(axis=1),
        tl.Fn('Normalize', lambda x: normalize(x))
    )

    model = tl.Parallel(q_processor, q_processor)

    return model

def TripletLossFn(v1, v2, margin=0.25):
    """Custom Loss function

    Args:
        v1 (numpy.ndarray): Array with dimension (batch_size, model_dimension) associated to Q1
        v2 (numpy.ndarray): Array with dimension (batch_size, model_dimension) associated to Q2
        margin (float, optional): Desired margin. Defaults to 0.25
    Returns:
        jax.interpreters.xla.DeviceArray: Triplet Loss
    """

    scores = fastnp.dot(v1, v2.T)

    batch_size = len(scores)

    positive = fastnp.diagonal(scores)

    negative_without_positive = scores - 2.0* fastnp.eye(batch_size)

    closet_negative = negative_without_positive.max(axis=1)

    negative_zero_on_duplicate = scores * (1.0 - fastnp.eye(batch_size))

    mean_negative = np.sum(negative_zero_on_duplicate, axis=1)/ (batch_size - 1)

    triplet_loss1 = fastnp.maximum(0.0, margin-positive+closet_negative)

    triplet_loss2 = fastnp.maximum(0.0, margin-positive+mean_negative)

    triplet_loss = fastnp.mean(triplet_loss1 + triplet_loss2)

    return triplet_loss


from functools import partial

def TripletLoss(margin=0.25):
    triplet_loss_fn = partial(TripletLossFn, margin=margin)
    return tl.Fn('TripletLoss', triplet_loss_fn)

























