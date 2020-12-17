import numpy as np
import pandas as pd
import string
from collections import defaultdict
import math


# define tags for Adverb, Noun and To (the preposition), respectively

tags = ['RB', 'NN', 'TO']

def create_transition_matrix(alpha, tag_counts, transition_counts):
    """
    :param alpha: number used for smoothing
    :param tag_counts: a dictionary mapping each tag to its respective count
    :param transition_counts: transition count for the previous word and tag
    :return: A: matrix of dimension (num_tags, num_tags)
    """
    # get a sorted list of unique POS tags
    all_tags = sorted(tag_counts.keys())

    # count the number of unique POS tags
    num_tags = len(all_tags)

    # initialize the transition matrix 'A'
    A = np.zeros((num_tags, num_tags))

    # get the unique transition tuples (previous POS, current POS)
    trans_keys = set(transition_counts.keys())

    # go through each row of the transition matrix A
    for i in range(num_tags):
        # go though each column of the transition matrix A
        for j in range(num_tags):
            # initialize the count of the (prev POS, current POS) to zero
            count = 0
            # define the tuple (prev POS, current POS)
            # get the tag at position i and tag at position j (from the all_tag list)
            key = (all_tags[i], all_tags[j])

            # check if the (prev POS, current POS) tuple
            # exists in the transition counts dictionary

            if transition_counts: # complete this line
                # get count from the transition_counts dictionary
                # for the (prev POS, current POS) tuple
                count = transition_counts[key]

            # get the count of the previous tag (index position i) from tag_counts
            count_prev_tag = tag_counts[all_tags[i]]

            # apply smoothing using countof the tuple, alpha,
            # count of the previous tag, alpha, and number of total tags

            A[i, j] = (count+alpha)/(count_prev_tag+alpha*num_tags)

    return A


def create_emission_matrix(alpha, tag_counts, emission_counts, vocab):
    """
    :param alpha: tuing parameters used in smoothing
    :param tag_counts: a dictionary mapping each tag to its respective count
    :param emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
    :param vocab: a dictionary where keys are words in vocabulary and value is an index
    :return: B: a matrix of dimension (num_tags, len(vocab))
    """

    # get the number of POS tag
    num_tags = len(tag_counts)

    # get the number of all POS tags
    all_tags = sorted(tag_counts.keys())

    # get the total number of unique words in the vocabulary
    num_words = len(vocab)

    # initialize the emission matrix B with places for
    # tag in the rows and words in the columns

    B = np.zeros((num_tags, num_words))

    # get a set of all (POS, word) tuples
    # from the keys of the emission_counts dictionary
    emis_keys = set(list(emission_counts.keys()))

    # go through each row (POS tags)
    for i in range(num_tags):

        # go through each column (words)
        for j in range(num_words):
            # initialize the emission count for the (POS tag, word) to zero
            count = 0

            # define the (POS tag, word) tuple for this row and column
            key = (all_tags[i], vocab[j])

            # check if the (POS tag, word) tuple exists as a key in emission counts
            if key in emission_counts.keys():
                # get the count of (POS_tag, word) from the emission_counts b
                count = emission_counts[key]

            # get the count of the POS tag
            count_tag = tag_counts[all_tags[i]]

            # apply smoothing and store the smoothed value
            # into the emission matrix B for this row and column

            B[i, j] = (count + alpha)/ (count_tag + alpha*num_words)

    return B