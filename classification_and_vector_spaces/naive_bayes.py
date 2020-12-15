import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import confidence_ellipse, process_tweet, lookup

import pdb
from nltk.corpus import stopwords, twitter_samples
import nltk
import string
from nltk.tokenize import TweetTokenizer
from os import getcwd

# add folder, tmp2, from our local workspace containing pre-downloaded corpora files to nltk's data path
filePath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)

# get set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# split the data into two pieces, one for training and one for testing (validation set)
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# avoid assumptions about the length of all_positive_tweets
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))




def count_tweets(result, tweets, ys):
    """
    Input:
        result: a dictionary that will be used to map each pair to its frequency
        tweets: a list of tweets
        ys: a list corresponding to the sentiment of each tweet (either 0 or 1)
    Output:
        result: a dictionary mapping each pair to its frequency
    """
    for y, tweet in zip(ys, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in result:
                result[pair] += 1
            else:
                result[pair] = 1
    return result


'''
Test the count_tweet function
'''
result = {}
tweets = ['i am happy', 'i am tricked', 'i am sad', 'i am tired', 'i am tired']
ys = [1, 0, 0, 0, 0]
print(count_tweets(result, tweets, ys))

# Train your model using Naive Bayes
freqs = count_tweets({}, train_x, train_y)


def train_naive_bayes(freqs, train_x, train_y):
    """
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels corresponding to the tweets (0,1)
    Output:
        logprior: the log prior (equation, 3 above)
        loglikelihood: the log likelihood of your Naive bayes equation
    """
    loglikelihood = {}
    logprior = 0

    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    # calculating N_pos and N_neg
    N_pos = N_neg = 0
    for pair in freqs.keys():
        if pair[1]>0:
            # Increment the number of positive words by the count for this (word, label) pair
            N_pos += freqs[pair]
        # else, the label is negative
        else:
            # increment the number of negative words by the count for this (word, label) pair
            N_neg += freqs[pair]

    # calculate D the number of documents
    D = len(train_y)

    # calculate D_pos, the number of positive documents
    D_pos = len(list(filter(lambda x: x>0, train_y)))
    # caluclate D_neg, the number of negative documents
    D_neg = len(list(filter(lambda x: x<=0, train_y)))

    # Calculate the logprior
    logprior = np.log(D_pos) - np.log(D_neg)

    # For each word in the vocabulary
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = lookup(freqs, word, 1)
        freq_neg = lookup(freqs, word, 0)

        # calculating the probability that each word is positive, and negative
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        loglikelihood[word] = np.log(p_w_pos/p_w_neg)
    return logprior, loglikelihood


def naive_bayes_predict(tweet, logprior, loglikelihood):
    """
    :param tweet: a sting
    :param logprior: a number
    :param loglikelihood: a dictionary of words mapping to numbers
    :return: p: the sum of all the loglikelihoods of each word in the tweet (if found in the dictionary) + logprior (a number)
    """
    word_l = process_tweet(tweet)
    # initialize the probability to 0
    p = 0
    # add the logprior
    p += logprior
    for word in word_l:
        p += loglikelihood[word]

    return p

my_tweet = "She smiled"
logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
print(f"The expected output is {p}")

'''
Test the Naive Baye's model
'''
def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    """
    :param test_x: A list of tweets
    :param test_y: the corresponding labels for the list of tweets
    :param logprior: the logprior
    :param loglikelihood: a dictionary with the loglikelihood for each word
    :return: accuracy: (# of tweets classified correctly)/ (total # of tweets)
    """
    accuracy = 0 # return this property
    y_hats = []
    for tweet in test_x:
        # if the prediction is > 0
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            y_hat_i = 1
        else:
            y_hat_i = 0

        y_hats.append(y_hat_i)
    error = np.mean(np.absolute(y_hats-test_y))
    # Accuracy is 1 minus the error
    accuracy = 1 - error
    return accuracy

print(f"Naive Bayes accuracy = {test_naive_bayes(test_x, test_y, logprior, loglikelihood)}")











