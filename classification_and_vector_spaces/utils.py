import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random
import numpy as np

import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def process_tweet(tweet):
    '''Process tweet function
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    '''
    stemmer = PorterStemmer()
    stop_words_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False,
                               strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stop_words_english and
            word not in string.punctuation):
            stem_word = stemmer.stem(word) # stemming word
            tweets_clean.append(stem_word)
    return tweets_clean


def build_freqs(tweets, ys):
    """Build frequencies
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair
        to its frequency
    """
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """Create a plot of the covariance confidence ellipse of `x` and `y`
    :parameters
    x, y: array_like, shape (n, )
        Input data
    ax: matplotlib.axes.Axes
    n_std: float
        The number of standard deviations to determine the ellipse's radiuses.
    :returns
    matplotlib.patches.Ellipse
    Other parameters:
    kwargs: ``matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be of the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1]/ np.sqrt(cov[0, 0]*cov[1, 1])
    # Using a special case to obtaiin the eigenvalues of this
    # two-dimensional dataset
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x*2,
                      height=ell_radius_y*2,
                      facecolor=facecolor,
                      **kwargs
                      )
    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0])*n_std
    mean_x = np.mean(x)

    # Calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1])*n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().\
        rotate_deg(45).\
        scale(scale_x, scale_y).\
        translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def lookup(freqs, word, label):
    """
    :param freqs: a dictionary with the frequency of each pair (or tuple)
    :param word: the word to look up
    :param label: the label corresponding to the word
    :return: the number of times the word with its corresponding label appears
    """
    n = 0 # freqs.get((word, label), 0)
    pair = (word, label)
    if (pair in freqs):
        n = freqs[pair]
    return n






