import string
import re
import os
import nltk

nltk.download('twitter_samples')
nltk.download('stopwords')

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, twitter_samples

tweet_tokenizer = TweetTokenizer(preserve_case=False,
                                 strip_handles=True,
                                 reduce_len=True)

# stopwords are messy and that compelling;
# 'very' and 'not' are considered stop words, but they are obviously expressing sentiment

# the porter stemmer lemmatizes 'was' to 'wa'

stopwords_english = stopwords.words('english')

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


def process_tweet(tweet):
    """
    :param tweet: a string containing a tweet
    :return: tweet_clean: a list of words containing processed_tweet
    """

    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    
    # remove oldstyle retweet text 'RT'
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
        if (word not in stopwords_english and
                word not in string.punctuation):

            stem_word = stemmer.stem(word) # stemming the word
            tweets_clean.append(stem_word)

    return tweets_clean


def load_tweets():

    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')

    return all_positive_tweets, all_negative_tweets


class Layer(object):
    """ Base class for layers """

    def __init__(self):
        self.weights = None

    def forward(self, x):
        raise NotImplementedError

    def init_weights_and_state(self, input_signature, random_key):
        pass

    def init(self, input_signature, random_key):
        self.init_weights_and_state(input_signature, random_key)

    def __call__(self, x):
        return self.forward(x)



























