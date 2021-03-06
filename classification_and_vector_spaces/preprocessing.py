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


all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
tweets = all_positive_tweets + all_negative_tweets


print(f'Number of positive tweets: {len(all_positive_tweets)}')
print(f'Number of positive tweets: {len(all_negative_tweets)}')
print(f'Type of tweets {type(all_positive_tweets)}')

'''
fig = plt.figure(figsize=(5,5))
labels = 'Positive', 'Negative'
sizes = [len(all_positive_tweets), len(all_negative_tweets)]
plt.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')
plt.show()
'''

'''
Processing raw strings for semantic analysis
'''

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


'''
Tests process_tweets
'''
for ind in range(4):
    idx = random.randint(1, 4000)
    tweet = all_positive_tweets[idx]
    print(f'tweet: {tweet}\nprocessed_tweet: {process_tweet(tweet)}')

'''
Tests create frequency dictionary
'''
labels = np.append(np.ones(len(all_positive_tweets)),
                   np.zeros(len(all_negative_tweets)))
freqs = build_freqs(tweets, labels)
print(f'type(freqs) {type(freqs)}')
print(freqs)





