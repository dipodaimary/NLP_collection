import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

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
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

def process_tweet(tweet):
    '''Process tweet function
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    '''
    stemmer = PorterStemmer()










