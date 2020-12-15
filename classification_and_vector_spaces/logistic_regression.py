import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples

from utils import process_tweet, build_freqs

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# split the data into two pieces, one for training and one for testing (validation set)
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# combine postive and negative labels
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)), axis=0)
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)), axis=0)

print(f'train_y.shape = {train_y.shape}')
print(f'test_y.shape = {test_y.shape}')

freqs = build_freqs(train_x, train_y)

# sigmoid function

def sigmoid(z):
    """
    Input:
        z: is the input (can be a scalaer or an array
    Output:
        h: the sigmoid of z
    """
    h = 1/(1 + np.exp(-z))
    return h

'''
Test the sigmoid function
'''
assert sigmoid(0) == 0.5
assert sigmoid(4.92) == 0.9927537604041685

def gradient_descent(x, y, theta, alpha, num_iters):
    """
    :param x: matrix of features which is (m, n+1)
    :param y: corresponding labels of the input matrix x, dimensions (m, 1)
    :param theta: weight vector of dimension (n+1, 1)
    :param alpha: learning rate
    :param num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    """
    m = x.shape[0]
    for i in range(0, num_iters):
        # get z, the dot product of x and theta
        z = np.dot(x, theta)

        # get the sigmoid of z
        h = sigmoid(z)

        #calculate the cost function
        J = -1/m*(np.dot(y.transpose(), np.log(h)) + np.dot((1-y).transpose(), np.log(1-h)))

        # update the weights theta
        theta = theta - (alpha/m)*np.dot(x.transpose(), (h-y))

        J = float(J)
    return J, theta

'''
Test of the gradient_descent function
'''
np.random.seed(1)
tmp_X = np.append(np.ones((10, 1)), np.random.rand(10, 2)*2000, axis=1)

tmp_Y = (np.random.rand(10, 1) > 0.35).astype(float)

tmp_J, tmp_theta = gradient_descent(tmp_X, tmp_Y, np.zeros((3,1)), 1e-8, 700)

print(f"The cost after training is {tmp_J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(tmp_theta)]}")


# Extracting the features from tweets
def extract_features(tweet, freqs):
    """
    Input:
        :param tweet: a list of words for one tweet
        :param freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output:
        x: a feature vector of dimension (1, 3)
    """
    word_1 = process_tweet(tweet)

    # 3 elements in the form of 1 x 3 vector
    x = np.zeros((1, 3))

    #bias term is set to 1
    x[0, 0] = 1

    # loop through each word in the list of words
    for word in word_1:
        # increment the word coount for the positive label 1
        x[0, 1] += freqs.get((word, 1.0), 0)

        # increment the word count for the negative label 0
        x[0, 2] += freqs.get((word, 0.0), 0)

    assert (x.shape == (1, 3))
    return x


'''
Testing the extract_features function
'''
tmp1 = extract_features(train_x[0], freqs)
print(tmp1)

tmp2 = extract_features('blorb bleeeeb blooop', freqs)
print(tmp2)

# Training the model
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :] = extract_features(train_x[i], freqs)

# training the labels corresponding to X
Y = train_y

# Apply gradient descent
J, theta = gradient_descent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
# print(f"The cost after training is {J:.8f}")
# print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")






















