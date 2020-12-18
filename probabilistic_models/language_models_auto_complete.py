import math
import random
import numpy as np
import pandas as pd
import nltk

def split_to_sentences(data):
    """
    :param data: str
    :return: a list of sentences
    """
    sentences = data.split("\n")
    # - remove leading and trailing spaces from each sentence
    # - drop sentences if they are empty strings
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s)>0]

    return sentences


def tokenize_sentences(sentences):
    """
    Tokenize sentences into tokens (words)
    :param sentences: list of strings
    :return: list of lists of tokens
    """

    tokenized_sentences = []

    # go through each sentence
    for sentence in sentences:
        # convert to lowercase letters
        sentence = sentence.lower()

        # convert into a list of words
        tokenized = nltk.word_tokenize(sentence)

        # append the list of words to the list of lists
        tokenized_sentences.append(tokenized)

    return tokenized_sentences

# test the tokenize_sentences function

sentences = ["Sky is blue.", "Leaves are green.", "Roses are red."]
print(tokenize_sentences(sentences))



def count_words(tokenized_sentences):
    """
    Count the number of word appearence in the tokenized sentences
    :param tokenized_sentences: list of lists of strings
    :return: dict that maps word (str) to the frequency (int)
    """

    word_counts = {}

    for sentence in tokenized_sentences:

        # go through each sentence
        for token in sentence:

            # if the word is not in the dictionary yet, set the count to 1
            if token not in word_counts.keys():
                word_counts[token] = 1
            else:
                word_counts[token] += 1

    return word_counts

# test the count_words function
tokenize_sentences = [
    ["sky", "is", "blue", "."],
    ["leaves", "are", "green", "."],
    ["roses", "are", "red", "."]
]

print(count_words(tokenize_sentences))



def get_words_with_n_plus_frequency(tokenized_sentences, count_threshold):
    """
    Find the words that occur n times or more
    :param tokenized_sentences:
    :param count_threshold:
    :return:
    """

    # initialize an empty list to contain the words that
    # appear at least 'minimum_freq' times.
    closed_vocab = []

    # get the word count of each tokenized sentences
    # use the function that you defined earlier to count the words
    word_counts = count_words(tokenized_sentences)

    # for each word and its count
    for word, cnt in word_counts.items():

        # check that the word's count
        # is at least as great as the minimum count

        if cnt >= count_threshold:

            # append the word to the list
            closed_vocab.append(word)

    return closed_vocab


# test the function get_words_with_n_plus_frequency

tokenized_sentences = [
    ["sky", "is", "blue", "."],
    ["leaves", "are", "green", "."],
    ["roses", "are", "red", "."]
]

tmp_closed_vocab = get_words_with_n_plus_frequency(tokenized_sentences, count_threshold=2)
print(f"closed vocabulary:\n{tmp_closed_vocab}")


def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    """
    Replace words not in the given vocabulary with '<unk>' token.
    :param tokenized_sentences: list of lists of string
    :param vocabulary: list of strings that we will use
    :param unknown_token: a string representing unknown (out-of-vocabulary) words
    :return: list of lists of strings, with words not in the vocabulary replaced
    """

    # place vocabulary into a set for faster search
    vocabulary = set(vocabulary)

    # initialize a list that will hold the sentences
    # after less frequent words are replaced by the unknown token
    replaced_tokenized_sentences = []

    # go through each sentence
    for sentence in tokenized_sentences:

        # initialize the list that will contain
        # a single sentence with "unknown_token" replacements
        replaced_sentence = []

        # for each token in the sentence
        for token in sentence:

            # check if the token is in the closed vocabulary
            if token in vocabulary:
                # if so append the word to the replaced_sentence
                replaced_sentence.append(token)
            else:
                # otherwise, append the unknown token instead
                replaced_sentence.append(unknown_token)

        # append the list of tokens to the list of lists
        replaced_tokenized_sentences.append(replaced_sentence)

    return replaced_tokenized_sentences


'''
Test the replace_oov_words_by_unk function
'''
tokenized_sentences = [["dogs", "run"], ["cats", "sleep"]]
vocabulary = ["dogs", "sleep"]
tmp_replaced_tokenized_sentences = replace_oov_words_by_unk(tokenized_sentences, vocabulary)
print(f"original sentence: {tokenized_sentences}")
print(f"tokenized sentence with less frequent words converted to '<unk>': {tmp_replaced_tokenized_sentences}")


'''
Develop n-gram based language models
'''
def count_n_grams(data, n, start_token='<s>', end_token='<e>'):
    """
    Count all n-grams in the data
    :param data: list of lists of words
    :param n: number of words in the sequence
    :param start_token:
    :param end_token:
    :return: a dictionary that maps a tuple of n-words to its frequency
    """

    # initialize dictionary of n-grams and their counts
    n_grams = {}

    # go through each sentence in the data
    for sentence in data: # complete this line

        # prepend start token n times, and append <e> one time
        sentence = [start_token]*n + sentence + [end_token]

        # convert list to tuple
        # so that the sequence of words can be used as
        # a key in the dictionary
        sentence = tuple(sentence)

        # use 'i' to indicate the start of the n-gram
        # from index 0
        # to the last index where the end of the n-gram
        # is within the sentence

        m = len(sentence) if n == 1 else len(sentence)-1
        for i in range(m):

            # get the n-gram from i to i+n
            n_gram = sentence[i:i+n]

            # check if the n-gram is in the dictionary
            if n_gram in n_grams.keys():
                # increment the count for this n-gram
                n_grams[n_gram] += 1
            else:
                # initialize this n-gram count to 1
                n_grams[n_gram] = 1

    return n_grams


sentences = [
    ["i", "like", "a", "cat"],
    ["this", "dog", "is", "like", "a", "cat"]
]

print(f"uni-gram: {count_n_grams(sentences, 1)}")
print(f"bi-gram: {count_n_grams(sentences, 2)}")


'''
Estimate the probability of a word given the prior 'n' words using the n-gram counts
K-smoothing
'''

def estimate_probability(word,
                         previous_n_gram,
                         n_gram_counts,
                         n_plus1_gram_counts,
                         vocabulary_size,
                         k=1.0):
    """
    Estimate the probabilities of a next word using the n-gram counts with k-smoothing
    :param word: next word
    :param previous_n_gram: a sequence of words of length n
    :param n_gram_counts: dictionary of counts of n-grams
    :param n_plus1_gram_counts: dictioanry of counts of (n+1)-grams
    :param vocabulary_size: number of words in the vocabulary
    :param k: positive constant, smoothing parameter
    :return: a probability
    """

    # convert list to tuple to use it as a dictionary key
    previous_n_gram = tuple(previous_n_gram)

    # set the denominator
    # if the previous n-gram exists in the dictionary of n-gram counts,
    # get its count. Otherwise set the count to zero
    # use the dictionary that has counts for n-grams
    previous_n_gram_count = n_gram_counts[previous_n_gram] if previous_n_gram in n_gram_counts else 0

    # calculate the denominator using the count of previous n gram
    # and apply k-smoothing
    denominator = previous_n_gram_count + k*vocabulary_size

    # define n plus 1 gram as the previous n-gram plus the current word as a tuple
    n_plus1_gram = previous_n_gram + (word,)

    # set the count to the count in the dictionary
    # otherwise 0 if not in the dictionary
    # use the dictionary that has counts for the n-gram plus current word
    n_plus1_gram_count = n_plus1_gram_counts[n_plus1_gram] if n_plus1_gram in n_plus1_gram_counts else 0

    # define the numerator use the count of the n-gram plus current word,
    # and apply smoothing
    numerator = n_plus1_gram_count + k

    # calculate the probability as the numerator divided by denominator
    probability = numerator/ denominator

    return probability


# test the estimate probability function
sentences = [
    ["i", "like", "a", "cat"],
    ["this", "dog", "is", "like", "a", "cat"]
]
unique_words = list(set(sentences[0]+sentences[1]))

unigram_counts = count_n_grams(sentences, 1)
bigram_counts = count_n_grams(sentences, 2)
tmp_prob = estimate_probability("cat", "a", unigram_counts, bigram_counts, len(unique_words), k=1)

print(f"The estimated probability of word 'cat' given the previous n-gram 'a' is: {tmp_prob:.4f}")


'''
Estimate probabilities for all words
'''
def estimate_probabilities(
        previous_n_gram,
        n_gram_counts,
        n_plus1_gram_counts,
        vocabulary,
        k=1.0
):
    """
    Estimate the probabilities of next word using the n-gram counts with k-smoothing
    :param previous_n_gram: a sequence of words of length n
    :param n_gram_counts: dictionary of counts of n-grams
    :param n_plus1_gram_counts: dictionary of counts of (n+1)-grams
    :param vocabulary: list of words
    :param k: positive constant, smoothing parameter
    :return: a dictionary mapping from next words to the probability
    """

    # convert list to tuple to use it as a dictionary key
    previous_n_gram = tuple(previous_n_gram)

    # add <e> <unk> to the vocabulary
    # <s> is not needed since it should not appear as the next word
    vocabulary = vocabulary + ["<e>", "<unk>"]
    vocabulary_size = len(vocabulary)

    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary_size, k=k)
        probabilities[word] = probability

    return probabilities

# test your code
sentences = [
    ["i", "like", "a", "cat"],
    ["this", "dog", "is", "like", "a", "cat"]
]
unique_words = list(set(sentences[0]+sentences[1]))
unigram_counts = count_n_grams(sentences, 1)
bigram_counts = count_n_grams(sentences, 2)
estimated_probabilities = estimate_probabilities("a", unigram_counts, bigram_counts, unique_words, k=1)
print(estimated_probabilities)





















