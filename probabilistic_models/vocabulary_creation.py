import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


text = "red pink pink blue blue yellow ORANGE BLUE BLUE PINK"

text_lowercase = text.lower()

words = re.findall(r'\w+', text_lowercase)

print(words)


# create vocabulary
vocab = set(words)

# add information with word count
counts_a = dict()
for w in words:
    counts_a[w] = counts_a.get(w, 0) + 1

print(counts_a)

# create vocab with collections.Counter
counts_b = dict()
counts_b = Counter(words)
print(counts_b)


'''
Candidates from string edits
'''
word = 'dearz'
splits = [(word[:i], word[i:]) for i in range(len(word)+1)]

for i in splits:
    print(i)

# deletes with a loop

for L, R in splits:
    if R:
        print(L+R[1:])


# deletes with lis comprehension

deletes = [L+R[1:] for L,R in splits]

for _ in deletes:
    print(_)

'''
Auto correct
'''

def get_count(word_l):
    '''
    :param word_l: a set of words representing the corpus
    :return: word_count_dict: the word count dictionary where key is the word and value is its frequency
    '''
    word_count_dict = {}
    word_count_dict = Counter(word_l)

    return word_count_dict

def get_probs(word_count_dict):
    '''
    :param word_count_dict: the wordcount dictionary where the key is the word and value is its frequency
    :return: probs: a dictionary where keys are the words and the values are the probability that a word will occur
    '''

    probs = {}

    m = sum(word_count_dict.values())

    for key in word_count_dict.keys():
        probs[key] = word_count_dict[key]/m

    return probs


'''
String manipulations
'''

def delete_letter(word, verbose=False):
    """
    :param word: the string/word for which we will generate all the possible words
    :param verbose:
    :return: delete_l: a list of all possible strings obtained by deleting 1 character from the word
    """

    delete_l = []
    split_l = []

    for c in range(len(word)):
        split_l.append((word[:c], word[c:]))

    for a, b in split_l:
        if b:
            delete_l.append(a+b[1:])

    if verbose: print(f"input word : {word}\nsplit_l : {split_l}\ndelete_l : {delete_l}")

    return delete_l


# Test the delete_letter function
delete_letter('cans', verbose=True)


def switch_letter(word, verbose=False):
    """
    :param word: input sting
    :param verbose: whether to print the logs
    :return: switches: a list of all possible strings with one adjacent character switched
    """

    switch_l = []
    split_l = []

    len_word = len(word)

    for c in range(len_word):
        split_l.append((word[:c], word[c:]))

    switch_l = [a + b[1] + b[0] + b[2:] for a,b in split_l if len(b)>=2]

    if verbose:
        print(f"word: {word}\nsplit_l : {split_l}\nswitch_l : {switch_l}")

    return switch_l

# test the
switch_letter("eta", verbose=True)



def replace_letter(word, verbose=False):

    """
    :param word: the input string/ word
    :param verbose: whether to print the logs
    :return: replaces: a list of all possible strings where we replaced one letter from the original word
    """

    letters = "abcdefghijklmnopqrstuvwxyz"
    replace_l = []
    split_l = []

    for c in range(len(word)):
        split_l.append((word[:c], word[c:]))

    replace_l = [a + l + (b[1:] if len(b) > 1 else '') for a,b in split_l if b for l in letters]
    replace_set = set(replace_l)
    replace_set.remove(word)

    replace_l = sorted(list(replace_set))

    if verbose:
        print(f"word : {word}\nsplit_l : {split_l}\nreplace_l : {replace_l}")

    return replace_l

# testing the replace_letter function
replace_letter(word="can", verbose=True)


def insert_letter(word, verbose=False):
    """
    :param word: the word/string
    :param verbose: whether to print the logs
    :return: inserts: a set of all possible strings with one new letter inserted at every offset
    """

    letters = "abcdefghijklmnopqrstuvwxyz"
    insert_l = []
    split_l = []

    for c in range(len(word)):
        split_l.append((word[:c], word[c:]))

    insert_l = [a + l + b for a,b in split_l for l in letters]

    if verbose:
        print(f"word : {word}\nsplit_l : {split_l}\ninsert_l : {insert_l}")

    return insert_l

insert_letter('at', True)


def edit_one_letter(word, allow_switches=True):
    """
    :param word: the string/word for which we will generate all possible words that are one edit away
    :param allow_switches:
    :return: edit_one_set: a set of words with one possible edit. Please return a set. and not a list
    """

    edit_one_set = set()

    edit_one_set.update(delete_letter(word))

    if allow_switches:
        edit_one_set.update(switch_letter(word))

    edit_one_set.update(replace_letter(word))
    edit_one_set.update(insert_letter(word))

    return edit_one_set

# test the edit_one_letter function

tmp_list = sorted(list(edit_one_letter("at")))

print(tmp_list)


def edit_two_letters(word, allow_switches=True):
    """
    :param word:
    :param allow_switches:
    :return:
    """
    edit_two_set = set()

    edit_one = edit_one_letter(word, allow_switches=allow_switches)

    for w in edit_one:
        if w:
            edit_two = edit_one_letter(w, allow_switches=allow_switches)
            edit_two_set.update(edit_two)

    return edit_two_set

print(edit_two_letters("a"))


def get_corrections(word, probs, vocab, n=2, verbose=True):
    """
    :param word: a user entered word to check for suggestions
    :param probs: a dictionary that maps each word to its probability in the corpus
    :param vocab: a set containing all the vocabulary
    :param n: number of possible word corrections you want returned in the dictionary
    :param verbose:
    :return: n_best: a list of tuples with most probable n corrected words and their probablilites
    """

    suggestions = []
    n_best = []

    suggestions = list((word in vocab and word) or edit_one_letter(word).intersection(vocab) or edit_two_letters(word).intersection(vocab))

    n_best = [[s, probs[s]] for s in list(reversed(suggestions))]

    if verbose:
        print(f"Suggestiongs: {suggestions}")

    return n_best


def min_edit_distance(source, target, ins_cost=1, del_cost=1, rep_cost=2):
    """
    :param source: a string which we are staring with
    :param target: the string that we want to end with
    :param ins_cost: an integer setting the inserting cost
    :param del_cost: an integer setting the deletion cost
    :param rep_cost: and integer setting the replacement cost
    :return: D: a matrix of len(source)+1 by len(target)+1 containing minimum edit distances
    med: the minimum edit distance (med) required to convert the source string to the target string
    """

    # use delete and insertion cost as 1
    m = len(source)
    n = len(target)

    # initialize the cost matrix with zeros and dimensions (m+1, n+1)
    D = np.zeros((m+1, n+1), dtype=int)

    # fill in column 0, from row 1 to row m both inclusive
    for row in range(1, m+1):
        D[row, 0] = D[row-1, 0] + del_cost

    # fill in row 0 for all columns from 1 to n, both inclusive
    for col in range(1, n+1):
        D[0, col] = D[0, col-1] + ins_cost

    for row in range(1, m+1):
        # loop through row 1 to m, both inclusive
        for col in range(1, n+1):
            # initialize r_cost to the 'replace' cost that is passed into this function
            r_cost = rep_cost
            # check to see if source character at the previous row
            # matches the target character at the previous column,
            if source[row-1]==target[col-1]:
                # update the replacement cost to 0 if source and target are the same
                r_cost = 0

            # update the cost at row, col based on previous entries in the cost matrix
            # refer to the equation calculate for D[i,j] (the minimum of three calculated costs)

            D[row, col] = min([D[row-1, col]+del_cost, D[row, col-1]+ins_cost, D[row-1, col-1]+r_cost])

    med = D[m,n]

    return D, med

'''
Test the med function
'''
source = 'paly'
target = 'stay'
matrix, med = min_edit_distance(source, target)

print(f"minimum edits: {med}")

idx = list('#'+source)
cols = list('#'+target)

print(pd.DataFrame(matrix, index=idx, columns=cols))







