'''
Calculating the Bilingual Evaluation Understudy (BLEU) score
'''

'''
Part 1 : BLEU Score
'''
import numpy as np
import nltk
nltk.download('punkt')
from nltk.util import ngrams
from collections import Counter
import sacrebleu
import matplotlib.pyplot as plt



reference = "The NASA Opportunity rover is battling a massive dust storm on planet Mars."
candiate_1 = "The Oppourtunity rover is combating a big sandstrom on planet Mars."
candiate_2 = "A NASA rover is fighting a massive storm on planet Mars."

tokenized_ref = nltk.word_tokenize(reference.lower())
tokenized_cand_1 = nltk.word_tokenize(candiate_1.lower())
tokenized_cand_2 = nltk.word_tokenize(candiate_2.lower())


# step 1: compute the Brevity Penalty
def brevity_penalty(reference, candidate):
    ref_length = len(reference)
    can_length = len(candidate)

    # Brevity Penalty
    if ref_length>can_length:
        BP = 1
    else:
        penalty = 1 - (ref_length/can_length)
        BP = np.exp(penalty)

    return BP

# step 2: computing the precision
def clipped_precision(reference, candidate):
    """Blue score function given a original and a machine translated sentences"""

    clipped_precision_score = []

    for i in range(1, 5):
        candidate_n_gram = Counter(
            ngrams(candidate, i)
        )   # counts of n-gram n=1,..4 tokens for the candidate

        reference_n_gram = Counter(
            ngrams(reference, i)
        )   # counts of n-gram n=1,..4 tokens for the reference

        c = sum(
            reference_n_gram.values()
        )   # sum of the values of the reference the denominator in the precision formula

        for j in reference_n_gram:  # for every n_gram token in the reference
            for j in candidate_n_gram:  # check if it is in the candidate n-gram
                if (reference_n_gram[j]>candidate_n_gram[j]):
                    # if the count of the reference n-gram is bigger
                    # than the corresponding count in the candidate n-gram
                    reference_n_gram[j] = candidate_n_gram[j]
                    # then set the count of the reference n-gram to be equal
                    # to the count of the candidate n-gram
                else:
                    reference_n_gram[j] = 0 # else reference n-gram = 0

        clipped_precision_score.append(sum(reference_n_gram.values())/c)

    weights = [0.25]*4

    s = (w_i*np.log(p_i) for w_i, p_i in zip(weights, clipped_precision_score))
    s = np.exp(np.sum(s))

    return s

# step 3: Computing the BLEU score
def bleu_score(reference, candidate):
    BP = brevity_penalty(reference, candidate)
    precision = clipped_precision(reference, candidate)
    return BP * precision

# step 4: testing with our example reference and candidates sentences
print(
    "Results reference versus candidate 1 our own code BLEU:",
    round(bleu_score(tokenized_ref, tokenized_cand_1)*100, 1),
)






















