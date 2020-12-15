import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import get_vectors

import nltk
from gensim.models import KeyedVectors

#embeddings = KeyedVectors.load_word2vec_format('./GoogleNews-xxx-xxx.bin'. binary=True)

def consine_similarity(A, B):
    """
    :param A: a numpy array which corresponds to a word vector
    :param B: a numpy array which corresponds to a word vector
    :return: numerical number representing the cosine similarity between A and B
    """
    dot = np.dot(A, B)
    norma = np.sqrt(np.dot(A, A))
    normb = np.sqrt(np.dot(B, B))
    cos = dot/ (norma*normb)
    return cos


def euclidean(A, B):
    """
    :param A: a numpy array which corresponds to a word vector
    :param B: a numpy array which corresponds to a word vector
    :return: numerical number representing the Euclidean distance between A and B
    """
    d = np.linalg.norm(A-B)
    return d


def compute_pca(X, n_components=2):
    """
    :param X: of dimension (m, n) where each row corresponds to a word vector
    :param n_components: number of pca components we want
    :return: X_reduced: data transformed in 2 dims/columns + regenerated original data
    """
    X_demeaned = X - np.mean(X, axis=0)

    # calculate the covariance matrix
    covariance_matrix = np.cov(X_demeaned, rowvar=False)

    # calculate the eigenvectors & eigenvalues of the covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix, UPLO='L')

    # sort eigenvalue in increasing order (get the indices from the sort)
    idx_sorted = np.argsort(eigen_vals)

    # reverse the order so that it's from highest to lowest
    idx_sorted_decreasing = idx_sorted[::-1]

    # sort the eigen values by idx_sorted_decreasing
    eigen_vals_sorted = eigen_vals[idx_sorted_decreasing]

    # sort the eigenvectors by idx_sorted_decreasing indices
    eigen_vecs_sorted = eigen_vecs[:, idx_sorted_decreasing]

    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    eigen_vecs_subset = eigen_vecs_sorted[:, 0:n_components]

    # transform the data by multiplying the transpose of the eigenvectors
    # with the transpose of the de-meaned data
    # Then take the transpose of that product.
    X_reduced = np.dot(eigen_vecs_subset.transpose(), X_demeaned.transpose()).transpose()

    return X_reduced

'''
Testing the PCA function
'''
np.random.seed(1)
X= np.random.rand(3, 10)
X_reduced = compute_pca(X, n_components=2)
print(f"Your original matrix was {str(X.shape)} and it became : ")
print(X_reduced)