import numpy as np
import pprint as pp
import matplotlib.pyplot as plt

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

def basic_hash_table(value_l, n_buckets):

    def hash_function(value, n_buckets):
        return int(value) % n_buckets

    hash_table = {i:[] for i in range(n_buckets)} # initialize all the buckets in the hash table as many lists

    for value in value_l:
        hash_value = hash_function(value, n_buckets) # get the hash key for the given value
        hash_table[hash_value].append(value) # add the element to the corresponding bucket

    return hash_table

'''
Test the hash function
'''
value_l = [100, 10, 14, 17, 97]
hash_table_example = basic_hash_table(value_l, n_buckets=10)
pp.pprint(hash_table_example)


def side_of_plane(P, v):
    dotproduct = np.dot(P, v.T) # get the dot product P * v'
    sign_of_dot_product = np.sign(dotproduct) # the sign of the elements of the dotproduct matrix
    sign_of_dot_product_scalar = sign_of_dot_product.item()
    return sign_of_dot_product_scalar

'''
Test the side_of_plane function
'''
P = np.array([[1, 1]]) # sample plane
v1 = np.array([[1, 2]])
v2 = np.array([[-1, 1]])
v3 = np.array([[-2, -1]])

print(f"side_of_plane(P, v1): {side_of_plane(P, v1)}")
print(f"side_of_plane(P, v2): {side_of_plane(P, v2)}")
print(f"side_of_plane(P, v3): {side_of_plane(P, v3)}")


# Hash function with multiple planes
P1 = np.array([[1, 1]]) # first plane
P2 = np.array([[-1, 1]]) # second plane
P3 = np.array([[-1, -1]]) # third plane
P_1 = [P1, P2, P3] # list of arrays, it is multi plane
v = np.array([[2, 2]])

def hash_multi_plane(P_1, v):
    hash_value = 0
    for i, P in enumerate(P_1):
        sign = side_of_plane(P, v)
        hash_i = 1 if sign >= 0 else 0
        hash_value += 2**i * hash_i
    return hash_value

print(f"hash_multi_plane(P_1, v): {hash_multi_plane(P_1, v)}") # find the number of plane that contains this value

# Random planes
np.random.seed(1)
num_dimensions = 2
num_planes = 3
random_planes_matrix = np.random.normal(
    size=(num_planes,
          num_dimensions)
)

print(random_planes_matrix)

v = np.array([[2, 2]])

# side of the plane function. The result is matrix
def side_of_plane_matrix(P, v):
    dotproduct = np.dot(P, v.T)
    sign_of_dot_product = np.sign(dotproduct) # get a boolean value telling if the value in the cell is positive or negative
    return sign_of_dot_product


sides_l = side_of_plane_matrix(
    random_planes_matrix,
    v
)

print(f"sides_l : {sides_l}")

def hash_multi_plane_matrix(P, v, num_planes):
    sides_matrix = side_of_plane_matrix(P, v) # get the side of planes for P and v
    hash_value = 0
    for i in range(num_planes):
        sign = sides_matrix[i].item() # get the value inside the matrix cell
        hash_i = 1 if sign >= 0 else 0
        hash_value += 2**i * hash_i #2^i*hash_i
        return hash_value

print(hash_multi_plane_matrix(random_planes_matrix, v, num_planes))


# generate embedding and transform matrices
def get_matrices(en_fr, french_vecs, english_vecs):
    """
    :param en_fr: English to French dictionary
    :param french_vecs: French words to their corresponding word embeddings
    :param english_vecs: English words to their corresponding word embeddings
    Output:
        X: a matrix where the columns are the English embeddings
        Y: a matrix where the column corresponding to the French embeddings
        R: the projection matrix that minimizes the F rom ||X R - Y||^2
    """
    # X_1 and Y_1 are lists of the english and french word embeddings
    X_1 = list()
    Y_1 = list()

    english_set = english_vecs.keys()
    french_set = french_vecs.keys()

    french_words = set(en_fr.values())

    for en_word, fr_word in en_fr.items():
        # check that the french word has an embedding and that the english french dictionary
        if fr_word in french_set and en_word in english_set:
            # get the english embedding
            en_vec = english_vecs[en_word]

            # get the french embedding
            fr_vec = french_vecs[fr_word]

            # add the english embedding to the list
            X_1.append(en_vec)

            #add the french embedding to the list
            Y_1.append(fr_vec)

    # stack the vectors of X_1 into a matrix X
    X = np.vstack(X_1)
    Y = np.vstack(Y_1)

    return X, Y

def compute_loss(X, Y, R):
    """
    :param X: a matrix of dimension (m, n) where the columns are the English embeddings.
    :param Y: a matrix of dimension (m, n) where the columns corresponding to the French embeddings.
    :param R: a matrix of dimension (m, n) - transformation matrix from English to French vector space embeddings
    Outputs:
        L: a matrix of dimension (m, n) - the value of the loss function for given X, Y and R
    """
    # m is the number of rows in X
    m = X.shape[0]
    # diff is XR - Y
    diff = np.dot(X, R) - Y
    # diff_squared is the element-wise square of the difference
    diff_squared = diff**2
    # sum_diff_squared is the sum of the squared elements
    sum_diff_squared = np.sum(diff_squared)

    # loss i the sum_diff_squared divided by the number of examples (m)
    loss = sum_diff_squared/m

    return loss



def compute_gradient(X, Y, R):
    """
    :param X: a matrix of dimension (m,n) where the columns are the English embeddings
    :param Y: a matrix of dimension (m,n) where the columns corresponding to the French embeddings
    :param R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings
    Output:
        g: a matrix of dimension (n,n) - gradient of the loss function L for given X, Y and R
    """
    m = X.shape[0]

    # gradient is X^T(XR - Y)*2/m
    gradient = np.dot(X.transpose(), np.dot(X,R)-Y)*(2/m)

    return gradient

# k-nearest neighbors algorithm

def nearest_neighbor(v, candidates, k=1):
    """
    :param v: the vector we are trying to find the nearest neighbor
    :param candidates: a set of vectors where we will find the neighbors
    :param k: top k nearest neighbors to find
    Output:
        - k_idx: the indices of the top k closet vectors in sorted form
    """
    similarity_l = []

    # for each candidate vector
    for row in candidates:
        # get the cosine similarity
        cos_similarity = consine_similarity(v, row)

        # append the similarity to the list
        similarity_l.append(cos_similarity)

    # sort the similarity list and get the indices of the sorted list
    sorted_ids = np.argsort(similarity_l)

    # get the indices of the k most similar candidate vectors
    k_idx = sorted_ids[-k:]

    return k_idx

v = np.array([1, 0, 1])
candidates = np.array([[1, 0, 5], [-2, 5, 3], [2, 0, 1], [6, -9, 5], [9, 9, 9]])
print(candidates[nearest_neighbor(v, candidates, 3)])



def approximate_knn(doc_id, v, planes_l, k=1, num_universes_to_use=N_UNIVERSES):
    """
    Search for k-NN using hashes
    """
    assert num_universes_to_use <= N_UNIVERSES

    # vectors that will be checked as possible nearest neighbor
    vecs_to_consider_l = list()

    # list of document IDS
    ids_to_consider_l = list()

    # create a set for ids to consider, for faster checking if a document ID already exists in the set
    ids_to_consider_set = set()

    # loop through the universe of planes
    for universe_id in range(num_universes_to_use):

        # get the set of planes from the planes_l list, for this particular universe_id
        planes = planes_l[universe_id]

        # get the hash value of the vector for this set of planes
        hash_value = hash_value_of_vector(v, planes)

        #get the hash table of the vector for this set of planes
        hash_table = hash_tables[universe_id]

        #get the list of document vectors for this hash table, where the key is the hash_value
        document_vectors_l = hash_table[hash_value]

        # get the id_table for this particular universe_id
        id_table = id_tables[universe_id]

        # get the subset of documents to consider as nearest neighbors from this id_table dictionary
        new_ids_to_consider = id_table[hash_value]

        # remove the id of the document that we're searching
        if doc_id in new_ids_to_consider:
            new_ids_to_consider.remove(doc_id)
            print(f"removed doc_id {doc_id} of input vector from new_ids_to_search")

        # loop through the subset of document vectors to consider
        for i, new_id in enumerate(new_ids_to_consider):

            # if the document ID is not yet in the set ids_to_consider
            if new_id not in ids_to_consider_set:
                # access document_vectors_l list at index i to get the embedding
                # then append it to the list of vectors to consider as possible nearest neigbors
                document_vectors_at_i = document_vectors_l[i]

                # append the new_id (the index for the document) to the list of ids to consider
                vecs_to_consider_l.append(document_vectors_at_i)
                ids_to_consider_l.append(new_id)

                # also add the new_id to the set of ids to consider
                # (use this to check if new_id is not already in the IDs to consider)
                ids_to_consider_set.add(new_id)
    # Now run k-NN on the smaller set of vecs-to-consider
    print(f"Fast considering {len(vecs_to_consider_l)} vecs")

    # convert the vecs to consider set to a list, then to a numpy array
    vecs_to_consider_arr = np.array(vecs_to_consider_l)

    # call nearest neghbors on the reduces list of candidates vectors
    nearest_neighbor_idx_1 = nearest_neighbor(v, vecs_to_consider_arr, k=k)

    # use the nearest neighbors index list as indices into the ids to consider
    # create a list of nearest neighbors by the document ids

    nearest_neighbor_ids = [ids_to_consider_l[idx]
                            for idx in nearest_neighbor_idx_1
                            ]

    return nearest_neighbor_ids





