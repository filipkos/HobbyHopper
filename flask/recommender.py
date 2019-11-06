import numpy as np

class CosineSimilarityRecommender(object):
    """
    Cosine similarity recommender.
    """

    def __init__(self, similarity= None, mn= 0, sig= 1):
        self.similarity = similarity
        self.mn = mn
        self.sig = sig

    def cos_similarity(self, in_matrix, regularizer= 1e-9):
        products = in_matrix.T.dot(in_matrix)
        norms = np.sqrt(np.diagonal(products))[np.newaxis, :] + regularizer
        return np.maximum(products / norms / norms.T, regularizer)

    def jaccard_index(self, in_matrix, regularizer= 1e-9):
        """
        Here we assume in_matrix is in form of zeroes and ones.
        """
        products = in_matrix.T.dot(in_matrix)
        norms =  in_matrix.sum(axis=0)
        norms = norms[np.newaxis, :] + norms[:, np.newaxis] - products + regularizer
        return products / norms

    def second_neighbors(self, similarity1, regularizer= 1e-9):
        norms = np.abs(similarity1).sum(axis=1)[np.newaxis, :]
        return (similarity1 / norms).dot(similarity1)

    def fit(self, in_matrix, normalization='euclid', scaling=False, alpha=0, regularizer= 1e-9):
        if normalization == 'euclid':
            if scaling:
                self.mn = in_matrix.mean(axis = 0)
                self.sig = np.sqrt(in_matrix.var(axis = 0))
            similarity1 = self.cos_similarity((in_matrix - self.mn) / (self.sig + regularizer), regularizer)
        elif normalization == 'jaccard':
            similarity1 = self.jaccard_index(in_matrix, regularizer)
        else:
            raise ValueError("The normalization parameter must be one of 'euclid' or 'jaccard'.")
        norms = np.abs(similarity1).sum(axis=1)[np.newaxis, :]
        similarity2 = self.second_neighbors(similarity1, regularizer)
        self.similarity = (similarity1 + alpha * similarity2) / norms

    def predict(self, test_matrix, regularizer= 1e-9):
        if len(test_matrix.shape) < 2:
            test_matrix = test_matrix[np.newaxis, :]
        test_matrix = (test_matrix - self.mn) / (self.sig + regularizer)
        predictions = test_matrix.dot(self.similarity)
        predictions = predictions * self.sig + self.mn
        return predictions


class BaselineRecommender(object):
    """
    Baseline recommender that always recommends most popular items.
    """
    def __init__(self, scores=None):
        self.scores = scores

    def fit(self, in_matrix):
        self.scores = in_matrix.sum(axis=0) / in_matrix.shape[0]

    def predict(self, test_matrix):
        if len(test_matrix.shape) < 2:
            test_matrix = test_matrix[np.newaxis, :]
        return np.array([self.scores for row in test_matrix])
