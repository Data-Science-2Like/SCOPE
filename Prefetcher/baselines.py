""" Baselines """
import numpy as np
from numpy.random import rand

from models.base import Recommender
from models.datasets import BagsWithVocab, Bags
from rank_bm25 import BM25Okapi
from tqdm import tqdm

class RandomBaseline(Recommender):
    """ Random Baseline """

    def __str__(self):
        return "RNDM baseline"

    def train(self, X):
        pass

    def predict(self, X):
        X = X.tocsr()
        random_predictions = rand(X.shape[0], X.shape[1])
        return random_predictions


class Countbased(Recommender):
    """ Item Co-Occurrence """
    def __init__(self, order=1):
        super().__init__()
        self.order = order

    def __str__(self):
        s = "Count-based Predictor"
        s += " (order {})".format(self.order)
        return s

    def train(self, X):
        X = X.tocsr()
        # Construct cooccurrence matrix
        self.cooccurences = X.T @ X
        for __ in range(0, self.order - 1):
            self.cooccurences = self.cooccurences.T @ self.cooccurences

    def predict(self, X):
        # Sum up values of coocurrences
        X = X.tocsr()
        return X @ self.cooccurences


class MostPopular(Recommender):
    """ Most Popular """
    def __init__(self):
        self.most_popular = None

    def __str__(self):
        return "Most Popular baseline"

    def train(self, X):
        X = X.tocsr()
        x_sum = X.sum(0)
        self.most_popular = x_sum

    def predict(self, X):
        return np.broadcast_to(self.most_popular, X.size())


class BM25Baseline(Recommender):
    """ BM25 Baseline """

    def __str__(self):
        return "BM25 Baseline"

    def train(self, X):
        # there are some entries which dont have a title
        self.corpus = [X.owner_attributes['title'].get(id) for id in X.index2token.values()]

        self.tokenized_corpus = []
        self.lookup_table = [] # has length of records with title
        for i, entry in enumerate(self.corpus):
            if entry is not None:
                self.tokenized_corpus.append(entry.split(" "))
                self.lookup_table.append(i)


        self.bm25 = BM25Okapi(self.tokenized_corpus)
        pass

    def predict(self, X):
        predictions = []
        for query in X:
            query_titles = [self.corpus[id] for id in query]
            doc_scores = np.zeros(len(self.corpus))
            for title in query_titles:
                filled_scores = np.zeros(len(self.corpus))
                if title != None:
                    tokenized_query = title.split(" ")
                    part_scores = np.array(self.bm25.get_scores(tokenized_query))


                    if len(self.lookup_table) < len(self.corpus):
                        # bm25 only returns a score for record with titles
                        # fill the rest with zeros
                        for idx, val in enumerate(self.lookup_table):
                            filled_scores[val] = part_scores[idx]
                        doc_scores = doc_scores + filled_scores
                    else:
                        # all entries have a title
                        doc_scores = doc_scores + part_scores
            predictions.append(doc_scores)
        return np.array(predictions)
