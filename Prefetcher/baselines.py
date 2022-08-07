""" Baselines """
import numpy as np
from numpy.random import rand
from typing import List


from .datasets import BagsWithVocab, Bags
from .citeworth import load_dataset
from .prefetcher import Prefetcher
from rank_bm25 import BM25Okapi
from tqdm import tqdm


class RandomBaseline(Prefetcher):

    def __init__(self):
        super().__init__()
        bags, x_train = load_dataset(2019, 2018, 2)
        self.bags = bags


    def __str__(self):
        return "RNDM baseline"

    def predict(self, already_cited: List[str], section: str, k: int) -> List[str]:

        # transform into vocab index for aae recommender
        #internal_q = [[self.bags.vocab[id] for id in already_cited]]
        internal_q = []
        not_found = []
        for id in already_cited:
            if id in self.bags.vocab.keys():
                internal_q.append(self.bags.vocab[id])
            else:
                not_found.append(id)

        if len(not_found) > 0:
            print(f"Warning: Could not find the following cited keys: {not_found}")

        pred = self._predict([internal_q])[0]


        # sort predictions by their score
        preds_sorted = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)

        # get keys for predictions index
        return_keys = [self.bags.index2token[i] for i in preds_sorted[:k]]

        return return_keys

    def _predict(self, X):
        X = self.bags.tocsr(X)
        random_predictions = rand(X.shape[0], X.shape[1])
        return random_predictions


class Countbased(Prefetcher):
    """ Item Co-Occurrence """
    def __init__(self, order=1):
        super().__init__()
        self.order = order
        bags, x_train = load_dataset(2019, 2018, 2)
        self.bags = bags
        self.train(x_train)

    def __str__(self):
        s = "Count-based Predictor"
        s += " (order {})".format(self.order)
        return s

    def predict(self, already_cited: List[str], section: str, k: int) -> List[str]:

        # transform into vocab index for aae recommender
        #internal_q = [[self.bags.vocab[id] for id in already_cited]]
        internal_q = []
        not_found = []
        for id in already_cited:
            if id in self.bags.vocab.keys():
                internal_q.append(self.bags.vocab[id])
            else:
                not_found.append(id)

        if len(not_found) > 0:
            print(f"Warning: Could not find the following cited keys: {not_found}")

        pred = self._predict([internal_q])[0]


        # sort predictions by their score
        preds_sorted = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)

        # get keys for predictions index
        return_keys = [self.bags.index2token[i] for i in preds_sorted[:k]]

        return return_keys

    def train(self, X):
        X = self.bags.tocsr(X)
        # Construct cooccurrence matrix
        self.cooccurences = X.T @ X
        for __ in range(0, self.order - 1):
            self.cooccurences = self.cooccurences.T @ self.cooccurences

    def _predict(self, X):
        X = self.bags.tocsr(X)
        return X @ self.cooccurences


class MostPopular(Prefetcher):
    """ Most Popular """
    def __init__(self):
        super().__init__()
        self.most_popular = None
        bags, x_train = load_dataset(2019, 2018, 2)
        self.bags = bags
        self.train(x_train)

    def __str__(self):
        return "Most Popular baseline"

    def predict(self, already_cited: List[str], section: str, k: int) -> List[str]:

        # transform into vocab index for aae recommender
        #internal_q = [[self.bags.vocab[id] for id in already_cited]]
        internal_q = []
        not_found = []
        for id in already_cited:
            if id in self.bags.vocab.keys():
                internal_q.append(self.bags.vocab[id])
            else:
                not_found.append(id)

        if len(not_found) > 0:
            print(f"Warning: Could not find the following cited keys: {not_found}")

        pred = self._predict([internal_q])[0]


        # sort predictions by their score
        preds_sorted = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)

        # get keys for predictions index
        return_keys = [self.bags.index2token[i] for i in preds_sorted[:k]]

        return return_keys

    def train(self, X):
        X = self.bags.tocsr(X)
        x_sum = X.sum(0)
        self.most_popular = x_sum

    def _predict(self, X):
        return np.broadcast_to(self.most_popular, X.size())


class BM25Baseline(Prefetcher):
    """ BM25 Baseline """
    def __init__(self):
        super().__init__()
        self.most_popular = None
        bags, x_train = load_dataset(2019, 2018, 2)
        self.bags = bags
        self.train(self.bags)


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

    def predict(self, already_cited: List[str], section: str, k: int) -> List[str]:

        # transform into vocab index for aae recommender
        #internal_q = [[self.bags.vocab[id] for id in already_cited]]
        internal_q = []
        not_found = []
        for id in already_cited:
            if id in self.bags.vocab.keys():
                internal_q.append(self.bags.vocab[id])
            else:
                not_found.append(id)

        if len(not_found) > 0:
            print(f"Warning: Could not find the following cited keys: {not_found}")

        pred = self._predict([internal_q])[0]


        # sort predictions by their score
        preds_sorted = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)

        # get keys for predictions index
        return_keys = [self.bags.index2token[i] for i in preds_sorted[:k]]

        return return_keys


    def _predict(self, X):
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
