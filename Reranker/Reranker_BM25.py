from typing import List, Dict

import numpy as np

from Reranker.Reranker import Reranker
from gensim_summarization_bm25 import BM25


class LocalBM25(Reranker, BM25):
    def __init__(self, all_papers: List[Dict[str, str]], use_year: bool = False,
                 k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        """
        all_papers and use_year are defined as in predict(...).
        However, all_papers does not only contain the prefetched candidate_papers but all possible candidate_papers.
        """
        Reranker.__init__(self)

        # create corpus for BM25
        self.corpus = {}
        for paper in all_papers:
            self.corpus[paper["id"]] = self._create_candidate_paper_representation(paper, use_year)
        self.paperid_to_corpusidx = {paper_id: corpus_idx for corpus_idx, paper_id in enumerate(self.corpus.keys())}

        word_corpus = [value.split() for value in self.corpus.values()]
        BM25.__init__(self, word_corpus, k1, b, epsilon)

    def predict(self, citation_context: Dict[str, str], candidate_papers: List[Dict[str, str]],
                citation_context_fields: List[str] = ("citation_context", "title", "abstract"),
                use_year: bool = None) -> List[Dict[str, str]]:
        # process the input
        query = self._create_citation_context_representation(citation_context, citation_context_fields)
        word_query = query.split()

        # perform the prediction
        candidate_papers_corpusidx = [self.paperid_to_corpusidx[paper["id"]] for paper in candidate_papers]
        ranking_scores = [self.get_score(word_query, idx) for idx in candidate_papers_corpusidx]

        # process the output
        relevance_idx = np.argsort(-np.asarray(ranking_scores))
        ranked_candidate_papers = [candidate_papers[i] for i in relevance_idx]
        return ranked_candidate_papers
