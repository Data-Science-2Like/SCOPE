from typing import List, Dict

import numpy as np

from Reranker.Reranker import Reranker
from Reranker.gensim_summarization_bm25 import BM25


class LocalBM25(Reranker, BM25):
    def __init__(self, all_papers: List[Dict[str, str]],
                 k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25,
                 citation_context_fields: List[str] = ("citation_context", "title", "paragraph"),
                 mask_citation_context_in_paragraph: bool = False):
        """
        all_papers are defined as in predict(...).
        However, all_papers does not only contain the prefetched candidate_papers but all possible candidate_papers.
        """
        Reranker.__init__(self, citation_context_fields)
        self.mask_citation_context_in_paragraph = mask_citation_context_in_paragraph

        # create corpus for BM25
        self.corpus = {}
        for paper in all_papers:
            self.corpus[paper["id"]] = self._create_candidate_paper_representation(paper)
        self.paperid_to_corpusidx = {paper_id: corpus_idx for corpus_idx, paper_id in enumerate(self.corpus.keys())}

        word_corpus = [value.split() for value in self.corpus.values()]
        BM25.__init__(self, word_corpus, k1, b, epsilon)

    def predict(self, citation_context: Dict[str, str], candidate_papers: List[Dict[str, str]]) -> List[Dict[str, str]]:
        # mask citation context in paragraph
        if self.mask_citation_context_in_paragraph and "paragraph" in self.citation_context_fields:
            citation_context["paragraph"] = citation_context["paragraph"].replace(citation_context["citation_context"],
                                                                                  "TARGETSENT")

        # process the input
        query = self._create_citation_context_representation(citation_context)
        word_query = query.split()

        # perform the prediction
        candidate_papers_corpusidx = [self.paperid_to_corpusidx[paper["id"]] for paper in candidate_papers]
        ranking_scores = [self.get_score(word_query, idx) for idx in candidate_papers_corpusidx]

        # process the output
        relevance_idx = np.argsort(-np.asarray(ranking_scores))
        ranked_candidate_papers = [candidate_papers[i] for i in relevance_idx]
        return ranked_candidate_papers
