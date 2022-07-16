from typing import List, Dict

import numpy as np
from simpletransformers.classification import ClassificationModel

from Reranker.Reranker import Reranker


class Transformer_Reranker(Reranker):
    def __init__(self, model_path: str, model_name: str = 'longformer', max_seq_length: int = None,
                 is_cased: bool = False, own_model_args: dict = None):
        super().__init__()
        if max_seq_length is None:
            max_seq_length = 4096 if model_name == 'longformer' else 512
        model_args = {
            "eval_batch_size": 50,
            "do_lower_case": not is_cased,
            "max_seq_length": max_seq_length,
            "wandb_kwargs": {"mode": "offline"}
        }
        if own_model_args is not None:
            model_args.update(own_model_args)

        # load the pretrained model
        self.model = ClassificationModel(model_name, model_path, args=model_args)

    def predict(self, citation_context: Dict[str, str], candidate_papers: List[Dict[str, str]],
                citation_context_fields: List[str] = ("citation_context", "title", "abstract"),
                use_year: bool = None) -> List[Dict[str, str]]:
        # mask citation context in paragraph
        if "paragraph" in citation_context_fields:
            citation_context["paragraph"] = citation_context["paragraph"].replace(citation_context["citation_context"],
                                                                                  "TARGETSENT")

        # process the input
        if use_year is None:
            use_year = "section" in citation_context_fields
        model_input = self._transform_predict_input_to_model_input(citation_context, candidate_papers,
                                                                   citation_context_fields, use_year)

        # perform the prediction
        _, raw_outputs = self.model.predict(model_input)
        ranking_scores = self._sigmoid(raw_outputs)

        # process the output
        relevance_idx = np.argsort(-ranking_scores)
        ranked_candidate_papers = [candidate_papers[i] for i in relevance_idx]
        return ranked_candidate_papers

    def _transform_predict_input_to_model_input(self, citation_context: Dict[str, str],
                                                candidate_papers: List[Dict[str, str]],
                                                citation_context_fields: List[str], use_year: bool):
        model_input = []

        # representation of citation context
        citation_context_rep = self._create_citation_context_representation(citation_context, citation_context_fields)

        for candidate_paper in candidate_papers:
            # representation of candidate paper
            candidate_paper_rep = self._create_candidate_paper_representation(candidate_paper, use_year)

            # perform custom truncation preprocessing
            citation_context_rep, candidate_paper_rep = self._perform_truncation_preprocessing(citation_context,
                                                                                               citation_context_rep,
                                                                                               candidate_paper_rep,
                                                                                               citation_context_fields)

            model_input.append([citation_context_rep, candidate_paper_rep])

        return model_input

    def _perform_truncation_preprocessing(self, query: Dict[str, str], query_rep: str, document_rep: str,
                                          query_fields: List[str], max_input_len: int = 512):
        if "paragraph" not in query_fields:
            # no truncation preprocessing required, longest-first truncation is sufficient
            return query_rep, document_rep
        # heuristic: one word = one token
        query_len = len(query_rep.split(" "))
        document_len = len(document_rep.split(" "))
        input_len = query_len + document_len
        max_input_len -= 3  # three special tokens (1x CLS, 2x SEP)
        truncation_len = input_len - max_input_len

        if truncation_len > 0:
            # longest-first truncation preprocessing with special treatment of paragraph

            # find out how many words need to be truncated from query and document entry respectively
            query_trunc_len = 0
            doc_trunc_len = 0
            query_document_len_diff = query_len - document_len
            if abs(query_document_len_diff) >= truncation_len:
                # either query or document needs be truncated
                if query_document_len_diff > 0:
                    query_trunc_len = truncation_len
                else:
                    doc_trunc_len = -truncation_len
            else:
                # query and document need to be truncated
                if query_document_len_diff > 0:
                    query_trunc_len = query_document_len_diff
                else:
                    doc_trunc_len = -query_document_len_diff
                truncation_len -= abs(query_document_len_diff)
                # remaining truncation_len is split equally between query and document
                truncation_len_smaller_half = int(truncation_len / 2)
                query_trunc_len += truncation_len_smaller_half
                doc_trunc_len += (truncation_len - truncation_len_smaller_half)

            # document / candidate paper -> remove from the end (abstract)
            if doc_trunc_len > 0:
                document_rep = document_rep.rsplit(" ", doc_trunc_len)[0]

            # query / citation context -> remove from the paragraph such text around citation context is preserved
            if query_trunc_len > 0:
                paragraph = query["paragraph"]
                sent_idx = paragraph.find("TARGETSENT")
                paragraph_len = len(paragraph.split(" "))
                paragraph_aimed_len_around = paragraph_len - query_trunc_len - 1
                if paragraph_aimed_len_around <= 0:
                    raise Exception("We did not expect that the whole paragraph or even more needs to be truncated.")
                right = int(paragraph_aimed_len_around / 2)
                left = paragraph_aimed_len_around - right
                if sent_idx + right >= paragraph_len:
                    # there are not enough words to the right
                    right = (paragraph_len - 1) - sent_idx
                    left = paragraph_aimed_len_around - right
                elif sent_idx - left < 0:
                    # there are not enough words to the left
                    left = sent_idx
                    right = paragraph_aimed_len_around - left
                paragraph = paragraph[sent_idx - left:sent_idx + right + 1]
                query_rep = self._create_citation_context_representation(query, query_fields,
                                                                         truncated_paragraph=paragraph)

        return query_rep, document_rep

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
