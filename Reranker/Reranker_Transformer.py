from typing import List, Dict

import numpy as np
from simpletransformers.classification import ClassificationModel

from Reranker.Reranker import Reranker


class Transformer_Reranker(Reranker):
    def __init__(self, model_path: str, model_name: str = 'longformer', max_seq_length: int = None,
                 is_cased: bool = False, own_model_args: dict = None):
        super().__init__()
        if max_seq_length is None:
            max_seq_length = 4096 if model_name is 'longformer' else 512
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
                citation_context_fields: List[str] = ("title", "abstract", "citation_context"),
                use_year: bool = None) -> List[Dict[str, str]]:
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

            model_input.append([citation_context_rep, candidate_paper_rep])

        return model_input

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
