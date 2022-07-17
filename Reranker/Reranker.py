from typing import List, Dict

import regex as re


class Reranker:
    def __init__(self):
        pass

    def predict(self, citation_context: Dict[str, str], candidate_papers: List[Dict[str, str]],
                citation_context_fields: List[str], use_year: bool) -> List[Dict[str, str]]:
        """
        :param citation_context: Representation for a single citation context.
                                 The Dictionary has to contain the keys listed in citation_context_fields.
                                 However, the following keys form a complete list:
                                    "id", "title", "abstract", "citation_context", "paragraph", "section"
        :param candidate_papers: Each candidate paper is represented by a dictionary within the list.
                                 Each dictionary has to contain the keys "id", "title", "abstract" and (optionally) "year".
        :param citation_context_fields: The herein named fields are extracted in the given order from citation_context
                                            and serve as an input to the model.
        :param use_year: Whether to make use of the year key in candidate_papers as an input to the model.
                         By default: Set to True if citation_context_fields contains "section", otherwise False.
        :return: list of candidate paper dictionaries sorted by their relevance (from relevant to not relevant)
        """
        raise NotImplementedError

    @staticmethod
    def _create_citation_context_representation(citation_context: Dict[str, str], citation_context_fields: List[str],
                                                truncated_paragraph=None):
        citation_context_rep = ""
        for field in citation_context_fields:
            if field == "paragraph" and truncated_paragraph is not None:
                citation_context_rep += truncated_paragraph + " "
            else:
                citation_context_rep += citation_context[field] + " "
        citation_context_rep = citation_context_rep[:-1]
        citation_context_rep = citation_context_rep.replace("\n", " ")
        citation_context_rep = re.sub(" +", " ", citation_context_rep)
        return citation_context_rep

    @staticmethod
    def _create_candidate_paper_representation(candidate_paper: Dict[str, str], use_year: bool):
        if use_year:
            candidate_paper_rep = candidate_paper["title"] + " " + candidate_paper["year"] + " " + candidate_paper["abstract"]
        else:
            candidate_paper_rep = candidate_paper["title"] + " " + candidate_paper["abstract"]
        candidate_paper_rep = candidate_paper_rep.replace("\n", " ")
        candidate_paper_rep = re.sub(" +", " ", candidate_paper_rep)
        return candidate_paper_rep
