from typing import List, Dict, Tuple


def preprocessing_to_prefetcher(raw_input: Dict[str, List[str]]) -> List[Tuple[List[str], str]]:
    """Transforms the extracted global citation structure into a list of queries for the prefetcher module

    :param raw_input: The input from the structure extraction module
    :return: A list of queries for the prefetcher module
    """
    return [(v, k) for k, v in raw_input.items()]


def preprocessing_to_citeworthiness_detection(raw_input: Dict[str, List[List[Tuple[str, bool]]]]) -> Dict[
    str, List[List[Tuple[str, str]]]]:
    """Transforms the extracted sentences structure into dict with  lists of queries for the citeworth module
    Notice how the entries of the dictionary will be lists of lists, to account that a section can have multiple
    paragraphs.

    :param raw_input: The input from the structure extraction module
    :return: A list of queries for the citeworth module
    """
    queries = dict()
    for section in raw_input.keys():
        section_queries = []
        for paragraph_sentences in raw_input[section]:
            # A query for the citworth module consists of a unique id and the sentence itself
            paragraph_queries = [(str(i), s) for i, (s, _) in enumerate(paragraph_sentences)]
            section_queries.append(paragraph_queries)
        queries[section] = section_queries
    return queries


def prefetcher_to_reranker(candidate_papers: List[str], papers_info: Dict[str,Dict[str,str]]) -> List[Dict[str,str]]:
    """Transforms the candidate papers  into lists of dicts for the reranker module.
        Each dictionary has to contain the keys "id", "title", "abstract" and (optionally) "year".

            :param candidate_papers: List of Candidate papers to be processed
            :param papers_info: Papers info file for the dataset

            :return: A list of queries for the reranker module
            """
    results = []
    for candidate_paper in candidate_papers:
        entry = dict()
        entry['id'] = candidate_paper
        entry['title'] = papers_info[candidate_paper]['paper_title']
        entry['abstract'] = papers_info[candidate_paper]['paper_abstract']
        entry['year'] = papers_info[candidate_paper]['paper_year']
        results.append(entry)
    return results


def citeworthiness_detection_to_reranker(sentences: List[str], paragraph: str, section: str, abstract: str, title: str) -> List[Dict[str,str]]:
    """Transforms the sentences  into d  lists of queries for the reranker module.
    Each query is a dict which contains the keys:
    "id", "title", "abstract", "citation_context", "paragraph", "section"

        :param sentences: List of Sentences to be processed
        :param paragraph: Paragraph Text for the current sentences
        :param section: Section for the current sentences
        :param abstract: Abstract of the current paper
        :param title: Title of the current paper

        :return: A list of queries for the reranker module
        """
    res = []
    for idx, sentence in enumerate(sentences):
        entry = dict()
        entry['id'] = str(idx)
        entry['title'] = title
        entry['abstract'] = abstract
        entry['citation_context'] = sentence
        entry['paragraph'] = paragraph
        entry['section'] = section
        res.append(entry)
    return res


def reranker_to_output():
    # TODO: print/log to command line / file, how should output look like (which information and how to represent it)?
    raise NotImplementedError
