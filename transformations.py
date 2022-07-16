from typing import List,Dict, Tuple


def preprocessing_to_prefetcher(raw_input: Dict[str,List[str]]):

    results = []
    for section in raw_input.keys():
        results.append((raw_input[section],section))
    return results


def preprocessing_to_citeworthiness_detection():
    # TODO
    raise NotImplementedError


def prefetcher_to_reranker():
    # TODO
    raise NotImplementedError


def citeworthiness_detection_to_reranker():
    # TODO: take cite-worthy sentences from citeworthiness-detection and return citation_contexts for reranker
    raise NotImplementedError


def reranker_to_output():
    # TODO: print/log to command line / file, how should output look like (which information and how to represent it)?
    raise NotImplementedError
