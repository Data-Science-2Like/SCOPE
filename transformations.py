def preprocessing_to_prefetcher():
    # TODO
    raise NotImplementedError


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
