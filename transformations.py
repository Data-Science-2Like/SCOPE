from typing import List, Dict, Tuple


def preprocessing_to_prefetcher(raw_input: Dict[str, List[str]]):
    results = []
    for section in raw_input.keys():
        results.append((raw_input[section], section))
    return results


def preprocessing_to_citeworthiness_detection(raw_input):
    results = dict()
    for section in raw_input.keys():
        sec_results = []
        for parag in raw_input[section]:
            tmp_result = []
            for idx, (sent, _) in enumerate(parag):
                tmp_result.append((str(idx), sent))
            sec_results.append(tmp_result)
        results[section] = sec_results
    return results


def prefetcher_to_reranker(candidate_papers: List[str], papers_info):
    # Each dictionary has to contain the keys "id", "title", "abstract" and (optionally) "year".

    results = []
    for candidate_paper in candidate_papers:
        entry = dict()
        entry['id'] = candidate_paper
        entry['title'] = papers_info[candidate_paper]['paper_title']
        entry['abstract'] = papers_info[candidate_paper]['paper_abstract']
        entry['year'] = papers_info[candidate_paper]['paper_year']

        results.append(entry)

    return results


def citeworthiness_detection_to_reranker(sentences, paragraph,section,abstract, title):
    #                              The Dictionary has to contain the keys listed in citation_context_fields.
    #                             However, the following keys form a complete list:
    #                                "id", "title", "abstract", "citation_context", "paragraph", "section"
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
