import json
import random
import os

from Prefetcher.log import log
from pathlib import Path
from Prefetcher.datasets import Bags, BagsWithVocab, corrupt_sets
from Prefetcher.transforms import lists2sparse

import numpy as np

CITE5_PATH = Path("./citeworth/aae_recommender_with_section_info_v5.jsonl")

SYNONYM_DICT = {
    "abstract": "abstract",
    "introduction": "introduction",
    "intro": "introduction",
    "overview": "introduction",
    "motivation": "introduction",
    "problem motivation": "introduction",
    "related work": "related work",
    "related works": "related work",
    "previous work": "related work",
    "literature": "related work",
    "background": "related work",
    "literature review": "related work",
    "state of the art": "related work",
    "current state of research": "related work",
    "requirement": "related work",
    "theory basics": "theory basics",
    "techniques": "techniques",
    "experiment": "experiment",
    "experiments": "experiment",
    "experiments and results": "experiment",
    "experimental result": "experiment",
    "experimental results": "experiment",
    "experimental setup": "experiment",
    "result": "experiment",
    "results": "experiment",
    "evaluation": "experiment",
    "performance evaluation": "experiment",
    "experiment and result": "experiment",
    "analysis": "experiment",
    "methodology": "method",
    "method": "method",
    "methods": "method",
    "material and method": "method",
    "material and methods": "method",
    "proposed method": "method",
    "evaluation methodology": "method",
    "procedure": "method",
    "implementation": "method",
    "experimental design": "method",
    "implementation detail": "method",
    "implementation details": "method",
    "system model": "method",
    "definition": "definition",
    "data set": "data set",
    "solution": "solution",
    "discussion": "discussion",
    "discussions": "discussion",
    "limitation": "discussion",
    "limitations": "discussion",
    "discussion and conclusion": "discussion",
    "discussion and conclusions": "discussion",
    "result and discussion": "discussion",
    "results and discussion": "discussion",
    "results and discussions": "discussion",
    "results and analysis": "discussion",
    "future work": "conclusion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "summary": "conclusion",
    "conclusion and outlook": "conclusion",
    "conclusion and future work": "conclusion",
    "conclusions and future work": "conclusion",
    "concluding remark": "conclusion"
}

PAPER_INFO = ['title', 'venue', 'author']


def rename_key(data, old: str, new: str) -> None:
    for entry in data:
        entry[new] = entry.pop(old)


def load_citeworth(path, use_synonym_dict=True):
    """ Loads a single file """
    print("Loading citworth data from", path)
    data = list()
    with open(path, 'r') as f:
        for i, l in enumerate(f):
            data.append(json.loads(l.strip()))

    rename_key(data, 'paper_title', 'title')
    # rename_key(data,'paper_authors','authors')
    rename_key(data, 'paper_year', 'year')
    # rename_key(data, 'outgoing_citations', 'references')
    rename_key(data, 'outgoing_citations_in_paragraph', 'references')
    rename_key(data, 'paper_id', 'id')

    # apply synonym dict
    if use_synonym_dict:
        for entry in data:
            entry['section_title'] = SYNONYM_DICT[entry['section_title'].lower()]

    for entry in data:
        if entry['year'] and entry['year'] != None:
            entry['year'] = int(entry['year'])

    # get all ids
    all_ids = []
    for entry in data:
        all_ids.append(entry['id'])

    return data


def aggregate_paper_info(paper, attributes):
    acc = []
    for attribute in attributes:
        if attribute in paper:
            acc.append(paper[attribute])
    return ' '.join(acc)


def unpack_papers(papers, aggregate=None, end_year=-1):
    """
    Unpacks list of papers in a way that is compatible with our Bags dataset
    format. It is not mandatory that papers are sorted.
    """
    # Assume track_uri is primary key for track
    if aggregate is not None:
        for attr in aggregate:
            assert attr in PAPER_INFO

    bags_of_refs, ids, side_info, years, authors, venue, sections = [], [], {}, {}, {}, {}, {}
    title_cnt = author_cnt = ref_cnt = venue_cnt = one_ref_cnt = year_cnt = section_cnt = 0
    for paper in papers:

        if 0 < end_year <= int(paper['year']):
            continue

        # Extract ids
        ids.append(paper["id"])
        # Put all ids of cited papers in here
        try:
            # References may be missing
            bags_of_refs.append(paper["references"])
            if len(paper["references"]) > 0:
                ref_cnt += 1
            if len(paper["references"]) == 1:
                one_ref_cnt += 1
        except KeyError:
            bags_of_refs.append([])
        # Use dict here such that we can also deal with unsorted ids
        try:
            side_info[paper["id"]] = paper["title"]
            if paper["title"] != "":
                title_cnt += 1
        except KeyError:
            side_info[paper["id"]] = ""
        try:
            years[paper["id"]] = paper["year"]
            if paper["year"] == None:
                years[paper["id"]] = -1
            if paper["year"] != None and paper["year"] > 0:
                year_cnt += 1
        except KeyError:
            years[paper["id"]] = -1
        try:
            authors[paper["id"]] = paper["authors"]
        except KeyError:
            authors[paper["id"]] = []
        try:
            venue[paper["id"]] = paper["venue"]
        except KeyError:
            venue[paper["id"]] = ""
        try:
            sections[paper["id"]] = paper["section_title"]
        except KeyError:
            sections[paper["id"]] = []

        try:
            if len(paper["authors"]) > 0:
                author_cnt += 1
        except KeyError:
            pass
        try:
            if len(paper["venue"]) > 0:
                venue_cnt += 1
        except KeyError:
            pass

        try:
            if len(paper["section_title"]) > 0:
                section_cnt += 1
        except KeyError:
            pass

        # We could assemble even more side info here from the track names
        if aggregate is not None:
            aggregated_paper_info = aggregate_paper_info(paper, aggregate)
            side_info[paper["id"]] += ' ' + aggregated_paper_info

    log(
        "Metadata-fields' frequencies: references={}, title={}, authors={}, venue={}, year={}, sections={} one-reference={}"
            .format(ref_cnt / len(papers), title_cnt / len(papers), author_cnt / len(papers), venue_cnt / len(papers),
                    year_cnt / len(papers), section_cnt / len(papers), one_ref_cnt / len(papers)))

    # bag_of_refs and ids should have corresponding indices
    # In side info the id is the key
    # Re-use 'title' and year here because methods rely on it
    return bags_of_refs, ids, {"title": side_info, "year": years, "author": authors, "venue": venue,
                               "section_title": sections}


def load_dataset(year, val_year=None, min_count=None, drop=1):
    """ Main function for training and evaluating AAE methods on DBLP data """
    dataset = 'Our S2ORC'
    use_sdict = True

    curr_dir = os.path.dirname(os.path.realpath(__file__))

    papers = load_citeworth(os.path.join(curr_dir,CITE5_PATH), use_sdict)

    print("Unpacking {} data...".format(dataset))
    bags_of_papers, ids, side_info = unpack_papers(papers)
    del papers
    bags = Bags(bags_of_papers, ids, side_info)

    log("Whole dataset:")
    log(bags)

    u = len(set(bags.owner_attributes['title']))
    log(f"Keeping {u} papers")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    train_set, test_set, = None, None
    if val_year is not None and val_year > 0:
        train_set, _, test_set = bags.train_val_test_split(val_year=val_year,
                                                           test_year=year)
    else:
        train_set, test_set = bags.train_test_split(on_year=year)
    train_set, test_set = bags.train_test_split(on_year=year)

    log("=" * 80)
    log("Train:", train_set)
    log("Test:", test_set)
    train_set = train_set.build_vocab(min_count=min_count,
                                      max_features=None,
                                      apply=True)
    test_set = test_set.apply_vocab(train_set.vocab)

    min_elements = 1
    # Train and test sets are now BagsWithVocab
    train_set.prune_(min_elements=min_elements)
    test_set.prune_(min_elements=min_elements)
    log("Train:", train_set)
    log("Test:", test_set)
    log("Drop parameter:", drop)

    noisy, missing = corrupt_sets(test_set.data, drop=drop)

    assert len(noisy) == len(missing) == len(test_set)

    test_set.data = noisy
    log("-" * 80)

    # just store for not recomputing the stuff
    x_test = lists2sparse(noisy, train_set.size(1)).tocsr(copy=False)
    return train_set, x_test
