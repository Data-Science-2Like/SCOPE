import json
import os
import transformations
from PreprocessingWithStructureAnalysis.id_translator import IdTranslator
from PreprocessingWithStructureAnalysis.structure_extraction import StructureExtraction
import shutil
from Reranker.Reranker_Transformer import Transformer_Reranker

from typing import List, Dict

from Prefetcher.aae_recommender import AAERecommender

from CiteworthinessDetection.CiteWorth import CiteWorth

from Prefetcher.baselines import BM25Baseline

import transformations as tf

import sys

sys.setrecursionlimit(10000)

K = 2000


def load_papers_info(file):
    papers_info = dict()

    with open(file) as f:
        for line in f:
            entry = json.loads(line.strip())

            keep_keys = ['paper_title', 'paper_abstract', 'paper_year']

            papers_info[entry['paper_id']] = dict([(k, v) for k, v in entry.items() if k in keep_keys])

    return papers_info


def av(l):
    avg = sum(l) / len(l)
    return avg


def get_citeworth_array(ids, predict_ids):
    # citeworthy_sents = [s for s, label in zip(sentences, preds) if label == 1]
    res = [True if id in predict_ids else False for id in ids]
    return res

def collect_already_processed():
    already_processed = set()
    with open('results_experiments.jsonl') as f:
        for line in f:
            entry = json.loads(line.strip())
            already_processed.add(entry['paper_id'])
    print(f"Loaded {len(already_processed)} results")

GPU = 2

if __name__ == "__main__":

    extraction = StructureExtraction('../tex-expanded')

    # valid_ids = extraction.get_valid_ids()

    valid_ids = json.load(open('correct_ids.json'))

    already_processed = collect_already_processed()

    # print(f"Already loaded {len(loaded_ids)} papers sucessfully")
    idx = 0
    # last_loaded_id = '211007045'
    # while int(valid_ids[idx]) <= int(last_loaded_id):
    #    idx += 1
    # idx += 2
    # json.dump(valid_ids, open('valid_ids','w'))
    papers_info = load_papers_info('./s2orc/papers.jsonl')

    print(f"Found {len(valid_ids)} tex files with corresponding bibtex entry")

    prefetcher = AAERecommender('./Prefetcher/trained/aae.torch',True)
    # prefetcher = BM25Baseline()

    reranker = Transformer_Reranker('./Reranker/trained')

    # citworth = CiteWorth('./CiteworthinessDetection/trained/citeworth-ctx-section-always-seed1000.pth','always')
    citworth = CiteWorth('./CiteworthinessDetection/trained/citeworth-ctx-section-always-seed1000.pth', 'always')

    print("Loaded models successfully")
    # input("Press to continue")
    # idx = 0

    # Sentences which are citeworthy
    global_citeworthy_count = 0
    # Sentences which are non citeworthy
    global_non_citeworthy_count = 0
    # Sentences which need a citation, get found by citeworth and reranker
    global_correct_citation_count_r5 = 0
    global_correct_citation_count_r10 = 0
    # Sentences which need a citation and get found by the citeworthy module
    global_correct_citeworthy_count = 0
    # Sentences which don't need a citation and get found by the citeworthy module
    global_correct_non_citeworthy_count = 0
    # Sentences which don't need a citation but get labeled as such
    global_incorrect_citeworthy_count = 0
    # Sentences which need a citation but don't get labeled as such
    global_incorrect_non_citeworthy_count = 0
    # Sentences which need a citation, get labeled as citeworth, but reranker doesn't find the right thing
    global_incorrect_citation_count_r5 = 0
    global_incorrect_citation_count_r10 = 0

    sentence_counts = []
    # while int(valid_ids[idx]) < int('200412152'):
    #    idx += 1

    while True:
        if valid_ids[idx] in already_processed:
            idx += 1
            continue

        if not extraction.set_as_active(valid_ids[idx]):
            idx += 1
            continue

        # if int(valid_ids[idx]) > int('200512152'):
        #    exit()

        print(f"Loaded paper {valid_ids[idx]}")
        # else:
        #    loaded_ids.append(valid_ids[idx])
        #    json.dump(loaded_ids, open('loaded_ids.json', 'w'))
        #    sc = extraction.get_sentence_count()
        #    sentence_counts.append(sc)
        #    print(f"{av(sentence_counts)} lines per paper")
        #    idx += 1
        #    continue

        # Sentences which are citeworthy
        citeworthy_count = 0
        # Sentences which are non citeworthy
        non_citeworthy_count = 0
        # Sentences which need a citation, get found by citeworth and reranker
        correct_citation_count_r5 = 0
        correct_citation_count_r10 = 0
        # Sentences which need a citation and get found by the citeworthy module
        correct_citeworthy_count = 0
        # Sentences which don't need a citation and get found by the citeworthy module
        correct_non_citeworthy_count = 0
        # Sentences which don't need a citation but get labeled as such
        incorrect_citeworthy_count = 0
        # Sentences which need a citation but don't get labeled as such
        incorrect_non_citeworthy_count = 0
        # Sentences which need a citation, get labeled as citeworth, but reranker doesn't find the right thing
        incorrect_citation_count_r5 = 0
        incorrect_citation_count_r10 = 0

        # Extract Global Information
        paper_section_list = extraction.get_section_titles()
        paper_title = extraction.get_title()
        paper_abstract = extraction.get_abstract()

        # Extract Information for Prefetcher
        paper_global_citations = extraction.get_citations_by_sections()

        # Extract Information for Citeworth
        paper_sentences_with_citeworth = extraction.get_section_text_citeworth()

        # Extract Information for Reranker
        paper_paragraphs = extraction.get_section_text_paragraph()
        paper_sentences_with_correct_citations = extraction.get_section_text_cit_keys()

        # Transform data
        transformed_prefetcher = tf.preprocessing_to_prefetcher(paper_global_citations)
        transformed_citeworth = tf.preprocessing_to_citeworthiness_detection(paper_sentences_with_citeworth)

        # Apply Prefetcher
        predicted_candidates = dict([(sec, prefetcher.predict(cits, sec, K)) for cits, sec in transformed_prefetcher])

        # sentences = extraction.get_section_text_citeworth()

        # paragraphs = extraction.get_section_text_paragraph()

        # correct_citations = extraction.get_section_text_cit_keys()

        # Apply CiteWorth
        predicted_citeworthiness = dict()
        for section, paragraph_queries in transformed_citeworth.items():
            section_predictions = []
            for sentence_queries, ground_truth in zip(paragraph_queries, paper_sentences_with_citeworth[section]):
                if len(sentence_queries) > 0:
                    # All ids in the current batch of sentence queries
                    ids, _ = zip(*sentence_queries)

                    # The values wich were extracted by structure extraction
                    _, extracted_labels = zip(*ground_truth)
                    extracted_labels = list(extracted_labels)

                    # Keep track on the amount of citeworthy and non citeworthy sentences
                    citeworthy_count += extracted_labels.count(True)
                    non_citeworthy_count += extracted_labels.count(False)

                    # Apply Citeworth
                    sentence_predictions = citworth.predict(sentence_queries, section)

                    # Check the predictions with our extracted truths
                    pred_ids = [id for id, _ in sentence_predictions]

                    pred_array = get_citeworth_array(ids, pred_ids)
                    eval_array = zip(extracted_labels, pred_array)
                    eval_array = list(eval_array)

                    # Keep track of the labeled sentence results
                    correct_citeworthy_count += eval_array.count((True, True))
                    correct_non_citeworthy_count += eval_array.count((False, False))
                    incorrect_citeworthy_count += eval_array.count((False, True))
                    incorrect_non_citeworthy_count += eval_array.count((True, False))

                    section_predictions.append(pred_array)
                else:
                    section_predictions.append([])
            predicted_citeworthiness[section] = section_predictions

        print(
            f"Found {correct_citeworthy_count} of {citeworthy_count} citations for paper {valid_ids[idx]} : {paper_title}")

        do_reranker = True
        if do_reranker:
            for section in paper_sentences_with_correct_citations.keys():
                for tup, gt_tuple, paragraph, citeworth in zip(
                        paper_sentences_with_correct_citations[section],
                        paper_sentences_with_citeworth[section],
                        paper_paragraphs[section],
                        predicted_citeworthiness[section]):

                    # Tuple in for loop syntax seems to not work on the server
                    sentences, correct_papers = zip(*tup)
                    _, gt_citeworthyness = zip(*gt_tuple)

                    # Keep only sentences labeled as citeworthy
                    only_citeworthy_sentences = [(s, t, cc) for s, t, c, cc in
                                                 zip(sentences, gt_citeworthyness, citeworth, correct_papers) if c]
                    if len(only_citeworthy_sentences) == 0:
                        # didn't detect citeworthy sentences
                        continue
                    c_sentences, c_gt, c_papers = zip(*only_citeworthy_sentences)

                    # Transform the data to reranker format
                    transformed_reranker_sentences = transformations.citeworthiness_detection_to_reranker(
                        c_sentences,
                        paragraph, section,
                        paper_abstract,
                        paper_title)

                    transformed_candidates = transformations.prefetcher_to_reranker(predicted_candidates[section],
                                                                                    papers_info)

                    for citation_context, c_citeworth, extracted_papers in zip(transformed_reranker_sentences, c_gt,
                                                                               c_papers):

                        if not c_citeworth:
                            # For our evaluation we can't assign correct values to citation context which aren't labeled as citeworthy
                            # to save time during our evaluation we will skip those contexts all together
                            continue

                        pred_reranker = reranker.predict(citation_context, transformed_candidates)

                        # Print out for user feedback
                        print(f"Context: {citation_context['citation_context']}")
                        for i, entry in enumerate(pred_reranker[:5]):
                            print(f"{i + 1}: {entry['id']} - {entry['title']}")

                        if c_citeworth:
                            # we only can check if citation is correct if there was a citation to begin with
                            def get_correct(k: int) -> List[str]:
                                # get correctly found papers
                                correct_found_papers = [pred for pred in pred_reranker[:k] if
                                                    pred['id'] in extracted_papers]
                                return correct_found_papers

                            r5 = get_correct(5)
                            r10 = get_correct(10)

                            if len(r5) > 0:
                                correct_citation_count_r5 += 1
                            else:
                                incorrect_citation_count_r5 += 1

                            if len(r10) > 0:
                                correct_citation_count_r10 += 1
                            else:
                                incorrect_citation_count_r10 += 1

            print(f"Found {correct_citation_count_r5} of {correct_citeworthy_count} citations for paper {valid_ids[idx]}")
            result_obj = {'paper_id': valid_ids[idx],
                          'citeworthy_count': citeworthy_count,
                          'non_citeworthy_count': non_citeworthy_count,
                          'correct_citation_count_r5': correct_citation_count_r5,
                          'correct_citation_count_r10': correct_citation_count_r10,
                          'correct_citeworthy_count': correct_citeworthy_count,
                          'correct_non_citeworthy_count': correct_non_citeworthy_count,
                          'incorrect_citeworthy_count': incorrect_citeworthy_count,
                          'incorrect_non_citeworthy_count': incorrect_non_citeworthy_count,
                          'incorrect_citation_count_r5': incorrect_citation_count_r5,
                          'incorrect_citation_count_r10': incorrect_citation_count_r10};

            with open('results_experiments.jsonl', 'a+') as f:
                f.write(f'{json.dumps(result_obj)}\n')

        # Copy counters to global variables
        global_citeworthy_count += citeworthy_count
        global_non_citeworthy_count += non_citeworthy_count
        global_correct_citation_count_r5 += correct_citation_count_r5
        global_correct_citation_count_r10 += correct_citation_count_r10
        global_correct_citeworthy_count += correct_citeworthy_count
        global_correct_non_citeworthy_count += correct_non_citeworthy_count
        global_incorrect_citeworthy_count += incorrect_citeworthy_count
        global_incorrect_non_citeworthy_count += incorrect_non_citeworthy_count
        global_incorrect_citation_count_r5 += incorrect_citation_count_r5
        global_incorrect_citation_count_r10 += incorrect_citation_count_r10
        idx += 1

    print(f"Found {global_correct_citation_count_r5} of {global_correct_citeworthy_count} citations for all papers")
    # json.dump(loaded_ids, open('loaded_ids.json','w'))
    global_result_obj = {'paper_id': 'all',
                         'citeworthy_count': global_citeworthy_count,
                         'non_citeworthy_count': global_non_citeworthy_count,
                         'correct_citation_count_r5': global_correct_citation_count_r5,
                         'correct_citation_count_r10': global_correct_citation_count_r10,
                         'correct_citeworthy_count': global_correct_citeworthy_count,
                         'correct_non_citeworthy_count': global_correct_non_citeworthy_count,
                         'incorrect_citeworthy_count': global_incorrect_citeworthy_count,
                         'incorrect_non_citeworthy_count': global_incorrect_non_citeworthy_count,
                         'incorrect_citation_count_r5': global_incorrect_citation_count_r5,
                         'incorrect_citation_count_r10': global_incorrect_citation_count_r10};
