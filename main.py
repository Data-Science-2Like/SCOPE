import json

import transformations
from PreprocessingWithStructureAnalysis.id_translator import IdTranslator
from PreprocessingWithStructureAnalysis.structure_extraction import StructureExtraction, DATA_DIR

from Reranker.Reranker_Transformer import Transformer_Reranker

#from Prefetcher.aae_recommender import AAERecommender

from CiteworthinessDetection.CiteWorth import  CiteWorth

from Prefetcher.baselines import BM25Baseline

import transformations as tf

def load_papers_info(file):
    papers_info = dict()

    with open(file) as f:
        for line in f:
            entry = json.loads(line.strip())

            keep_keys = ['paper_title', 'paper_abstract', 'paper_year']

            papers_info[entry['paper_id']] = dict([(k, v) for k, v in entry.items() if k in keep_keys])

    return papers_info

def get_citeworth_array(ids,predict_ids):
    # citeworthy_sents = [s for s, label in zip(sentences, preds) if label == 1]
    res = [True if id in predict_ids else False for id in ids]
    return res

if __name__ == "__main__":

    extraction = StructureExtraction(DATA_DIR)

    valid_ids = extraction.get_valid_ids()
    # json.dump(valid_ids, open('valid_ids','w'))
    papers_info = load_papers_info('./s2orc/papers.jsonl')

    print(f"Found {len(valid_ids)} tex files with corresponding bibtex entry")

    # prefetcher = AAERecommender('./Prefetcher/trained/aae.torch',True)
    prefetcher = BM25Baseline()

    reranker = Transformer_Reranker('./Reranker/trained/reranker-acl200-5epochs')

    #citworth = CiteWorth('./CiteworthinessDetection/trained/citeworth-ctx-section-always-seed1000.pth','always')
    citworth = CiteWorth('./CiteworthinessDetection/trained/citeworth-ctx-section-always-seed1000.pth','always')

    print("Loaded models successfully")
    input("Press to continue")
    idx = 0

    # Sentences which are citeworthy
    global_citeworthy_count = 0
    # Sentences which are non citeworthy
    global_non_citeworthy_count = 0
    # Sentences which need a citation, get found by citeworth and reranker
    global_correct_citation_count = 0
    # Sentences which need a citation and get found by the citeworthy module
    global_correct_citeworthy_count = 0
    # Sentences which don't need a citation and get found by the citeworthy module
    global_correct_non_citeworthy_count =  0
    # Sentences which don't need a citation but get labeled as such
    global_incorrect_citeworthy_count = 0
    # Sentences which need a citation but don't get labeled as such
    global_incorrect_non_citeworthy_count = 0
    # Sentences which need a citation, get labeled as citeworth, but reranker doesn't find the right thing
    global_incorrect_citation_count = 0

    while True:
        if not extraction.set_as_active(valid_ids[idx]):
            idx += 1
            continue

        # Sentences which are citeworthy
        citeworthy_count = 0
        # Sentences which are non citeworthy
        non_citeworthy_count = 0
        # Sentences which need a citation, get found by citeworth and reranker
        correct_citation_count = 0
        # Sentences which need a citation and get found by the citeworthy module
        correct_citeworthy_count = 0
        # Sentences which don't need a citation and get found by the citeworthy module
        correct_non_citeworthy_count = 0
        # Sentences which don't need a citation but get labeled as such
        incorrect_citeworthy_count = 0
        # Sentences which need a citation but don't get labeled as such
        incorrect_non_citeworthy_count = 0
        # Sentences which need a citation, get labeled as citeworth, but reranker doesn't find the right thing
        incorrect_citation_count = 0


        section_list = extraction.get_section_titles()


        title = extraction.get_title()

        abstract = extraction.get_abstract()
        # Extract Information for Prefetcher

        glob_cits = extraction.get_citations_by_sections()

        transformed_prefetcher = tf.preprocessing_to_prefetcher(glob_cits)

        # Apply Prefetcher
        candidates = dict()
        for cits, section in transformed_prefetcher:
            candidates[section] = prefetcher.predict(cits,section,200)

        # Extract Information for CiteWorth

        sentences = extraction.get_section_text_citeworth()

        paragraphs = extraction.get_section_text_paragraph()

        correct_citations = extraction.get_section_text_cit_keys()

        transformed_citeworth = tf.preprocessing_to_citeworthiness_detection(sentences)

        pred_citeworthyness = dict()
        for section, paragraph in transformed_citeworth.items():
            sec_result = []
            for sent, gt in zip(paragraph,sentences[section]):
                par_result = []
                if len(sent) > 0:
                    ids, _ = zip(*sent)

                    _, extracted_labels = zip(*gt)

                    citeworthy_count += list(extracted_labels).count(True)
                    non_citeworthy_count += list(extracted_labels).count(False)

                    predictions = citworth.predict(sent,section)
                    pred_ids = []
                    if len(predictions) > 0:
                        pred_ids, _ = zip(*predictions)

                    pred_array = get_citeworth_array(ids,pred_ids)

                    eval_array = zip(extracted_labels,pred_array)

                    eval_array = list(eval_array)

                    correct_citeworthy_count += eval_array.count((True,True))
                    correct_non_citeworthy_count += eval_array.count((False,False))
                    incorrect_citeworthy_count += eval_array.count((False,True))
                    incorrect_non_citeworthy_count += eval_array.count((True,False))

                    # print(get_citeworth_array(ids,pred_ids))
                    par_result.append(pred_array)
                else:
                    par_result.append([])
                sec_result.append(par_result)
            pred_citeworthyness[section] = sec_result

        print(f"Found {correct_citeworthy_count} of {citeworthy_count} citations for paper {valid_ids[idx]} : {title}")

        do_reranker = False
        if do_reranker:

            for section in sentences.keys():
                for sents, paragraph, cite, correct_papers in zip(sentences[section], paragraphs[section], pred_citeworthyness[section], correct_citations[section]):
                    only_cite = [(s,g,cc) for (s,g),c,cc in zip(sents,cite,correct_papers) if c]
                    if len(only_cite) > 0:
                        only_sent, gt, corr = zip(*only_cite)
                        transformed_reranker = transformations.citeworthiness_detection_to_reranker(only_sent,paragraph,section,abstract,title)

                        transformed_candidates = transformations.prefetcher_to_reranker(candidates[section],papers_info)

                        for cont, ground, corr_papers in zip(transformed_reranker,gt, corr):
                            pred_reranker = reranker.predict(cont,transformed_candidates)[:5]
                            for i, entry in enumerate(pred_reranker):
                                print(f"Context: {cont['citation_context']}")
                                print(f"{i+1}: {entry['id']} - {entry['title']}")

                            if ground:
                                correct_citeworthy_count += 1

                            correct_found_papers = [pred for pred in pred_reranker if pred['id'] in corr_papers]
                            if len(correct_found_papers) > 0:
                                correct_citation_count += 1
                            else:
                                incorrect_citation_count += 1

            print(f"Found {correct_citation_count} of {correct_citeworthy_count} citations for paper {valid_ids[idx]}")

        global_citeworthy_count += citeworthy_count
        global_non_citeworthy_count += non_citeworthy_count
        global_correct_citation_count += correct_citation_count
        global_correct_citeworthy_count += correct_citeworthy_count
        global_correct_non_citeworthy_count += correct_non_citeworthy_count
        global_incorrect_citeworthy_count += incorrect_citeworthy_count
        global_incorrect_non_citeworthy_count += incorrect_non_citeworthy_count
        global_incorrect_citation_count += incorrect_citation_count

        idx += 1

    print(f"Found {global_correct_citation_count} of {global_correct_citeworthy_count} citations for all papers")