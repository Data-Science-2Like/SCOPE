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
    while True:
        if not extraction.set_as_active(valid_ids[idx]):
            idx += 1
            continue

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
            for sent in paragraph:
                par_result = []
                if len(sent) > 0:
                    ids, _ = zip(*sent)
                    predictions = citworth.predict(sent,section)
                    pred_ids = []
                    if len(predictions) > 0:
                        pred_ids, _ = zip(*predictions)

                    # print(get_citeworth_array(ids,pred_ids))
                    par_result.append(get_citeworth_array(ids,pred_ids))
                else:
                    par_result.append([])
                sec_result.append(par_result)
            pred_citeworthyness[section] = sec_result


        citation_count = 0
        correct_count = 0

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
                            citation_count += 1

                        for pred in pred_reranker:
                            if pred['id'] in corr_papers:
                                correct_count += 1
                                break

        print(f"Found {correct_count} of {citation_count} citations for paper {valid_ids[idx]}")


        idx += 1
