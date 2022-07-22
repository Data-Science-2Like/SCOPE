import json

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
    res = [ True for id in ids if id in predict_ids]
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

        transformed_citeworth = tf.preprocessing_to_citeworthiness_detection(sentences)

        for paragraph, section in transformed_citeworth:
            for sent in paragraph:
                ids, _ = zip(*sent)
                predictions = citworth.predict(sent,section)
                pred_ids, _ = zip(*predictions)

                print(get_citeworth_array(ids,pred_ids))



        idx += 1
