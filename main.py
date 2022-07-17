
from PreprocessingWithStructureAnalysis.id_translator import IdTranslator
from PreprocessingWithStructureAnalysis.structure_extraction import StructureExtraction, DATA_DIR

from Reranker.Reranker_Transformer import Transformer_Reranker

#from Prefetcher.aae_recommender import AAERecommender

from CiteworthinessDetection.CiteWorth import  CiteWorth

from Prefetcher.baselines import BM25Baseline

import transformations as tf



if __name__ == "__main__":

    extraction = StructureExtraction(DATA_DIR)

    valid_ids = extraction.get_valid_ids()
    # json.dump(valid_ids, open('valid_ids','w'))

    print(f"Found {len(valid_ids)} tex files with corresponding bibtex entry")

    # prefetcher = AAERecommender('./Prefetcher/trained/aae.torch',True)
    prefetcher = BM25Baseline()

    reranker = Transformer_Reranker('./Reranker/trained/reranker-acl200-5epochs')

    #citworth = CiteWorth('./CiteworthinessDetection/trained/citeworth-ctx-section-always-seed1000.pth','always')
    citworth = CiteWorth('./CiteworthinessDetection/trained/citeworth-ctx-section-always-seed1000.pth','always')

    print("Loaded models successfully")
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
        candidates = []
        for cits, section in transformed_prefetcher:
            candidates.append(prefetcher.predict(cits,section,200))

        # Extract Information for CiteWorth

        sentences = extraction.get_section_text_citeworth()

        transformed_citeworth = tf.preprocessing_to_citeworthiness_detection(sentences)
        # print(sentences)

        sentences2 = extraction.get_section_text_cit_keys()

        idx += 1
