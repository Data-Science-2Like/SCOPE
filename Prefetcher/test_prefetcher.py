import os

from aae_recommender import AAERecommender

from PreprocessingWithStructureAnalysis.structure_extraction import StructureExtraction, DATA_DIR



if __name__ == "__main__":
    # Test the capabilities of the prefetcher module

    #local_path = os.path.join('./../PreprocessingWithStructureAnalysis', DATA_DIR)

    extraction =StructureExtraction(DATA_DIR)

    prefetcher = AAERecommender('trained/aae.torch', False)

    valid_ids = extraction.get_valid_ids()

    print(f"Found {len(valid_ids)} tex files with corresponding bibtex entry")

    idx = 0
    while True:
        if not extraction.set_as_active(valid_ids[idx]):
            idx += 1
            continue

        # initialize section title mappings
        section_list = extraction.get_section_titles()

        already_cited = extraction.get_citations_by_sections()

        for section, cited in already_cited.items():
            candidate_pool = prefetcher.predict(cited,section,10)
            print(candidate_pool)