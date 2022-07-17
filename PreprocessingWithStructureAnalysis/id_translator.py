
from rank_bm25 import BM25Okapi

import json
import re
import numpy as np
import os
PAPER_TITLES = 'pipeline_title_id_v5.json'


class IdTranslator:

    def __init__(self):
        self.directory = os.path.dirname(os.path.realpath(__file__))
        self.papers_path = os.path.join(self.directory,PAPER_TITLES)

        with open(self.papers_path,'r') as f:
            self.title_dict = json.load(f)
            self.title_list = list(self.title_dict.keys())
            self.tokenized_corpus = [e.split(' ') for e in self.title_list]
            self.bm25 = BM25Okapi(self.tokenized_corpus)


    def query(self,bib_entries):

        bm25_cnt = 0

        results = list()

        for bib in bib_entries:
            if bib is not None:
                title = bib['title']
                cleaned  = re.sub('[^A-Za-z0-9 ]+', '', title.lower())
                try:
                    results.append(self.title_dict[cleaned])
                    print(f'Real match on {cleaned}')                
                except KeyError as e:
                    bm25_cnt += 1
                    tokenized = cleaned.split(' ')
                    scores = np.array(self.bm25.get_scores(tokenized))

                    matched = self.title_list[np.argmax(scores)]
                    results.append(self.title_dict[matched])
            else:
                results.append(None)
        if len(results) > 0:
            print(f'Warning using bm25 fallback on {bm25_cnt} of {len(results)} queries')
        return results