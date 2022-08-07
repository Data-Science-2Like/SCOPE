from Prefetcher.baselines import BM25Baseline
from Prefetcher.citeworth import load_dataset

import json

if __name__ == "__main__":
    #prefetcher = BM25Baseline()

    #bags, x_train = load_dataset(2019, 2018, 2)

    loaded_ids = json.load(open('../loaded_ids.json'))
    print(f"Already loaded {len(loaded_ids)} papers sucessfully")
    #print(bags)