import joblib
import glob
import json

DATA_DIR = 'C:/Users/Simon/Desktop/prefetcher_outputs'

RERANKER_FILE = 'C:/Users/Simon/Desktop/reranker_test_citing_ids.joblib'
reranker_citing_ids = joblib.load(RERANKER_FILE)

def load_papers_info(file):
    papers_info = dict()

    with open(file) as f:
        for line in f:
            entry = json.loads(line.strip())

            keep_keys = ['paper_title', 'paper_abstract', 'paper_year']

            papers_info[entry['paper_id']] = dict([(k, v) for k, v in entry.items() if k in keep_keys])

    return papers_info

paper_info = load_papers_info('../s2orc/papers.jsonl')


def do_joblib_check(data: dict):
    citing_papers = set()

    for k in data.keys():
        citing_papers.add(k)

        assert None not in data[k]

    too_much_ids = citing_papers - reranker_citing_ids

    missing_ids = reranker_citing_ids - citing_papers


    print(f"Citing papers {len(citing_papers)}")
    print(f"Missing papers {len(missing_ids)}")

    print(f"To Much papers {len(too_much_ids)}")

    print(data['83458577'].keys())

    find_paper_ids(missing_ids)


def find_paper_ids(searching_ids: set):
    version = 7
    found_ids = set()
    with open(f"../s2orc/aae_recommender_with_section_info_v{version}.jsonl") as f:
        for idx, line in enumerate(f):
            entry = json.loads(line.strip())

            if (entry['paper_id'] in searching_ids):
                found_ids.add(entry['paper_id'])
                print(f"Found missing paper_id at index {idx}")
                print(entry)

    now_missing = searching_ids - found_ids
    print(f"Now missing papers {len(now_missing)}")



if __name__ == '__main__':
    job_files= glob.glob(f'{DATA_DIR}/*.joblib')
    print(f"Found {len(job_files)} joblib files")

    data = joblib.load(job_files[0])
    do_joblib_check(data)