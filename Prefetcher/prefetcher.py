from typing import List, Tuple

class Prefetcher:
    def __init__(self):
        pass

    def predict(self, already_cited: List[str], section: str, k: int) -> List[str]:
        """
        :param already_cited: A list of papers which is alredy cited in the current context
        :param section: The name of the section for which we want a prediction
        :param k: Specifies the size of the candidate pool
        :return: list of candidate paper. each paper is identified by a unique id
        """
        raise NotImplementedError
    