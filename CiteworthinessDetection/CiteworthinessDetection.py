"""
©2022 Carolin Schindler
The implementation in this file is licensed under the CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International) License.
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

from typing import List, Tuple


class CiteworthinessDetection:
    def __init__(self):
        pass

    def predict(self, sentences: List[Tuple[str, str]], section: str) -> List[Tuple[str, str]]:
        """
        :param sentences: All sentences are processed together, i.e., they form a single input to the model.
                          Still, each sentence gets its own prediction regarding its cite-worthiness.
                          Each sentence is represented by a tuple of citation_context id and the string of the sentence.
        :param section: The name of the section these sentences belong to.
        :return: list of sentences that are cite-worthy (sentence is tuple of citation_context id and the string of the sentence)
        """
        raise NotImplementedError
