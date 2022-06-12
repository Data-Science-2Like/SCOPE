from typing import List

import numpy as np

from Prefetcher.prefetcher import Prefetcher

class AAERecommender(Prefetcher):
    def __init__(self, model_path: str, use_section_info):
        super().__init__()
        self.use_section_info = use_section_info

        
