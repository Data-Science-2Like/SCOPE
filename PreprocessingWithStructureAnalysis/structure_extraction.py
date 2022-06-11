from pathlib import Path
from TexSoup import TexSoup

from nltk.stem import WordNetLemmatizer
import nltk

import re

DATA_DIR = 'D:/expanded'

nltk.download('omw-1.4')

class StructureExtraction:

    def __init__(self,data_dir):
        self.data_dir =  Path(data_dir)
        self.valid_ids = []
        self.soup = None
        self.lemmatizer = WordNetLemmatizer()
        if not self.data_dir:
            raise ValueError('Data dir not found')

    """ Gets all valid ids in directory which exist as .tex and .bib file
    """
    def get_valid_ids(self):
        if len(self.valid_ids) > 0:
            return self.valid_ids
        ids = []
        pathlist = self.data_dir.glob('*.tex')
        for path in pathlist:
            # check if bib file exists
            if path.with_suffix('.bib').exists():
                ids.append(path.stem)
        self.valid_ids = ids
        return ids

    def set_as_active(self, id : str) -> bool:
        tex_file = self.data_dir / str(id + ".tex")
        try:
            self.soup = TexSoup(open(tex_file), tolerance=1)
            return True
        except:
            print(f"Could not load document {tex_file}")
            return False

    def get_section_titles(self,preprocessing = False,use_sdict = False, apply_rules = False):
        if self.soup is None:
            raise ValueError("No file loaded")
        section_list = [s.string.lower() for s in self.soup.find_all('section')]
        if preprocessing:
            cleaned = [re.sub('[^A-Za-z0-9 ]+', '', s) for s in section_list]
            stemmed = [" ".join([self.lemmatizer.lemmatize(w) for w in nltk.word_tokenize(s)]) for s in cleaned]
            return stemmed

        return section_list




if __name__ == "__main__":
    # Test the capabilities of structure extraction

    extraction = StructureExtraction(DATA_DIR)

    valid_ids = extraction.get_valid_ids()
    print(f"Found {len(valid_ids)} tex files with corresponding bibtex entry")


    # Find a document which can be loaded
    idx = 0
    while True:
        if not extraction.set_as_active(valid_ids[idx]):
            idx += 1
            continue
        section_list = extraction.get_section_titles(True)
        print(section_list)
        idx += 1


