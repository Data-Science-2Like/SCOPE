from pathlib import Path
from TexSoup import TexSoup

from nltk.stem import WordNetLemmatizer
import nltk
import bibtexparser

import re

DATA_DIR = 'D:/expanded'

nltk.download('omw-1.4')


class StructureExtraction:

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.valid_ids = []
        self.soup = None
        self.bib = None
        self.active = None
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

    def set_as_active(self, id: str) -> bool:
        tex_file = self.data_dir / str(id + ".tex")
        bib_file = self.data_dir / str(id + ".bib")
        try:
            self.soup = TexSoup(open(tex_file), tolerance=1)
            self.bib = bibtexparser.load(open(bib_file))
            self.active = str(id)
            return True
        except:
            print(f"Could not load document {tex_file}")
            return False

    def _preprocess_section_title(self, title: str) -> str:
        cleaned = re.sub('[^A-Za-z0-9 ]+', '', title.lower())
        stemmed = " ".join([self.lemmatizer.lemmatize(w) for w in nltk.word_tokenize(cleaned)])
        return stemmed

    def get_section_titles(self, preprocessing=False, use_sdict=False, apply_rules=False):
        if self.soup is None:
            raise ValueError("No file loaded")
        section_list = [s.string.lower() for s in self.soup.find_all('section')]
        if preprocessing:
            return [self._preprocess_section_title(s) for s in section_list]
        return section_list

    def _get_citations_by_sections(self, transform_citation):
        if self.soup is None:
            raise ValueError("No file loaded")
        section_citation_list = self.soup.find_all(['section', 'cite', 'citet', 'citep'])
        cur_sec = ''
        cite_dict = dict()
        for elm in section_citation_list:
            if elm.name == 'section':
                # begin new section
                cur_sec = self._preprocess_section_title(elm.string.lower())
                cite_dict[cur_sec] = []
            else:
                try:
                    # is a citation
                    # TODO issue when multiple citations in citation marker
                    cite_obj = transform_citation(elm)
                    cite_dict[cur_sec].append(cite_obj)
                except:
                    # some error in transfor_citation occured
                    print(f"Error on transform_citation {elm} in document {self.active}")

        return cite_dict

    def get_citations_by_sections(self):
        if self.soup is None:
            raise ValueError("No file loaded")

        def get_bib_entry(elm):
            return self.bib.entries_dict[str(elm.string)]

        return self._get_citations_by_sections(get_bib_entry)

    def get_citations_with_pos_by_section(self):
        if self.soup is None:
            raise ValueError("No file loaded")

        def get_citation_pos(elm):
            # return self.soup.char_pos_to_line(elm.position)
            return elm.position, len(str(elm.expr)), str(elm.string)  # use raw position for file seek

        return self._get_citations_by_sections(get_citation_pos)

    def _process_fulltext(self,section_title: str,section_start: int, text: str, transform_cits, mask_citations=True):
        citations_pos = self.get_citations_with_pos_by_section()[section_title]
        if mask_citations:
            for cit_pos, cit_len, cit_key in citations_pos:
                relative_pos = cit_pos - section_start

                # don't change length of text here, so that citation offsets are still valid
                text = text[:relative_pos] + str('X' * cit_len) + text[relative_pos + cit_len:]

        split_paragraphs = True
        if split_paragraphs:
            offset = 0
            result_paragraphs = []
            paragraph_candidates = text.split('\n\n')

            for paragraph in paragraph_candidates:
                if len(paragraph) == 0:
                    offset += 2
                    continue

                result_sentences = []
                sentences = nltk.sent_tokenize(paragraph)
                for sentence in sentences:
                    cits_in_sentence = [c for c in citations_pos if c[0]-section_start > offset and c[0]-section_start < offset +len(sentence)]
                    result_sentences.append((re.sub('[X]{4,}','',sentence),transform_cits(cits_in_sentence)))
                    offset += len(sentence)

                result_paragraphs.append(result_sentences)
            if offset != len(text):
                print(f"Missmatch between offset: {offset} and text len {len(text)}")

            return result_paragraphs
        else:
            # Only return fulltext
            text = re.sub('[X]{4,}','',text)
            return text

    def _get_section_lines(self,transform_cits):
        if self.soup is None:
            raise ValueError("No file loaded")

        sObjs = self.soup.find_all('section')
        section_list = [(self._preprocess_section_title(str(s.string)), s.position, len(str(s.expr))) for s in sObjs]

        section_dict = dict()

        curr_section = ''
        start_pos = None
        end_pos = None

        for sec_title, position, tag_len in section_list:
            if start_pos is None:
                curr_section = sec_title
                start_pos = position
                continue
            end_pos = position
            tex_file = self.data_dir / str(self.active + ".tex")
            with open(tex_file, 'r') as f:
                f.seek(start_pos, 0)
                section_text = f.read(end_pos - start_pos)

                # section text now seems off by a few characters
                strange_offset = section_text.find('\section')

                # correct readin buffer
                section_text = section_text[strange_offset:] + f.read(strange_offset)

                # mask section tag so that it gets later removed
                section_text =str('X' * tag_len) + section_text[tag_len:]

                result = self._process_fulltext(curr_section, start_pos, section_text,transform_cits,True)

                section_dict[curr_section] = result

            start_pos = end_pos
            curr_section = sec_title

            # TODO process last section

        return section_dict

    def get_section_text_citeworth(self):

        def to_citeworth(cits):
            return len(cits) > 0

        return self._get_section_lines(to_citeworth)

    def get_section_text_cit_keys(self):

        def to_keys(cits):
            return [c[2] for c in cits]

        return self._get_section_lines(to_keys)


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

        # citation_pos = extraction.get_citations_with_pos_by_section()

        sentences = extraction.get_section_text_citeworth()
        print(sentences)

        sentences2 = extraction.get_section_text_cit_keys()
        print(sentences2)
        # test citation extraction
        # citations = extraction.get_citations_by_sections()
        # for section, cit_list in citations.items():
        #   print(f"{section}: {cit_list}")

        # test section title extraction
        # section_list = extraction.get_section_titles(True)
        # print(section_list)
        idx += 1
