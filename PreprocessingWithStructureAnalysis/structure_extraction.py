import tempfile
from pathlib import Path
from TexSoup import TexSoup, TexNode
from TexSoup.data import TexText

from nltk.stem import WordNetLemmatizer
from .tree import *

from .id_translator import IdTranslator

import nltk
import bibtexparser

import json
import os

import re

DATA_DIR = 'D:/expanded'

SYNONYM_DICT = './synonym_dict.json'
UNWANTED_ENVS = './unwanted_envs.txt'
UNWANTED_CMDS = './unwanted_cmds.txt'

TEMPLATES = './templates.json'

nltk.download('omw-1.4')


class StructureExtraction:

    def __init__(self, data_dir):
        self.dir = os.path.dirname(os.path.realpath(__file__))

        self.data_dir = Path(data_dir)
        self.valid_ids = []
        self.soup = None
        self.bib = None
        self.active = None
        self.lemmatizer = WordNetLemmatizer()
        # self.tmp_dir = Path(tempfile.mkdtemp())
        self.tmp_dir = Path('C:\\Users\\Simon\\Desktop\\test2')
        self.translator = IdTranslator()
        with open(os.path.join(self.dir,SYNONYM_DICT)) as f:
            self.s_dict = json.load(f)
        with open(os.path.join(self.dir,TEMPLATES)) as f:
            self.templates = json.load(f)
        self.tree = Tree(['*'], None)
        for t in self.templates:
            update_tree(self.tree, t)
        self.atree = t2anytree(self.tree)
        for pre, _, node in RenderTree(self.atree):
            print("%s%s" % (pre, node.name))

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
            #self._analyze_structure()
            self.active = str(id)
            self._filter_unwanted_stuff()
            tmp_file = self.tmp_dir / str(id + ".tex")
            with open(tmp_file, 'w') as f:
                f.write(str(self.soup))
            self.bib = bibtexparser.load(open(bib_file))
            return True
        except:
            print(f"Could not load document {tex_file}")
            return False

    def _analyze_structure(self):
        table_cnt = self.soup.count(r'\begin{table}')
        tabular_cnt = self.soup.count(r'\begin{tabular}')
        figure_cnt = self.soup.count(r'\begin{figure}')
        print(f'Table Count: {table_cnt}')
        print(f'Tabular Count: {tabular_cnt}')
        print(f'Figure Count: {figure_cnt}')

    def _filter_unwanted_stuff(self):
        unwanted_env = list()
        with open(os.path.join(self.dir,UNWANTED_ENVS),'r') as f:
            unwanted_env =  [l.strip() for l in f]
        # TODO is center only used for tables?

        unwanted_cmds = list()
        with open(os.path.join(self.dir,UNWANTED_CMDS), 'r') as f:
            unwanted_cmds = [l.strip() for l in f]

        keep_content = ['emph', 'textit', 'texttt', 'textbf']

        #small_cmds = ['_','#','%']

        # TODO itemize must be kept
        # TODO what about footnotes

        # TODO parapgraph structure gets lost

        try:
            #ens = self.soup.find_all(['enumerate','itemize'])
            # concatenate all children together
            #for e in ens:
            #    conc = ' '.join([str(c.expr) for c in e.children])
            #    e.replace_with(TexNode(TexText(conc)))

            # now reload tex Soup to ensure that removed itemize, enumerate gets read correctly
            #tmp_file = self.tmp_dir / str(self.active + ".tex")
            #with open(tmp_file, 'w') as f:
            #    f.write(str(self.soup))
            #self.soup = TexSoup(open(tmp_file), tolerance=1)


            for item in self.soup.document:
                if type(item) == TexNode and item.name in unwanted_env:
                    item.delete()
                if type(item) == TexNode and item.name in unwanted_cmds:
                    item.delete()
                if type(item) == TexNode and item.name in keep_content:
                    item.replace_with(TexNode(TexText(item.string)))
                #if type(item) == TexNode and item.name in small_cmds:
                #    item.replace_with(TexNode(TexText(' ')))
            return True
        except Exception as e:
            #
            return False

    def _preprocess_section_title(self, title: str) -> str:
        cleaned = re.sub('[^A-Za-z0-9 ]+', '', title.lower())
        stemmed = " ".join([self.lemmatizer.lemmatize(w) for w in nltk.word_tokenize(cleaned)])
        if stemmed in self.s_dict.keys():
            stemmed = self.s_dict[stemmed]

        return stemmed

    def _match_template_rules(self, sections):
        # Declare root of tree
        root = self.atree.root

        sect_tuple = tuple(sections)
        ##
        # iterate through tree

        most_specific_rule_len = 0

        for item in PreOrderIter(root):
            res = eval(item.name)
            if sub(sect_tuple, res):
                if most_specific_rule_len < len(res):
                    most_specific_rule_len = len(res)
                print(item)

        if most_specific_rule_len > 3:
            # now change all to methods
            return True
        return False

    def get_section_titles(self):
        if self.soup is None:
            raise ValueError("No file loaded")
        section_list = []
        for s_candidate in self.soup.find_all(['section', 'appendix']):
            if s_candidate.name == 'appendix':
                break
            if s_candidate.string.lower() == 'conclusion':
                section_list.append(s_candidate.string.lower())
                break
            section_list.append(s_candidate.string.lower())

        result = [self._preprocess_section_title(s) for s in section_list]

        # remove duplicates for rule matching
        reduced = list(dict.fromkeys(result))

        matched = self._match_template_rules(reduced)

        result = []
        self.section_mapping = dict()
        for i in range(0, len(section_list)):
            if matched and section_list[i] not in self.s_dict.keys():
                self.section_mapping[section_list[i]] = 'method'
                result.append('method')
            else:
                self.section_mapping[section_list[i]] = section_list[i]
                result.append(section_list[i])

        return result

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
                    if isinstance(cite_obj, list):
                        cite_dict[cur_sec].extend(cite_obj)
                    else:
                        cite_dict[cur_sec].append(cite_obj)
                except Exception as e:
                    # some error in transfor_citation occured
                    print(f"Error on transform_citation {elm} in document {self.active}")

        return cite_dict

    def get_citations_by_sections(self):
        if self.soup is None:
            raise ValueError("No file loaded")

        def get_bib_entry(elm):
            keys = str(elm.args[-1])[1:-1].split(',')
            bib_entries = [self.bib.entries_dict[str(x.strip())] for x in keys]
            return self.translator.query(bib_entries)

        return self._get_citations_by_sections(get_bib_entry)

    def get_citations_with_pos_by_section(self):
        if self.soup is None:
            raise ValueError("No file loaded")

        def get_citation_pos(elm):
            keys = str(elm.args[-1])[1:-1].split(',')
            return elm.position, len(str(elm.expr)), keys  # use raw position for file seek

        return self._get_citations_by_sections(get_citation_pos)

    def _process_fulltext(self, section_title: str, section_start: int, text: str, transform_cits, mask_citations=True):
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
                    cits_in_sentence = [c for c in citations_pos if
                                        c[0] - section_start > offset and c[0] - section_start < offset + len(sentence)]
                    result_sentences.append((re.sub('[X]{4,}', '', sentence), transform_cits(cits_in_sentence)))
                    offset += len(sentence)

                result_paragraphs.append(result_sentences)
            if offset != len(text):
                print(f"Missmatch between offset: {offset} and text len {len(text)}")

            return result_paragraphs
        else:
            # Only return fulltext
            text = re.sub('[X]{4,}', '', text)
            return text

    def _get_section_lines(self, transform_cits):
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

            # loaded the processed tex file
            p_tex_file = self.tmp_dir / str(self.active + ".tex")
            with open(p_tex_file, 'r') as f:
                f.seek(start_pos, 0)
                section_text = f.read(end_pos - start_pos)

                # section text now seems off by a few characters
                strange_offset = section_text.find('\section')

                # correct readin buffer
                section_text = section_text[strange_offset:] + f.read(strange_offset)

                # mask section tag so that it gets later removed
                section_text = str('X' * tag_len) + section_text[tag_len:]

                result = self._process_fulltext(curr_section, start_pos, section_text, transform_cits, True)

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
            cit_keys = [k for c in cits for k in c[2]]
            bib_entries = [self.bib.entries_dict[k] if k in self.bib.entries_dict.keys() else None for k in cit_keys]
            return self.translator.query(bib_entries)

        return self._get_section_lines(to_keys)


if __name__ == "__main__":
    # Test the capabilities of structure extraction

    #translator = IdTranslator()

    extraction = StructureExtraction(DATA_DIR)

    valid_ids = extraction.get_valid_ids()
    # json.dump(valid_ids, open('valid_ids','w'))

    print(f"Found {len(valid_ids)} tex files with corresponding bibtex entry")

    # Find a document which can be loaded
    idx = 0
    while True:
        if not extraction.set_as_active(valid_ids[idx]):
            idx += 1
            continue

        section_list = extraction.get_section_titles()

        # citation_pos = extraction.get_citations_with_pos_by_section()

        # citations = extraction.get_citations_by_sections()

        #translated = dict()
        #
        #for key, value in citations.items():
        #    translated[key] = translator.query(value)


        sentences = extraction.get_section_text_citeworth()
        #print(sentences)

        sentences2 = extraction.get_section_text_cit_keys()
        #print(sentences2)

        # Map<List<Tuple<str,List<str,Bool>>>> CiteWorth

        # Map<List<Tuple<str,List<str,List<int>>>>> Reranker

        # Map<List<int>> Prefetcher
        # test citation extraction
        # citations = extraction.get_citations_by_sections()
        # for section, cit_list in citations.items():
        #   print(f"{section}: {cit_list}")

        # test section title extraction
        # section_list = extraction.get_section_titles(True)
        # print(section_list)
        idx += 1
