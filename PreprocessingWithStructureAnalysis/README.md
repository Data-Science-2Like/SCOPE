# Preprocessing with Structure Analysis

Contains the structure extraction module of the SCOPE pipeline. It can extract differently structured citation context from a given `.tex` file.
The usage is described below. The structure extraction expects the code for each paper to be in a single `.tex` file.
The bibtex information have to be present in a `.bib` file for each paper.

## Usage

To use the structure extraction module follow the code below:

```
# Set the working directory of the structure extraction
extraction = StructureExtraction('../path/to/texfiles')

# Set a specific paper as active. 
# When calling this method the module checks if all needed files are existent for the given paper id.
# Returns `True` when the file could be loaded successfully.

success = extraction.set_as_active('200512152')


if success:
	# Extract Global Information
	
	# Get Section titles has to be called before extracting any citation context to successfully apply template matching to the section headings
    paper_section_list = extraction.get_section_titles()
    paper_title = extraction.get_title()
    paper_abstract = extraction.get_abstract()
	
```

## Structure

- `id_translator.py`: Contains a utility class to match a given paper title to the list of candidate papers. If no exact match could be found. an approximation is done via BM25.

- `structure_extraction.py`: The main module of the structure extraction.

- `synonym_dict.json`: Contains the synoym dictionary which gets applied before template matching

- `templates.json`: Contains the tree of template rules that get applied during template matching.

- `tree.py`: Utility class to generate the template tree.

- `unwanted_cmds.txt`: List of unwanted LaTeX commands that get filtered during `set_as_active`.

- `unwanted_envs.txt`: List of unwanted LaTeX enviroments that get filtered during `set_as_active`.

Each code file consists of indepth documentation and comments as well as type annotations