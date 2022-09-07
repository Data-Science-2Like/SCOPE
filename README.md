# SCOPE: Section Citation Online PipelinE
![Component-Structure of SCOPE](SCOPE_ComponentStructure.png "Component-Structure of SCOPE")

Our modular end-to-end global context-aware citation recommendation pipeline SCOPE is depicted above and entails four sub-tasks:
- Preprocessing with our Structure Analysis (see directory `PreprocessingWithStructureAnalysis`)
- Identification of Citation Contexts aka. cite-worthiness detection (see directory `CiteworthinesDetection`)
- Creating a Pool of Candidate Papers for which we are utilizing a global citation recommender (see directory `Prefetcher`)
- Performing the Citation Recommendation for each identified citation context (see directory `Reranker`)

## Getting started

To get all needed dependencies, create a new environment via:

```
conda env create -n scope --file environment.yml
```

Make sure that our [Modified S2ORC Dataset](https://github.com/Data-Science-2Like/dataset-creation) is accessible to the pipeline.
Run `run.sh` to perform the citation extraction and recommendation on all provided papers.

To change the files which are going to be loaded change the content of `correct_ids.json`. Alternatively you can use `extraction.get_valid_ids()` from the Structure Extraction module.
Notice that this selection can still contain files that can not be loaded.


## Known issues
- [ ] Some paper differ in the way that citation keys are written in document and the corresponding bib file
- ~~Title extraction fails if title is in format: \title[Something]{Actual Title}~~