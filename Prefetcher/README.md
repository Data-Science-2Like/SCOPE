# Prefetcher

this module of the pipeline performs the global citation recommendation for each section in each paper to generate a list of candidate papers of length `k`.
As input it recieves a list of already cited papers for this section.

## Structure

- `prefetcher.py`: implementes the equally named abstract base class `Prefetcher. This class should be extended by any specific prefetcher module to match the supplied signature for the predict method.
   By defining the in- and outputs independent of the realisation of the citation recommendation, different methods performing the context-aware citation recommendation can be easily interchanged in the pipeline. 
   Moreover, the method signature was designed in such a way that it can be utilized for asynchronous operation of the pipeline with in- and output queues later on.

- `aae_recommender.py`: implements the AAE-Recommender as a prefetcher. The additional files `citeworth.py`, `condition.py`, `datasets.py`, `log.py`, `transform.py` and `ub.py` are needed by this implementation and have been
   copied directly from the repository of the [module](https://github.com/Data-Science-2Like/aae-rec-with-section-info).

- `baselines.py`: Implement our global baselines as prefetchers.

- `calc_candidate_pool.py`: Calculates static candidate pools for usage with a reranker module.

## References

[1] Iacopo Vagliano, Lukas Galke, and Ansgar Scherp. 2021. Recommendations
	for Item Set Completion: On the Semantics of Item Co-Occurrence With Data
	Sparsity, Input Size, and Input Modalities. arXiv:2105.04376 [cs.IR]