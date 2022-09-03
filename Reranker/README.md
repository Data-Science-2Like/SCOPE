# Reranker
This module of the pipeline performs the context-aware citation recommendation for each sentence identified as cite-worthy by the cite-worthiness detection.
Thereby, it only takes the respective pool of candidate papers created by the prefetcher into account.
This way, the context-aware citation recommender reranks the candidate papers that were retrieved by the prefetcher.    

## Structure
`Reranker.py` implements the equally named abstract class `Reranker`.
This class should be extended by any specific reranker / context-aware citation recommendation module such that the predict method has a constant and pre-defined signature.  
By defining the in- and outputs independent of the realisation of the citation recommendation,
different methods performing the context-aware citation recommendation can be easily interchanged in the pipeline.
Moreover, the method signature was designed in such a way that it can be utilized for asynchronous operation of the pipeline with in- and output queues later on.  

`Reranker_BM25.py` implements the Local BM25 baseline that performs the context-aware citation recommendation by means of the information retrieval method BM25 [[2]](#2).
The `gensim_summarization_bm25.py` contains the former gensim library implementation of BM25. We had to copy it into a separate file since a recent version of the library, in which the BM25 implementation got removed, is required for the AAE-Recommender to run as a prefetcher.
For BM25, no pre-trained model is required.

`Reranker_Transformer.py` implements the context-aware citation recommendation by means of the SciBERT Reranker [[1]](#1) (although any other transformer model besides SciBERT could be utilized).
A SciBERT Reranker model utilizing section information can be trained as described in this repository: https://github.com/Data-Science-2Like/SciBERT-Reranker

For the three reranker files a documentation in the code is provided.

## References
<a id="1">[1]</a> 
Nianlong Gu, Yingqiang Gao, and Richard H. R. Hahnloser. 2022. Local Citation Recommendation with Hierarchical-Attention Text Encoder and SciBERT-Based Reranking. In Advances in Information Retrieval - 44th European Conference on IR Research, ECIR 2022 (Lecture Notes in Computer Science, Vol. 13185), Matthias Hagen, Suzan Verberne, Craig Macdonald, Christin Seifert, Krisztian Balog, Kjetil Nørvåg, and Vinay Setty (Eds.). Springer, 274–288. https://doi.org/10.1007/978-3-030-99736-6_19

<a id="2">[2]</a>
Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Framework: BM25 and Beyond. Found. Trends Inf. Retr. 3, 4 (apr 2009), 333–389. https://doi.org/10.1561/1500000019