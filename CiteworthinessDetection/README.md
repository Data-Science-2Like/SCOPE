# Cite-worthiness Detection
This module of the pipeline performs the classification of sentences regarding their cite-worthiness.
Only sentences that are cite-worthy are further processed by the pipeline.

## Structure
`CiteworthinessDetection.py` implements the equally named abstract class `CiteworthinessDetection`.
This class should be extended by any specific cite-worthiness detection module such that the predict method has a constant and pre-defined signature.  
By defining the in- and outputs independent of the realisation of the cite-worthiness detection,
different methods performing the cite-worthiness detection can be easily interchanged in the pipeline.
Moreover, the method signature was designed in such a way that it can be utilized for asynchronous operation of the pipeline with in- and output queues later on.  

`CiteWorth.py` implements the cite-worthiness detection by means of the CiteWorth [[1]](#1) module.  
A CiteWorth model utilizing section information can be trained as described in this repository: https://github.com/Data-Science-2Like/cite-worth-with-section-info

For both files a documentation in the code is provided.

## References
<a id="1">[1]</a>
Dustin Wright and Isabelle Augenstein. 2021. CiteWorth: Cite-Worthiness Detec-tion for Improved Scientific Document Understanding. InFindings of the Associ-ation for Computational Linguistics: ACL-IJCNLP 2021. Association for Computa-tional Linguistics, Online, 1796–1807. https://doi.org/10.18653/v1/2021.findings-acl.157
