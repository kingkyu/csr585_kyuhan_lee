# Large Scale Fact Checking Algorithm: From A Network Theory Perspective

This repository contains materials for a large scale fact checking algorithm. The contents of the repository will very likely to expand as the research progresses. Currently, we, in this repository, present related data sets and the implementaion of a baseline model.

To be specific, the repository includes a set of JSON files (i.e., "wiki-001.jsonl" - "wiki-109.jsonl") for June 2017 Wikipedia dump processed with Stanfor CoreNLP, training and test data files (i.e., "train.jsonl" and "test.jsonl"), a python file for indexing each Wikipedia page (i.e., "data_processing.py"), and another python file for the implementaion of the baseline model (i.e., "model.py").

The base-line model is the implementation of one of the models described in Thorne et al. (2018) and Riedel et al. (2017). 

- Riedel, B., Augenstein, I., Spithourakis, G. P., & Riedel, S. (2017). A simple but tough-to-beat baseline for the Fake News Challenge stance detection task. arXiv preprint arXiv:1707.03264.
- Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). FEVER: a large-scale dataset for Fact Extraction and VERification. arXiv preprint arXiv:1803.05355.
