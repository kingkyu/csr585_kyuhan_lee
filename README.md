# Large Scale Fact Checking Algorithm: From A Network Theory Perspective

This repository contains materials for a large scale fact checking algorithm. The contents of the repository will very likely to expand as the research progresses. Currently, we, in this repository, present related data sets and the implementaion of a baseline model.

To be specific, the repository includes a set of JSON files (i.e., "wiki-001.jsonl" - "wiki-109.jsonl") for June 2017 Wikipedia dump processed with Stanfor CoreNLP, training and test data files (i.e., "train.jsonl" and "test.jsonl"), a python file for indexing each Wikipedia page (i.e., "data_processing.py"), and another python file for the implementaion of the baseline model (i.e., "model.py").

The problem to be solved with contents in this repository is to verify whether a claim is "true", "false", or "neither true or false due to a lack of evidence". To elaborate, we validate whether a claim falls into one the three classes given documents that contain proper evidences for checking the veracity of the claim. For detailed description of the problem, please refer to Thorne et al. (2018). Note that while there are different models suggested in Thorne et al. (2018), we, in this repository, contains the implementation of the model of which results are presented in Table 5 of the paper.

The baseline model included in this repository uses TF and TF-IDF vectors of claims and their corresponding evidences as inputs for a multilayer perceptron algorithm and predicts the calims to belong to one of the three aforementioned classes. For a detailed explanation of the model, please refer to Riedel et al. (2017).

## Reproducibility

The implementaion of the baseline model can be easily reproduced by downloading all the contents in a local server and executing "data_processing.py" and "model.py" in a sequence.

### Prerequisites

The system was developed under the following versions of softwares and packages:
'''
Python==3.5.2
numpy==1.14.5
scipy==1.1.0
sklearn==0.19.1
'''

### A step-by-step "how to"
(1) 

## References
- Riedel, B., Augenstein, I., Spithourakis, G. P., & Riedel, S. (2017). A simple but tough-to-beat baseline for the Fake News Challenge stance detection task. arXiv preprint arXiv:1707.03264.
- Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). FEVER: a large-scale dataset for Fact Extraction and VERification. arXiv preprint arXiv:1803.05355.
