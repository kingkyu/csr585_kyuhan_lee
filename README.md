# Large Scale Fact Checking Algorithm Using Entity-based Structural Balance 


This repository contains materials for a large scale fact checking algorithm. The contents of the repository will very likely to expand as the research progresses. Currently, we, in this repository, present related data sets and the implementaion of baseline models and our initial model.

To be specific, the repository includes a set of JSON files (i.e., "wiki-001.jsonl" - "wiki-109.jsonl") for June 2017 Wikipedia dump processed with Stanfor CoreNLP, training and test data files (i.e., "train.jsonl" and "test.jsonl"), a python file for indexing each Wikipedia page (i.e., "data_processing.py"), and multiple python files for the implementaion of the fact-checking models (i.e., "model.py").

The problem to be solved with contents in this repository is to verify whether a claim is "true", "false", or "neither true or false due to a lack of evidence". To elaborate, we validate whether a claim falls into one the three classes given documents that contain proper evidences for checking the veracity of the claim. For detailed description of the problem, please refer to Thorne et al. (2018). Note that while there are different models suggested in Thorne et al. (2018), we, in this repository, contains the implementation of the model of which results are presented in Table 5 of the paper.

The baseline model included in this repository uses TF and TF-IDF vectors of claims and their corresponding evidences as inputs for a multilayer perceptron algorithm and predicts the calims to belong to one of the three aforementioned classes. For a detailed explanation of the model, please refer to Riedel et al. (2017).

## Reproducibility

The implementaion of the baseline model can be easily reproduced by downloading all the contents in a local server and executing "data_processing.py" and "model.py" in a sequence.

### Prerequisites

The system was developed under the following versions of softwares and packages:

- sqlite3

- python==3.5.2

- numpy==1.14.5

- scipy==1.1.0

- sklearn==0.19.1

- spacy==2.0.16


### A step-by-step "how to":
#### (For an easy reproduction, we provide a link to the database containing processed data. If you download a sqlite3 db file from the link and place it in the same folder you put the files download from the repository, you will be able to reproduce the model only following step (1) and (6). In other words, you can skip step from (2) to (5).) 
#### Link: https://drive.google.com/open?id=1kPfE6u4m3qZW28CCpmTErhQBnBsRdAQW

(1) Download all the files in a local server.

(2) Download Wikipedia dumps from the following URL: https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
    
(3) Unzip the downloaded zip file and place the unzipped JSON files in the same directory where the files, downloaded from the 
    repository, are located.
    
(4) Make sure you delete the first line of "wiki-001.jsonl", one of the unzipped files you will encounter.

(5) Execute "data_processing.py" which will create a sqlite3 db called "fever.db".
    This database contains indices of individual Wikipedia pages which facilitates their retrieval process.
    
(6) Download a pretrained language model provided by Spacy. Enter "python -m spacy download en_core_web_lg" in your command line tool.
    
(7) Execute "model.py" which prints out the outcomes and produces "result.txt" that compares gold classes of data points and their predicted classes.


## References
- Riedel, B., Augenstein, I., Spithourakis, G. P., & Riedel, S. (2017). A simple but tough-to-beat baseline for the Fake News Challenge stance detection task. arXiv preprint arXiv:1707.03264.
- Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). FEVER: a large-scale dataset for Fact Extraction and VERification. arXiv preprint arXiv:1803.05355.
