# CSE3000-Research-Project-Neural-Ranking-Models
The repository stores codebase for the implementation of experiments with rank fusion function in the Fast-Forward indexes setting.
The framework takes the retrieve-and-rerank approach of neural ranking models with the following pipeline:
1. Retrieve candidates using sparse scores.
2. Compute dense scores of the retrieved candidates.
3. Interpolate the two scores using a rank fusion function and obtain the final ranking of candidates.

The implementation of the FF indexes is available in the following repository: https://github.com/mrjleo/fast-forward-indexes

This repository saves the experiment code and data of the results 
for the _Neural Ranking Models_ project for the 2024 CSE3000 Research Project.
The documentations of the project can be found in https://github.com/TU-Delft-CSE/Research-Project?tab=readme-ov-file

## Steps to Follow for Reproducing the Experiment
There are three experiments conducted for this research. 
In prior to the experiment, run indexing.py for all datasets to retrieve the FF index.
### Ranking Effectiveness Experiment
1. Validation: run validation.py for a dataset if it is available (FiQA-2018, MS MARCO, DBPedia, FEVER, NFCorpus, QUORA) 
2. Experiment: run experiment.py
### Latency Experiment
Latency experiment is available only for Arguana and QUORA. Run the latency_experiment.py.
### Ranking Change Experiment
Available via the Jupyter notebook file.

## Experiment Results
The experiment results used for the project is stored under the results folder.