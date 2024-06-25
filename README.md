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
1. Indexing: run indexing.py to create the FF indexes for each dataset
2. Running the experiment: 