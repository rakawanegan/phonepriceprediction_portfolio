# Phone Price Prediction Bigenner Cup

=============================

This repository contains the following files and directories:

- `Makefile`: Makefile for project automation.
- `README.md`: Readme file providing an overview of the project.
- `analysis/`: Directory containing Jupyter Notebook files for analysis.
  - `first_analysis.ipynb`: Jupyter Notebook for initial analysis.
- `baseline.py`: Python script for the baseline implementation.
- `config/`: Directory containing configuration files.
  - `sample.ini`: Sample configuration file.
- `data/`: Directory containing data files.
  - `additive_data/`: Directory for additional data files.
  - `official_data/`: Directory for official data files.
    - `sample_submission.csv`: Sample submission file.
    - `test.csv`: Test data file.
    - `train.csv`: Training data file.
- `main.py`: Main Python script for the project.
- `predictor/`: Directory containing model prediction files.
  - `LightGBMoptunamodel.py`: Python script for LightGBM model with Optuna.
  - `NeuralNetworkmodel.py`: Python script for Neural Network model.
  - `__pycache__/`: Cached files for Python scripts.
    - `LightGBMoptunamodel.cpython-310.pyc`: Cached file for LightGBM model with Optuna.
    - `NeuralNetworkmodel.cpython-310.pyc`: Cached file for Neural Network model.
    - `kNearestNeighbor.cpython-310.pyc`: Cached file for kNearestNeighbor.
    - `kNearestNeighbormodel.cpython-310.pyc`: Cached file for kNearestNeighbor model.
  - `kNearestNeighbormodel.py`: Python script for k-Nearest Neighbor model.
- `results/`: Directory for result files.
  - `model/`: Directory for model files.
    - `LightGBM.model`: Trained LightGBM model file.
    - `NeuralNetwork.model`: Trained Neural Network model file.
    - `kNearestNeighbor.model`: Trained k-Nearest Neighbor model file.
  - `submit/`: Directory for submission files.
    - `knnpredict.csv`: Submission file for k-Nearest Neighbor model.
    - `lgbmsample1predict.csv`: Submission file for LightGBM model.
    - `nnsample1predict.csv`: Submission file for Neural Network model.
