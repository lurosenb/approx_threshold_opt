# Approximately Fair Models through Thresholding

### Description
Here we will describe the layout of our project and the purpose of each file.

### Files
- `README.md`: This file.
- `data/`: Directory containing the data used in the project. NOTE: Some datasets are not included, but can be downloaded using the `matrices.py` file.
- `matrices.py`: Python file containing the code to download and preprocess the datasets used in the project. From folktables, etc.
- `approx_thresh_general.py`, `approx_thresh_pytorch`: These are the sklearn-like implementations of the approximate threshold search methods (brute-force = general, sgd = pytorch, and epsilon net). They expect to fit and store the model object (i.e. XGBoostClassifier, etc.). They also provide plotting functions to visualize the results.
- `approx_thresh_light.py`, `approx_thresh_light_pytorch`: These are the light implementations of the approximate threshold search methods. They only expect and store the scores and labels (i.e. the y_prob output from predict_proba and y_true). They do not provide plotting functions. These are useful for sending to new threads, as it is less memory intensive.
- `pipeline.py`: Contains a class representing an entire pipeline for a given dataset. It includes the data preprocessing, model fitting, and approximate threshold search. It tracks parameters for each run and metrics and dumps everything into a dataframe.
- `runner.py`,`runner.sh`: Contains the code to iterate over datasets, run a pipeline for each datasets and each model (if so desired).
- `runtime_comparison.ipynb`: Jupyter notebook to compare the runtime of the approximate threshold search methods. It also contains the code to generate the plots for the runtime comparison.
- `ablation_scores.py`: Contains the code to run ablations over the datasets and models. 
- `ablation_scores_plotting.ipynb`: Contains the code to generate the plots for the ablation scores.
- `sample.ipynb`: Jupyter notebook to demonstrate how to use the approximate threshold search methods. It also contains the code to generate the plots for the sample, which gives intuition for the search space.
- `visualizing_proofs.ipynb`: Jupyter notebook containing code that we used to create some visuals for our proofs.
- `visualizing_soft_metrics.ipynb`: Jupyter notebook containing code that we used to create some visuals for visualizing soft metrics (NOTE: defaulting to synthetic data for convenience, but we ran on our real data for the plots in the paper).

### Note on results dataframes
We do not exhaustively include the results dataframes in this repository (produced by runner.py or by ablations.py over all datasets), as they are quite large. However, we can provide them upon request. All code to generate them is present in this repository.

### Requirements
In the `requirements.txt` file, we list the packages that are required to run the code in this repository. To install these packages, run the following command: `pip install -r requirements.txt`