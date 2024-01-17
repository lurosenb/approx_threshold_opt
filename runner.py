import yaml

import gc

from metrics import tpr, fpr, precision, npv, accuracy, f1, selection_rate

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

from pipeline import FairDataset, FairPipeline, tpr_score, fpr_score, npv_score, selection_rate_score

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

my_classifiers = {
    # 'lgb': lgb.LGBMClassifier(verbose=-1)
    'xgb': xgb.XGBClassifier()
    # 'logistic_regression': LogisticRegression(),
    # 'random_forest': RandomForestClassifier(),
    # 'gradient_boosting': GradientBoostingClassifier(),
    # 'svc': SVC(probability=True),
    # 'knn': KNeighborsClassifier(),
    # 'mlp': MLPClassifier()
    # decision tree
}

# These are the metrics included in the results dict
metrics_dict = {
    'tpr': tpr_score,
    'fpr': fpr_score,
    'precision': precision_score,
    'npv': npv_score,
    'accuracy': accuracy_score,
    'f1': f1_score,
    'selection_rate': selection_rate_score
}

# This is what we are equalizing in the objective
metrics_functions = {
    'tpr': tpr,
    'fpr': fpr,
    'precision': precision,
    'npv': npv,
    'selection_rate': selection_rate
}

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

CONFIG_PATH = 'configs/master_config.yml'

config = load_config(CONFIG_PATH)
datasets = config['datasets']
classifier_config_path = 'configs/classifier_config.yml'

# set to True to estimate runtime
ESTIMATE_RUNTIME = False

all_results = pd.DataFrame()
for dataset_name, sensitive_attrs in datasets.items():
    print(f"Running pipeline for dataset: {dataset_name}")
    if dataset_name in ('ACSEmployment','ACSIncome','ACSMobility','ACSPublicCoverage','ACSTravelTime'):
        X = pd.read_csv(f'matrices/{dataset_name}/Xs.csv')
        y = pd.read_csv(f'matrices/{dataset_name}/ys.csv').squeeze()
    else:
        X = pd.read_csv(f'matrices/{dataset_name}/X.csv')
        y = pd.read_csv(f'matrices/{dataset_name}/y.csv').squeeze()

    # remove any rows that have null or nan
    X.dropna(inplace=True)
    y = y[X.index]

    dataset = FairDataset(X, y, sensitive_attrs)

    for sensitive_attr in sensitive_attrs:
        pipeline = FairPipeline(classifiers=my_classifiers, 
                                classifier_config_path=classifier_config_path, 
                                metrics=metrics_dict,
                                metric_functions=metrics_functions,
                                max_error=0.01, max_total_combinations=50000)

        pipeline.tune_and_evaluate(dataset, dataset_name, sensitive_attr)
        results = pipeline.results_df
        results['sensitive_attr'] = sensitive_attr
        results['dataset'] = dataset_name
        # all_results = all_results.append(results, ignore_index=True)
        all_results = pd.concat([all_results, results], ignore_index=True)
        print()
        print('Overall max_error')
        print(pipeline.overall_max_error)
        print()

    # this avoids memory issues
    del X, y, dataset
    gc.collect()

print("Pipeline run completed.")

all_results.to_pickle('all_results.pkl')