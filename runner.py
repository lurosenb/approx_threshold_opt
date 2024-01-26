import yaml
import argparse
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

global_metrics_map = {
    'f1': f1,
    'precision': precision,
    'npv': npv,
    'accuracy': accuracy,
}

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

parser = argparse.ArgumentParser(description='Run the pipeline for a specific dataset and with a custom output file name.')
parser.add_argument('--dataset_name', type=str, help='Name of the dataset to run the pipeline for', required=True)
parser.add_argument('--output_file', type=str, help='Name for the output .pkl file', required=True)

args = parser.parse_args()

CONFIG_PATH = 'configs/master_config.yml'

config = load_config(CONFIG_PATH)
datasets = config['datasets']
datasets_settings = config['datasets_settings']
classifier_config_path = 'configs/classifier_config.yml'

DATASET_NAME = args.dataset_name
OUTPUT_FILE = args.output_file

print(DATASET_NAME)
print(OUTPUT_FILE)

# set to True to estimate runtime
ESTIMATE_RUNTIME = False

if __name__ == '__main__':    
    all_results = pd.DataFrame()
    if DATASET_NAME in datasets:
        sensitive_attrs = datasets[DATASET_NAME]
        global_metric_setting = datasets_settings[DATASET_NAME][0]

        print(f"Running pipeline for dataset: {DATASET_NAME}")
        if DATASET_NAME in ('ACSEmployment','ACSIncome','ACSMobility','ACSPublicCoverage','ACSTravelTime'):
            X = pd.read_csv(f'matrices/{DATASET_NAME}/Xs.csv')
            y = pd.read_csv(f'matrices/{DATASET_NAME}/ys.csv').squeeze()
        else:
            X = pd.read_csv(f'matrices/{DATASET_NAME}/X.csv')
            y = pd.read_csv(f'matrices/{DATASET_NAME}/y.csv').squeeze()

        # remove any rows that have null or nan
        X.dropna(inplace=True)
        y = y[X.index]

        dataset = FairDataset(X, y, sensitive_attrs)

        for sensitive_attr in sensitive_attrs:
            pipeline = FairPipeline(classifiers=my_classifiers, 
                                    classifier_config_path=classifier_config_path, 
                                    metrics=metrics_dict,
                                    metric_functions=metrics_functions,
                                    global_metric=global_metrics_map[global_metric_setting],
                                    max_error=0.01, max_total_combinations=50000)

            pipeline.tune_and_evaluate(dataset, DATASET_NAME, sensitive_attr)
            results = pipeline.results_df
            results['sensitive_attr'] = sensitive_attr
            results['dataset'] = DATASET_NAME
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

        all_results.to_pickle(OUTPUT_FILE)
    else:
        print(f"Dataset {DATASET_NAME} not found in the configuration.")