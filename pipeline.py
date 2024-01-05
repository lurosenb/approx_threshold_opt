import numpy as np
import pandas as pd
import yaml
import time

from sklearn.model_selection import GridSearchCV

from approx_thresh_general import ApproxThresholdNet as ApproxThresholdGeneral

from tqdm import tqdm

def loweest_performing(metric_func, y_true, y_pred, sensitive_attr_values):
    # calculates a given metric for each subgroup and returns the value for the lowest performing group.
    unique_groups = np.unique(sensitive_attr_values)
    group_metrics = [metric_func(y_true[sensitive_attr_values == group], y_pred[sensitive_attr_values == group]) for group in unique_groups]
    return min(group_metrics)

def accuracy(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return np.sum(y_pred == y_true) / len(y_true)

def f1(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

class FairDataset:
    def __init__(self, dataframe, target, sensitive_attrs):
        self.dataframe = dataframe
        self.target = target
        self.sensitive_attrs = sensitive_attrs

    def get_data(self):
        return self.dataframe, self.target, self.sensitive_attrs
    
class FairPipeline:
    def __init__(self, classifiers, 
                 classifier_config_path, 
                 metrics, 
                 metric_functions, 
                 lambdas=[0.1,0.3,0.5,0.7,0.9], 
                 max_error=0.02, 
                 max_total_combinations=1000):
        self.classifiers = classifiers
        self.metrics = metrics
        self.metric_functions = metric_functions
        self.lambdas = lambdas
        self.param_grids = self.load_param_grids(classifier_config_path)
        self.results_df = pd.DataFrame()
        self.max_error = max_error
        self.max_total_combinations = max_total_combinations

    def load_param_grids(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def tune_and_evaluate(self, dataset, dataset_name, sensitive_attr):
        X, y, _ = dataset.get_data()
        A = dataset.dataframe[sensitive_attr].values 

        for clf_name, clf in self.classifiers.items():
            if clf_name in self.param_grids:
                grid_search = GridSearchCV(clf, self.param_grids[clf_name])
                grid_search.fit(X, y)
                best_clf = grid_search.best_estimator_
                best_params = grid_search.best_params_

                self.evaluate_classifier(best_clf, dataset, clf_name, best_params, 'original', dataset_name, X, y, A)

                for l in self.lambdas:
                    fair_clf = ApproxThresholdGeneral(clf, 
                                                    self.metric_functions, 
                                                    l, 
                                                    max_error=self.max_error, 
                                                    max_total_combinations=self.max_total_combinations)
                    fair_clf.fit(X, y, A)
                    
                    fair_clf_info = {
                        'best_objective_value': fair_clf.best_objective_value,  
                        'best_thresholds': fair_clf.thresholds_,  
                        'best_epsilons': fair_clf.epsilons_,
                        'max_epsilon': fair_clf.max_epsilon,
                        'lambda': fair_clf.lambda_,
                        'global_metric': str(fair_clf.global_metric.__name__),
                    }
                            
                    self.evaluate_classifier(fair_clf, dataset, clf_name, best_params, 'fair', dataset_name, X, y, A, fair_clf_info)
                    del fair_clf

                del grid_search
                del best_clf
                
                
    def evaluate_classifier(self, classifier, dataset, clf_name, hyperparams, method, dataset_name, X, y, A, fair_clf_info=None):
        if method == 'fair':
            y_pred = classifier.predict(X, A)  # fair classifiers requires sensitive attribute
        else:
            y_pred = classifier.predict(X)

        overall_metrics = self.compute_metrics(y, y_pred)
        overall_result = {
            'dataset': dataset_name,
            'classifier': clf_name,
            'dataset_subset': 'overall',
            'method': method,
            **overall_metrics,
            **hyperparams
        }

        if method == 'fair' and fair_clf_info:
            overall_result.update(fair_clf_info)

        print(f"Overall metrics for {clf_name} ({method}): {overall_metrics}")
        print(f"Hyperparameters: {hyperparams}")
        self.results_df = self.results_df.append(overall_result, ignore_index=True)

        for group_value in np.unique(A):
            group_metrics = self.compute_metrics(y[A == group_value], y_pred[A == group_value])
            group_result = {
                'dataset': dataset_name,
                'classifier': clf_name,
                'dataset_subset': f'{group_value}',
                'method': method,
                **group_metrics,
                **hyperparams
            }

            if method == 'fair' and fair_clf_info:
                group_result.update(fair_clf_info)

            self.results_df = self.results_df.append(group_result, ignore_index=True)
            print(f"Metrics for {clf_name} ({method}) on {group_value}: {group_metrics}")
            print(f"Hyperparameters: {hyperparams}")
    
    def compute_metrics(self, y_true, y_pred):
        metrics_result = {}
        for metric_name, metric_func in self.metrics.items():
            metrics_result[metric_name] = metric_func(y_true, y_pred)
        return metrics_result
    
    def estimate_runtime(self, dataset, sensitive_attr):
        total_estimated_time = 0
        X, y, _ = dataset.get_data()
        A = dataset.dataframe[sensitive_attr].values

        print("Estimating runtime for each classifier...")
        for clf_name, clf in tqdm(self.classifiers.items(), desc="Classifiers"):
            if clf_name in self.param_grids:
                print(f"Estimating runtime for classifier: {clf_name}")

                param_setting = {k: v[0] for k, v in self.param_grids[clf_name].items()}
                clf.set_params(**param_setting)

                start_time = time.time()
                clf.fit(X, y)
                runtime_original = time.time() - start_time

                fair_clf = ApproxThresholdGeneral(clf, self.metric_functions, self.lambda_, max_error=self.max_error, max_total_combinations=self.max_total_combinations)
                start_time = time.time()
                fair_clf.fit(X, y, A)
                runtime_fair = time.time() - start_time

                num_param_configs = np.prod([len(values) for values in self.param_grids[clf_name].values()])
                estimated_clf_time = (runtime_original + runtime_fair) * num_param_configs
                print(f"Estimated time for {clf_name} (both versions): {estimated_clf_time:.2f} seconds")

                total_estimated_time += estimated_clf_time

        print(f"Total estimated runtime (including fairness adjustments): {total_estimated_time:.2f} seconds")
        print()
        return total_estimated_time
