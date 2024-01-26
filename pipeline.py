import numpy as np
import pandas as pd
import yaml
import time

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from fairlearn.postprocessing import ThresholdOptimizer

from sklearn.metrics import roc_auc_score, f1_score
from metrics import tpr, fpr, precision, npv, accuracy, f1, selection_rate

# DELETE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from approx_thresh_light import ApproxThresholdNet, ApproxThresholdGeneral

from tqdm import tqdm

def tpr_score(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    P = np.sum(y_true == 1)
    return TP / P if P > 0 else 0

def fpr_score(y_true, y_pred):
    FP = np.sum((y_pred == 1) & (y_true == 0))
    N = np.sum(y_true == 0)
    return FP / N if N > 0 else 0

def npv_score(y_true, y_pred):
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return TN / (TN + FN) if (TN + FN) > 0 else 0

def selection_rate_score(y_true, y_pred):
    total_predictions = len(y_pred)
    positive_predictions = np.sum(y_pred == 1)

    return positive_predictions / total_predictions if total_predictions > 0 else 0

def lowest_performing(metric_func, y_true, y_pred, sensitive_attr_values):
    # calculates a given metric for each subgroup and returns the value for the lowest performing group.
    unique_groups = np.unique(sensitive_attr_values)
    group_metrics = [metric_func(y_true[sensitive_attr_values == group], y_pred[sensitive_attr_values == group]) for group in unique_groups]
    return min(group_metrics)
    
class FairDataset:
    def __init__(self, dataframe, target, sensitive_attrs):
        self.dataframe = dataframe
        self.target = target
        self.sensitive_attrs = sensitive_attrs

    def get_data(self):
        return self.dataframe, self.target, self.sensitive_attrs
    
class FairPipeline:
    def __init__(self, classifiers, classifier_config_path, metrics, metric_functions, global_metric=f1, 
                       lambdas=[0.1,1.0,10.0], 
                       max_error=0.02, 
                       max_total_combinations=1000, 
                       random_state=42, 
                       calibrate=False,
                       hardt_model_default_constraint='selection_rate_parity',
                       hardt_model_default_objective='balanced_accuracy_score',):
        self.classifiers = classifiers
        self.metrics = metrics
        self.metric_functions = metric_functions
        self.lambdas = lambdas
        self.param_grids = self.load_param_grids(classifier_config_path)
        self.results_df = pd.DataFrame()
        self.max_error = max_error
        self.max_total_combinations = max_total_combinations
        self.overall_max_error = 0
        self.random_state = random_state
        self.calibrate = calibrate
        self.global_metric = global_metric
        self.hardt_model_default_constraint = hardt_model_default_constraint
        self.hardt_model_default_objective = hardt_model_default_objective

    def load_param_grids(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def tune_and_evaluate(self, dataset, dataset_name, sensitive_attr):
        X, y, _ = dataset.get_data()
        A = dataset.dataframe[sensitive_attr].values

        X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A, test_size=0.3, random_state=self.random_state)
        

        for clf_name, clf in self.classifiers.items():
            if clf_name not in self.param_grids:
                continue

            random_search = RandomizedSearchCV(clf, 
                                                param_distributions=self.param_grids[clf_name], 
                                                n_iter=10, 
                                                cv=2, 
                                                verbose=2, 
                                                random_state=self.random_state, 
                                                n_jobs=-1)
            random_search.fit(X_train, y_train)
            best_clf = random_search.best_estimator_
            best_params = random_search.best_params_

            yhat_train_erm = best_clf.predict_proba(X_train)[:,1]
            yhat_test_erm = best_clf.predict_proba(X_test)[:,1]                          
            
            train_erm = pd.DataFrame({'s':yhat_train_erm, 'y': y_train.astype('int')})
            test_erm = pd.DataFrame({'s':yhat_test_erm, 'y': y_test.astype('int')})
            roc_auc_score(train_erm['y'], train_erm['s'])
            roc_auc_score(test_erm['y'], test_erm['s'])
            print('Following metrics are OVERALL:')
            print('PRE Train AUC: ', roc_auc_score(train_erm['y'], train_erm['s']))
            print('PRE Test AUC: ', roc_auc_score(test_erm['y'], test_erm['s']))
            print('PRE Calibration Train F1 Score: ', f1_score(y_train, best_clf.predict(X_train)))
            print('PRE Calibration Test F1 Score: ', f1_score(y_test, best_clf.predict(X_test)))

            if self.calibrate:
                print('Calibrating...')
                best_clf = CalibratedClassifierCV(best_clf, method='isotonic', cv='prefit')
                best_clf.fit(X_train, y_train)

                print('POST Calibration Train AUC: ', roc_auc_score(y_train, best_clf.predict_proba(X_train)[:, 1]))
                print('POST Calibration Test AUC: ', roc_auc_score(y_test, best_clf.predict_proba(X_test)[:, 1]))
                print('POST Calibration Train F1 Score: ', f1_score(y_train, best_clf.predict(X_train)))
                print('POST Calibration Test F1 Score: ', f1_score(y_test, best_clf.predict(X_test)))

            print()
            del random_search

            y_prob_train = best_clf.predict_proba(X_train)[:, 1]
            y_prob_test = best_clf.predict_proba(X_test)[:, 1]

            self.evaluate_classifier(best_clf, dataset, clf_name, best_params, 'original', dataset_name, X_test, y_test, A_test, y_prob=y_prob_test)

            hardt_model = ThresholdOptimizer(
                estimator=best_clf,
                constraints=self.hardt_model_default_constraint,
                objective=self.hardt_model_default_objective,
                prefit=True,
                predict_method='predict_proba'
            )
            hardt_model.fit(X_train, y_train, sensitive_features=X_train['RAC1P'])

            self.evaluate_classifier(hardt_model, dataset, clf_name, best_params, 'hardt', dataset_name, X_test, y_test, A_test, y_prob=y_prob_test, best_clf=best_clf)

            for l in self.lambdas:
                if 'mfopt' in dataset_name:
                    fair_clf = ApproxThresholdNet(metric_functions=self.metric_functions, 
                                                    lambda_=l, 
                                                    max_error=0.001, 
                                                    max_total_combinations=self.max_total_combinations,
                                                    global_metric=self.global_metric)
                else:
                    fair_clf = ApproxThresholdNet(metric_functions=self.metric_functions, 
                                                    lambda_=l, 
                                                    max_error=self.max_error, 
                                                    max_total_combinations=self.max_total_combinations,
                                                    global_metric=self.global_metric)

                fair_clf.fit(y_prob_train, y_train, A_train)
                
                if self.overall_max_error < fair_clf.max_error:
                    self.overall_max_error = fair_clf.max_error

                fair_clf_info = {
                    'best_objective_value': fair_clf.best_objective_value,
                    'best_thresholds': fair_clf.thresholds_,
                    'best_epsilons': fair_clf.epsilons_,
                    'max_epsilon': fair_clf.max_epsilon,
                    'lambda': fair_clf.lambda_,
                    'global_metric': str(fair_clf.global_metric.__name__),
                }
                        
                self.evaluate_classifier(fair_clf, dataset, clf_name, best_params, 'fair', dataset_name, X_test, y_test, A_test, fair_clf_info, y_prob=y_prob_test, best_clf=best_clf)
                del fair_clf
            

    def evaluate_classifier(self, classifier, dataset, clf_name, hyperparams, method, dataset_name, X, y, A, fair_clf_info=None, y_prob=None, best_clf=None):
        # NOTE: best_clf and classifier are the same classifier, this is just semantic
        if method == 'fair':
            y_prob = best_clf.predict_proba(X)[:, 1]
        elif method == 'hardt':
            y_prob = best_clf.predict_proba(X)[:, 1]
        else:
            y_prob = classifier.predict_proba(X)[:, 1]

        auc_score = roc_auc_score(y, y_prob)

        if method == 'fair':
            y_pred = classifier.predict(y_prob, A)
        elif method == 'hardt':
            y_pred = classifier.predict(X, sensitive_features=A)
        else:
            y_pred = classifier.predict(X)

        overall_metrics = self.compute_metrics(y, y_pred)
        overall_metrics['AUC'] = auc_score  # Add AUC to metrics
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
        self.results_df = pd.concat([self.results_df, pd.DataFrame([overall_result])], ignore_index=True)

        for group_value in np.unique(A):
            group_metrics = self.compute_metrics(y[A == group_value], y_pred[A == group_value])
            group_metrics['AUC'] = roc_auc_score(y[A == group_value], y_prob[A == group_value]) 
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

            self.results_df = pd.concat([self.results_df, pd.DataFrame([group_result])], ignore_index=True)
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

                y_prob = clf.predict_proba(X)[:, 1]
                fair_clf = ApproxThresholdGeneral(self.metric_functions, self.lambdas[0], max_error=self.max_error, max_total_combinations=self.max_total_combinations)
                start_time = time.time()
                fair_clf.fit(y_prob, y, A)
                runtime_fair = time.time() - start_time

                num_param_configs = np.prod([len(values) for values in self.param_grids[clf_name].values()])
                estimated_clf_time = (runtime_original + runtime_fair) * num_param_configs
                print(f"Estimated time for {clf_name} (both versions): {estimated_clf_time:.2f} seconds")

                total_estimated_time += estimated_clf_time

        print(f"Total estimated runtime (including fairness adjustments): {total_estimated_time:.2f} seconds")
        print()
        return total_estimated_time
