import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objects as go

from itertools import product
from math import ceil

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
from metrics import tpr, fpr, precision, npv, accuracy, f1, selection_rate

MAX_WORKERS=32

class ApproxThresholdGeneral(BaseEstimator, ClassifierMixin):
    def __init__(self, metric_functions, lambda_=0.5, global_metric=f1, max_epsilon=1.0, max_error=0, max_total_combinations=1):
        self.metric_functions = metric_functions  # list of metric functions
        self.lambda_ = lambda_
        self.thresholds_ = None
        self.epsilons_ = None
        self.global_metric = global_metric
        self.max_epsilon = max_epsilon
        self.best_objective_value = None
        self.y_prob = None

    def fit(self, y_prob, y, A):
        self.group_metrics = {}
        self.group_thresholds = {}
        unique_groups = np.unique(A)
        
        self.thresholds_, self.epsilons_ = self._find_intersection(y, y_prob, A, unique_groups, self.metric_functions, lambda_=self.lambda_)
        return self

    def predict(self, y_prob, A):
        adjusted_labels = np.zeros(A.shape)

        unique_groups = np.unique(A)
        for group in unique_groups:
            mask = A == group
            adjusted_labels[mask] = np.where(y_prob[mask] > self.thresholds_[group], 1, 0)
        
        return adjusted_labels

    def _compute_metrics(self, y_true, y_prob, threshold):
        metrics = {}
        for metric_name, metric_func in self.metric_functions.items():
                metrics[metric_name] = metric_func(y_true, y_prob, threshold)
        return metrics

    def _compute_all_thresholds_per_group(self, y_true, y_prob, unique_groups, A):
        group_thresholds = {}
        for group in unique_groups:
            mask_group = A == group
            group_thresholds[group] = np.unique(y_prob[mask_group])
        return np.array(np.meshgrid(*group_thresholds.values())).T.reshape(-1, len(unique_groups))

    def _find_intersection(self, y_true, y_prob, A, unique_groups, metrics_functions, lambda_=0.5, resource_constraint=None):
        best_objective_value = float('inf')
        best_thresholds = {}
        best_epsilons = {}
        all_threshold_combinations = self._compute_all_thresholds_per_group(y_true, y_prob, unique_groups, A)

        for group in unique_groups:
            self.group_metrics[group] = {}

        for idx, threshold_combination in enumerate(tqdm(all_threshold_combinations, desc="Processing combinations")):
            combined_preds = []
            combined_true = []
            objective = 0
            temp_group_metrics = {}
            total_size = 0
            weighted_acc = 0
            max_epsilon_violation = False

            for i, group in enumerate(unique_groups):
                mask_group = A == group
                temp_group_metrics[group] = self._compute_metrics(y_true[mask_group], y_prob[mask_group], threshold_combination[i])
                self.group_metrics[group][idx] = temp_group_metrics[group]

                adjusted_labels = y_prob[mask_group] > threshold_combination[i]
                combined_preds.extend(adjusted_labels)
                combined_true.extend(y_true[mask_group])

                group_acc = self.global_metric(y_true[mask_group], y_prob[mask_group], threshold_combination[i])
                group_size = len(y_true[mask_group])
                weighted_acc += group_acc * group_size
                total_size += group_size

            weighted_acc /= total_size

            temp_epsilons = {}
            for i, group in enumerate(unique_groups):
                group_metric_vector = np.array([temp_group_metrics[group][metric_name] for metric_name in metrics_functions.keys()])
                temp_epsilons[group] = {}

                for other_group in unique_groups:
                    if group != other_group:
                        other_group_metric_vector = np.array([temp_group_metrics[other_group][metric_name] for metric_name in metrics_functions.keys()])
                        distances = cdist(group_metric_vector.reshape(1, -1), other_group_metric_vector.reshape(1, -1), metric='euclidean')
                        objective += np.sum(distances)

                        # Check for max epsilon violation
                        epsilon_diff = np.abs(group_metric_vector - other_group_metric_vector)
                        if np.any(epsilon_diff > self.max_epsilon):
                            max_epsilon_violation = True
                        temp_epsilons[group][other_group] = epsilon_diff

            objective += lambda_ * (1 - weighted_acc)

            if resource_constraint is not None and sum(combined_preds) > resource_constraint:
                continue

            if not max_epsilon_violation and objective < best_objective_value:
                best_objective_value = objective
                best_thresholds = dict(zip(unique_groups, threshold_combination))
                best_epsilons = temp_epsilons
                self.best_index = idx

        print(f'Best objective value: {best_objective_value}')
        print(f'Best thresholds: {best_thresholds}')
        print(f'Epsilon differences: {best_epsilons}')
        self.best_objective_value = best_objective_value
        return best_thresholds, best_epsilons


class ApproxThresholdNet(ApproxThresholdGeneral):
    """
    Here, we use an epsilon net to find the best threshold combination. 
    This is a more efficient method than the previous one, 
    and is guaranteed to find a solution within some error margin to the optimal,
    assuming the metric functions are Lipschitz continuous with constant less than or
    equal to 1. Then, an overall Lipschitz constant of the objective function is 
    less than or equal to 2 * sqrt(2) * |G| * |M| + lambda, where |G| is the number 
    of groups and |M| is the number of metric functions.
    See derivation in the paper.

    This provides a more efficient method for finding the best threshold combination,
    because the number of points in the epsilon net is only dependent on the error margin
    and a Lipschitz constant of the objective function, which is much smaller than the
    number of points in the data.

    However, the metric functions are also assumed to be 1-Lipschitz continuous, which
    is not always the case. The "soft" or smoothed versions of the metric functions
    are (likely) 1-Lipschitz continuous, but the original metric functions are not.
    So, though principled, this method is still heuristic.
    """
    def __init__(self, metric_functions, lambda_=0.5, global_metric=f1, max_epsilon=1.0, max_error=0.01, L_f=None, max_total_combinations=100000):
        super().__init__(metric_functions, lambda_=lambda_, global_metric=global_metric, max_epsilon=max_epsilon)
        self.max_error = max_error
        self.L_f = L_f
        self.max_total_combinations = max_total_combinations

    def evaluate_combination(self, threshold_combination, y_true, y_prob, A, unique_groups, metrics_functions, lambda_):
        combined_preds = []
        combined_true = []
        objective = 0
        temp_group_metrics = {}
        total_size = 0
        weighted_acc = 0
        max_epsilon_violation = False

        for i, group in enumerate(unique_groups):
            mask_group = A == group
            temp_group_metrics[group] = self._compute_metrics(y_true[mask_group], y_prob[mask_group], threshold_combination[i])
            
            adjusted_labels = y_prob[mask_group] > threshold_combination[i]
            combined_preds.extend(adjusted_labels)
            combined_true.extend(y_true[mask_group])
            
            # NOTE: change to be self.global_metric
            group_acc = self.global_metric(y_true[mask_group], y_prob[mask_group], threshold_combination[i])
            group_size = len(y_true[mask_group])
            weighted_acc += group_acc * group_size
            total_size += group_size

        weighted_acc /= total_size

        temp_epsilons = {}
        for i, group in enumerate(unique_groups):
            temp_epsilons[group] = {}
            group_metric_vector = np.array([temp_group_metrics[group][metric_name] for metric_name in metrics_functions.keys()])

            for other_group in unique_groups:
                if group != other_group:
                    other_group_metric_vector = np.array([temp_group_metrics[other_group][metric_name] for metric_name in metrics_functions.keys()])
                    distances = cdist(group_metric_vector.reshape(1, -1), other_group_metric_vector.reshape(1, -1), metric='euclidean')
                    objective += np.sum(distances)
            
                    # Check for max epsilon violation
                    epsilon_diff = np.abs(group_metric_vector - other_group_metric_vector)
                    if np.any(epsilon_diff > self.max_epsilon):
                        max_epsilon_violation = True
                    temp_epsilons[group][other_group] = epsilon_diff
        
        objective += lambda_ * (1 - weighted_acc)

        return objective, threshold_combination, temp_epsilons, max_epsilon_violation

    def _find_intersection(self, y_true, y_prob, A, unique_groups, metrics_functions, lambda_=0.5):
        binom_metric_funcs = len(metrics_functions) * (len(metrics_functions) - 1) / 2

        # NOTE: this is often theoretically 1/n, but we conservatively use 1/log(n)
        lipschitz_constant_g_i = 2 * (1 / np.log(len(y_true)))

        if self.L_f is None:
            self.L_f = lipschitz_constant_g_i * len(unique_groups) * binom_metric_funcs + lambda_

        range_f = len(unique_groups) * binom_metric_funcs + lambda_

        max_error = self.max_error
        epsilon = max_error * range_f / self.L_f
        N = ceil(1 / epsilon) + 1
        total_combinations = N ** len(unique_groups)

        # adjust max_error if total_combinations exceeds max_total_combinations
        if total_combinations > self.max_total_combinations:
            N = ceil((self.max_total_combinations ** (1/len(unique_groups)))) + 1
            epsilon = 1 / (N - 1)
            max_error = epsilon * self.L_f / range_f
            total_combinations = N ** len(unique_groups)
            print(f"Adjusted max_error to {max_error} to limit total combinations to approximately {self.max_total_combinations}")
            self.max_error = max_error
            
        print(f'Number of points in the epsilon net: {N}')
        print(f'Adjusted max_error: {max_error}')
        print(f'Number of points in data: {len(y_true)}')

        threshold_points = np.linspace(0, 1, N)
        objectives = {}
        epsilons = {}

        # best_epsilons = {}
        combos = list(product(threshold_points, repeat=len(unique_groups)))
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_combination = {executor.submit(self.evaluate_combination, tc, y_true, y_prob, A, unique_groups, metrics_functions, lambda_): tc for tc in combos}
            
            for future in tqdm(as_completed(future_to_combination), total=total_combinations, desc="Threshold Combinations"):
                threshold_combination = future_to_combination[future]
                objective_value, _, temp_epsilons, max_epsilon_violation = future.result()
                
                if not max_epsilon_violation:
                    objectives[threshold_combination] = objective_value
                    epsilons[threshold_combination] = temp_epsilons

            executor.shutdown(wait=True)

        # find the best objective value and corresponding thresholds after collecting all results
        best_threshold_combination = min(objectives, key=objectives.get)
        best_objective_value = objectives[best_threshold_combination]
        best_epsilons = epsilons[best_threshold_combination]
        best_thresholds = dict(zip(unique_groups, best_threshold_combination))

        print(f'Best objective value: {best_objective_value}')
        print(f'Best thresholds: {best_thresholds}')
        print(f'Epsilon differences: {best_epsilons}')
        self.best_objective_value = best_objective_value
        return best_thresholds, best_epsilons
    