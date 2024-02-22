import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin

from itertools import product
from math import ceil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from metrics import f1

MAX_WORKERS=4

class ApproxThreshold(BaseEstimator, ClassifierMixin):
    def __init__(self, metric_functions, lambda_=0.5, global_metric=f1, max_epsilon=1.0):
        self.metric_functions = metric_functions  # list of metric functions
        self.lambda_ = lambda_
        self.thresholds_ = None
        self.epsilons_ = None
        self.global_metric = global_metric
        self.max_epsilon = max_epsilon
        self.best_objective_value = None
        self.y_prob = None
        self.resource_constraint = None

    def fit(self, y_prob, y, A):
        """
        Assumes a prefit model, so just finds the best threshold combination

        :param y_prob: predicted probabilities
        :param y: true labels
        :param A: group labels
        """
        self.group_metrics = {}
        self.group_thresholds = {}
        unique_groups = np.unique(A)
        
        self.thresholds_, self.epsilons_ = self._find_intersection(y, y_prob, A, unique_groups, self.metric_functions, lambda_=self.lambda_)
        return self

    def predict(self, y_prob, A):
        """
        Again, assumes a prefit model, so just adjusts the labels based on the fit thresholds
        
        :param y_prob: predicted probabilities
        :param A: group labels

        :return: adjusted labels
        """
        adjusted_labels = np.zeros(A.shape)

        unique_groups = np.unique(A)
        for group in unique_groups:
            mask = A == group
            adjusted_labels[mask] = np.where(y_prob[mask] > self.thresholds_[group], 1, 0)
        
        return adjusted_labels

    def _compute_metrics(self, y_true, y_prob, threshold):
        """
        Convenience function to compute all metrics for a given threshold

        :param y_true: true labels
        :param y_prob: predicted probabilities
        :param threshold: threshold for binary classification

        :return: dictionary of metric_name: metric_value
        """
        metrics = {}
        for metric_name, metric_func in self.metric_functions.items():
                metrics[metric_name] = metric_func(y_true, y_prob, threshold)
        return metrics

    def _compute_all_thresholds_per_group(self, y_prob, unique_groups, A):
        """
        Convenience function to compute all possible threshold combinations
        
        :param y_prob: predicted probabilities
        :param unique_groups: unique group labels
        :param A: group labels

        :return: all possible threshold combinations
        """
        group_thresholds = {}
        for group in unique_groups:
            mask_group = A == group
            group_thresholds[group] = np.unique(y_prob[mask_group])
        return np.array(np.meshgrid(*group_thresholds.values())).T.reshape(-1, len(unique_groups))
    
    def _objective_function(self, 
                            tau,  # threshold_combination 
                            Y,  # y_true
                            Y_hat,  # y_prob
                            groups,  # A
                            unique_groups,
                            metric_functions,  # g_i
                            lambda_val,
                            gamma_map = None,
                            diff_strategy = 'euclidean'):  # lambda
        
        if gamma_map is None:
            gamma_map = {metric_name: 1.0 / len(metric_functions) for metric_name in metric_functions.keys()}

        combined_Y_hat_adjusted = []
        combined_Y_true = []
        objective = 0
        group_metrics = {}
        total_size = 0
        global_performance_weighted = 0
        max_epsilon_violation = False
        temp_epsilons = {} 

        for i, group in enumerate(unique_groups):
            mask_group = groups == group
            group_Y = Y[mask_group]
            group_Y_hat = Y_hat[mask_group]
            tau_group = tau[i]
            
            group_metrics[group] = self._compute_metrics(group_Y, group_Y_hat, tau_group)
            
            Y_hat_adjusted = group_Y_hat > tau_group
            combined_Y_hat_adjusted.extend(Y_hat_adjusted)
            combined_Y_true.extend(group_Y)

            # NOTE: This by default is weighting the global performance for each group by the group size
            # BUT there's a world in which we might instead weight evenly
            group_global_performance = self.global_metric(group_Y, group_Y_hat, tau_group)
            group_size = len(group_Y)
            global_performance_weighted += group_global_performance * group_size
            total_size += group_size

        global_performance_weighted /= total_size
        
        for i, group in enumerate(unique_groups):
            temp_epsilons[group] = {}
            group_metric_vector = np.array([group_metrics[group][metric_name] for metric_name in metric_functions.keys()])

            for other_group in unique_groups:
                if group != other_group:
                    if diff_strategy == 'euclidean':
                        other_group_metric_vector = np.array([group_metrics[other_group][metric_name] for metric_name in metric_functions.keys()])
                        distances = cdist(group_metric_vector.reshape(1, -1), other_group_metric_vector.reshape(1, -1), metric='euclidean')
                        objective += np.sum(distances)
                        epsilon_diff = np.abs(group_metric_vector - other_group_metric_vector)
                    else:
                        epsilon_diff = []
                        for metric_name in metric_functions.keys():
                            diff = group_metrics[group][metric_name] - group_metrics[other_group][metric_name]
                            objective += gamma_map[metric_name] * (diff ** 2)
                            epsilon_diff.append(abs(diff))

                    temp_epsilons[group][other_group] = epsilon_diff
                    if any(diff > self.max_epsilon for diff in epsilon_diff):
                        max_epsilon_violation = True
        
        A_choose_2 = len(unique_groups) * (len(unique_groups) - 1) / 2
        objective = (1 - lambda_val) * (objective / A_choose_2) + lambda_val * (1 - global_performance_weighted)

        if self.resource_constraint and sum(combined_Y_hat_adjusted) > self.resource_constraint:
            return np.inf, temp_epsilons, max_epsilon_violation

        return objective, temp_epsilons, max_epsilon_violation


class ApproxThresholdBrute(ApproxThreshold):
    def __init__(self, metric_functions, lambda_=0.5, global_metric=f1, max_epsilon=1.0, max_error=0, max_total_combinations=1):
        super().__init__(metric_functions, lambda_=lambda_, global_metric=global_metric, max_epsilon=max_epsilon)
        self.metric_functions = metric_functions  # list of metric functions
        self.lambda_ = lambda_
        self.thresholds_ = None
        self.epsilons_ = None
        self.global_metric = global_metric
        self.max_epsilon = max_epsilon
        self.best_objective_value = None
        self.y_prob = None

    def _find_intersection(self, 
                           y_true,
                           y_prob, 
                           A, 
                           unique_groups, 
                           metrics_functions, 
                           lambda_=0.5, 
                           resource_constraint=None):
        """
        Brute force method to find the best threshold combination.

        :param y_true: true labels
        :param y_prob: predicted probabilities
        :param A: group labels vector
        :param unique_groups: unique group labels
        :param metrics_functions: dictionary of metric_name: metric_function
        :param lambda_: global utility parameter
        :param resource_constraint: maximum number of positive predictions allowed

        :return: best thresholds, best epsilons
        """
        best_objective_value = float('inf')
        best_thresholds = {}
        best_epsilons = {}
        all_threshold_combinations = self._compute_all_thresholds_per_group(y_prob, unique_groups, A)

        for group in unique_groups:
            self.group_metrics[group] = {}

        for idx, threshold_combination in enumerate(tqdm(all_threshold_combinations, desc="Processing combinations")):
            objective, temp_epsilons, max_epsilon_violation = self._objective_function(threshold_combination, 
                                                                                       y_true, 
                                                                                       y_prob, 
                                                                                       A, 
                                                                                       unique_groups, 
                                                                                       metrics_functions, 
                                                                                       lambda_)

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


class ApproxThresholdNet(ApproxThreshold):
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
        """
        Convenience function to evaluate a threshold combination in parallel
        
        :param threshold_combination:
        :param y_true:
        :param y_prob:
        :param A:
        :param unique_groups:
        :param metrics_functions:
        :param lambda_:
        """
        objective, temp_epsilons, max_epsilon_violation = self._objective_function(threshold_combination, 
                                                                                       y_true, 
                                                                                       y_prob, 
                                                                                       A, 
                                                                                       unique_groups, 
                                                                                       metrics_functions, 
                                                                                       lambda_)

        return objective, threshold_combination, temp_epsilons, max_epsilon_violation

    def _find_intersection(self, y_true, y_prob, A, unique_groups, metrics_functions, lambda_=0.5):
        """
        Use an epsilon net to find the best threshold combination.
        
        :param y_true:
        :param y_prob:
        :param A:
        :param unique_groups:
        :param metrics_functions:
        :param lambda_:
        """
        binom_metric_funcs = len(metrics_functions) * (len(metrics_functions) - 1) / 2

        # NOTE: this is often theoretically 1/n, but we conservatively use 1/log(n)
        # lipschitz_constant_g_i = 2 * (1 / np.log(len(y_true)))
        lipschitz_constant_g_i = 0.25 

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

        combos = list(product(threshold_points, repeat=len(unique_groups)))
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            print(executor._mp_context)
            print('Enters pool')
            future_to_combination = {executor.submit(self.evaluate_combination, tc, y_true, y_prob, A, unique_groups, metrics_functions, lambda_): tc for tc in combos}
            print('Submits futures')
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
    