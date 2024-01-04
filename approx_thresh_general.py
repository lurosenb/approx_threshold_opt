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

def tpr(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    P = np.sum(y_true == 1)
    return TP / P if P > 0 else 0

def fpr(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    FP = np.sum((y_pred == 1) & (y_true == 0))
    N = np.sum(y_true == 0)
    return FP / N if N > 0 else 0

def precision(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    return TP / (TP + FP) if (TP + FP) > 0 else 0

class ApproxThresholdGeneral(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, metric_functions, lambda_=0.5):
        self.base_model = base_model
        self.metric_functions = metric_functions  # list of metric functions i.e. lambda
        self.lambda_ = lambda_
        self.thresholds_ = None

    def fit(self, X, y, A, efficient=True):
        self.base_model.fit(X, y)
        y_prob = self.base_model.predict_proba(X)[:, 1]
        
        self.group_metrics = {}
        self.group_thresholds = {}
        unique_groups = np.unique(A)
        self.thresholds_ = self._find_intersection(y, y_prob, A, unique_groups, self.metric_functions, lambda_=self.lambda_)

        return self

    def predict(self, X, A):
        y_prob = self.base_model.predict_proba(X)[:, 1]
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
        all_threshold_combinations = self._compute_all_thresholds_per_group(y_true, y_prob, unique_groups, A)
        
        for group in unique_groups:
            self.group_metrics[group] = {}
        
        for idx, threshold_combination in enumerate(all_threshold_combinations):
            combined_preds = []
            combined_true = []
            objective = 0
            temp_group_metrics = {}
            total_size = 0
            weighted_acc = 0

            for i, group in enumerate(unique_groups):
                mask_group = A == group
                temp_group_metrics[group] = self._compute_metrics(y_true[mask_group], y_prob[mask_group], threshold_combination[i])
                self.group_metrics[group][idx] = temp_group_metrics[group]
                
                adjusted_labels = y_prob[mask_group] > threshold_combination[i]
                combined_preds.extend(adjusted_labels)
                combined_true.extend(y_true[mask_group])
                
                group_acc = accuracy_score(y_true[mask_group], adjusted_labels)
                group_size = len(y_true[mask_group])
                weighted_acc += group_acc * group_size
                total_size += group_size

            weighted_acc /= total_size

            for i, group in enumerate(unique_groups):
                # here, we need to make a full vector of metrics
                group_metric_vector = np.array([temp_group_metrics[group][metric_name] for metric_name in metrics_functions.keys()])

                for other_group in unique_groups:
                    if group != other_group:
                        other_group_metric_vector = np.array([temp_group_metrics[other_group][metric_name] for metric_name in metrics_functions.keys()])

                        # and then do euclidean distance between that metric and the other group's metric vector
                        distances = cdist(group_metric_vector.reshape(1, -1), other_group_metric_vector.reshape(1, -1), metric='euclidean')
                        objective += np.sum(distances)
            
            objective += lambda_ * (1 - weighted_acc)

            if resource_constraint is not None and sum(combined_preds) > resource_constraint:
                continue

            if objective < best_objective_value:
                best_objective_value = objective
                best_thresholds = dict(zip(unique_groups, threshold_combination))
                self.best_index = idx

        print(f'Best objective value: {best_objective_value}')
        print(f'Best thresholds: {best_thresholds}')
        return best_thresholds


    def plot_matplotlib(self, metric_keys=None):
        """
        This method plots the metric curves for each group based on the A membership.
        If three metrics are provided, it will produce a 3D plot; otherwise, a 2D plot.
        :param metric_keys: Optional list of metric keys to plot. Defaults to all keys.
        """
        if metric_keys is None:
            metric_keys = list(self.metric_functions.keys())

        num_metrics = len(metric_keys)

        if num_metrics == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            for group in self.group_metrics.keys():
                x_data = [metrics[metric_keys[0]] for idx, metrics in self.group_metrics[group].items()]
                y_data = [metrics[metric_keys[1]] for idx, metrics in self.group_metrics[group].items()]
                z_data = [metrics[metric_keys[2]] for idx, metrics in self.group_metrics[group].items()]
                
                ax.plot(x_data, y_data, z_data, label=f'Group {group}')
                
                best_metrics = self.group_metrics[group][self.best_index]
                ax.scatter(best_metrics[metric_keys[0]], best_metrics[metric_keys[1]], best_metrics[metric_keys[2]], s=100, marker='o', label=f'Best Threshold for Group {group}')

            ax.set_xlabel(metric_keys[0])
            ax.set_ylabel(metric_keys[1])
            ax.set_zlabel(metric_keys[2])
            ax.set_title('3D Metric plot')
            ax.legend()
            plt.show()

        elif num_metrics == 2:
            fig, ax = plt.subplots(figsize=(8, 6))

            for group in self.group_metrics.keys():
                x_data = [metrics[metric_keys[0]] for idx, metrics in self.group_metrics[group].items()]
                y_data = [metrics[metric_keys[1]] for idx, metrics in self.group_metrics[group].items()]
                
                ax.plot(x_data, y_data, z_data, label=f'Group {group}')
                
                best_metrics = self.group_metrics[group][self.best_index]
                ax.scatter(best_metrics[metric_keys[0]], best_metrics[metric_keys[1]], s=100, marker='o', label=f'Best Threshold for Group {group}')

            ax.set_xlabel(metric_keys[0])
            ax.set_ylabel(metric_keys[1])
            ax.set_title('2D Metric plot')
            ax.legend()
            plt.show()

        else:
            raise ValueError("The number of metrics must be 2 or 3.")

    def plot_plotly(self, metric_keys=None):
        """
        This method plots the metric curves for each group based on the A membership using Plotly.
        If three metrics are provided, it will produce a 3D plot; otherwise, a 2D plot.
        :param metric_keys: Optional list of metric keys to plot. Defaults to all keys.
        """
        if metric_keys is None:
            metric_keys = list(self.metric_functions.keys())

        num_metrics = len(metric_keys)

        if num_metrics == 3:
            fig = go.Figure()

            for group in self.group_metrics.keys():
                x_data = [metrics[metric_keys[0]] for idx, metrics in self.group_metrics[group].items()]
                y_data = [metrics[metric_keys[1]] for idx, metrics in self.group_metrics[group].items()]
                z_data = [metrics[metric_keys[2]] for idx, metrics in self.group_metrics[group].items()]

                fig.add_trace(go.Scatter3d(x=x_data, y=y_data, z=z_data, mode='lines', name=f'Group {group} Line'))

                best_metrics = self.group_metrics[group][self.best_index]
                fig.add_trace(go.Scatter3d(x=[best_metrics[metric_keys[0]]],
                                        y=[best_metrics[metric_keys[1]]],
                                        z=[best_metrics[metric_keys[2]]],
                                        mode='markers', marker=dict(size=8, symbol='circle'),
                                        name=f'Best Threshold for Group {group}'))

            fig.update_layout(title='3D Metric Plot',
                            scene=dict(xaxis_title=metric_keys[0],
                                        yaxis_title=metric_keys[1],
                                        zaxis_title=metric_keys[2]))
            fig.show()

        elif num_metrics == 2:
            fig = go.Figure()

            for group in self.group_metrics.keys():
                x_data = [metrics[metric_keys[0]] for idx, metrics in self.group_metrics[group].items()]
                y_data = [metrics[metric_keys[1]] for idx, metrics in self.group_metrics[group].items()]

                fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=f'Group {group} Line'))

                best_metrics = self.group_metrics[group][self.best_index]
                fig.add_trace(go.Scatter(x=[best_metrics[metric_keys[0]]],
                                        y=[best_metrics[metric_keys[1]]],
                                        mode='markers', marker=dict(size=8, symbol='circle'),
                                        name=f'Best Threshold for Group {group}'))

            fig.update_layout(title='2D Metric Plot',
                            xaxis_title=metric_keys[0],
                            yaxis_title=metric_keys[1])
            fig.show()

        else:
            raise ValueError("The number of metrics must be 2 or 3.")

    def plot_performance_comparison(self, X_test, y_test, A_test):
        """
        Plot performance metrics comparison between original and adjusted thresholds.
        
        Note:
        Assumes calibration i.e. a default threshold of 0.5 for the original predictions.
        """
        y_prob = self.base_model.predict_proba(X_test)[:, 1]
        original_predictions = np.where(y_prob > 0.5, 1, 0)

        original_accuracy = accuracy_score(y_test, original_predictions)
        original_f1 = f1_score(y_test, original_predictions)
        original_precision = precision_score(y_test, original_predictions)
        original_recall = recall_score(y_test, original_predictions)
        
        adjusted_labels = self.predict(X_test, A_test)

        adjusted_accuracy = accuracy_score(y_test, adjusted_labels)
        adjusted_f1 = f1_score(y_test, adjusted_labels)
        adjusted_precision = precision_score(y_test, adjusted_labels)
        adjusted_recall = recall_score(y_test, adjusted_labels)

        metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        original_values = [original_accuracy, original_f1, original_precision, original_recall]
        adjusted_values = [adjusted_accuracy, adjusted_f1, adjusted_precision, adjusted_recall]

        fig = go.Figure(data=[
            go.Bar(name='Original', x=metrics, y=original_values),
            go.Bar(name='Adjusted', x=metrics, y=adjusted_values)
        ])

        fig.update_layout(barmode='group', title='Performance Comparison: Original vs. Adjusted Thresholds',
                        yaxis=dict(title='Score', range=[0, 1.05]),
                        xaxis=dict(title='Metrics'))
        fig.show()

    def plot_performance_comparison_groups(self, X_test, y_test, A_test):
        y_prob = self.base_model.predict_proba(X_test)[:, 1]
        original_predictions = np.where(y_prob > 0.5, 1, 0)

        unique_groups = np.unique(A_test)
        metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        
        fig = go.Figure()

        for group in unique_groups:
            group_mask = A_test == group

            original_accuracy = accuracy_score(y_test[group_mask], original_predictions[group_mask])
            original_f1 = f1_score(y_test[group_mask], original_predictions[group_mask])
            original_precision = precision_score(y_test[group_mask], original_predictions[group_mask])
            original_recall = recall_score(y_test[group_mask], original_predictions[group_mask])
            original_values = [original_accuracy, original_f1, original_precision, original_recall]

            adjusted_labels = self.predict(X_test[group_mask], A_test[group_mask])
            adjusted_accuracy = accuracy_score(y_test[group_mask], adjusted_labels)
            adjusted_f1 = f1_score(y_test[group_mask], adjusted_labels)
            adjusted_precision = precision_score(y_test[group_mask], adjusted_labels)
            adjusted_recall = recall_score(y_test[group_mask], adjusted_labels)
            adjusted_values = [adjusted_accuracy, adjusted_f1, adjusted_precision, adjusted_recall]

            fig.add_trace(go.Bar(name=f'Original Group {group}', x=metrics, y=original_values, marker_color='blue'))
            fig.add_trace(go.Bar(name=f'Adjusted Group {group}', x=metrics, y=adjusted_values, marker_color='red'))

        fig.update_layout(barmode='group', title='Performance Comparison: Original vs. Adjusted Thresholds by Group',
                        yaxis=dict(title='Score', range=[0, 1.05]),
                        xaxis=dict(title='Metrics'))
        fig.show()

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
    def __init__(self, base_model, metric_functions, lambda_=0.5, max_error=0.01, L_f=None, max_total_combinations=100000):
        super().__init__(base_model, metric_functions, lambda_=lambda_)
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

        for i, group in enumerate(unique_groups):
            mask_group = A == group
            temp_group_metrics[group] = self._compute_metrics(y_true[mask_group], y_prob[mask_group], threshold_combination[i])
            
            adjusted_labels = y_prob[mask_group] > threshold_combination[i]
            combined_preds.extend(adjusted_labels)
            combined_true.extend(y_true[mask_group])
            
            group_acc = accuracy_score(y_true[mask_group], adjusted_labels)
            group_size = len(y_true[mask_group])
            weighted_acc += group_acc * group_size
            total_size += group_size

        weighted_acc /= total_size

        for i, group in enumerate(unique_groups):
            group_metric_vector = np.array([temp_group_metrics[group][metric_name] for metric_name in metrics_functions.keys()])

            for other_group in unique_groups:
                if group != other_group:
                    other_group_metric_vector = np.array([temp_group_metrics[other_group][metric_name] for metric_name in metrics_functions.keys()])
                    distances = cdist(group_metric_vector.reshape(1, -1), other_group_metric_vector.reshape(1, -1), metric='euclidean')
                    objective += np.sum(distances)
        
        objective += lambda_ * (1 - weighted_acc)

        return objective, threshold_combination

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

        print(f'Number of points in the epsilon net: {N}')
        print(f'Adjusted max_error: {max_error}')
        print(f'Number of points in data: {len(y_true)}')

        threshold_points = np.linspace(0, 1, N)
        objectives = {}
        combos = list(product(threshold_points, repeat=len(unique_groups)))
        with ProcessPoolExecutor() as executor:
            future_to_combination = {executor.submit(self.evaluate_combination, tc, y_true, y_prob, A, unique_groups, metrics_functions, lambda_): tc for tc in combos}
            
            for future in tqdm(as_completed(future_to_combination), total=total_combinations, desc="Threshold Combinations"):
                threshold_combination = future_to_combination[future]
                objective_value, _ = future.result()
                objectives[threshold_combination] = objective_value

            executor.shutdown(wait=True)

        # find the best objective value and corresponding thresholds after collecting all results
        best_threshold_combination = min(objectives, key=objectives.get)
        best_objective_value = objectives[best_threshold_combination]
        best_thresholds = dict(zip(unique_groups, best_threshold_combination))

        print(f'Best objective value: {best_objective_value}')
        print(f'Best thresholds: {best_thresholds}')
        return best_thresholds
    