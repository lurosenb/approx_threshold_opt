import torch
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def tpr_torch(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).float()
    TP = torch.sum((y_pred == 1) & (y_true == 1)).float()
    P = torch.sum(y_true == 1).float()
    return TP / P if P > 0 else torch.tensor(0.0)

def fpr_torch(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).float()
    FP = torch.sum((y_pred == 1) & (y_true == 0)).float()
    N = torch.sum(y_true == 0).float()
    return FP / N if N > 0 else torch.tensor(0.0)

def precision_torch(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).float()
    TP = torch.sum((y_pred == 1) & (y_true == 1)).float()
    FP = torch.sum((y_pred == 1) & (y_true == 0)).float()
    return TP / (TP + FP) if (TP + FP) > 0 else torch.tensor(0.0)

def accuracy_torch(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).float()
    TP = torch.sum((y_pred == 1) & (y_true == 1)).float()
    TN = torch.sum((y_pred == 0) & (y_true == 0)).float()
    P = torch.sum(y_true == 1).float()
    N = torch.sum(y_true == 0).float()
    return (TP + TN) / (P + N) if (P + N) > 0 else torch.tensor(0.0)

def sigmoid(x, beta=1.0):
    return 1 / (1 + torch.exp(-beta * x))

def precision_sigmoid(y_true, y_prob, threshold=0.5, beta=10):
    y_prob_sigmoid = sigmoid(y_prob - threshold, beta)
    TP_soft = torch.sum(y_prob_sigmoid * y_true)
    FP_soft = torch.sum(y_prob_sigmoid * (1 - y_true))
    denominator = TP_soft + FP_soft
    return TP_soft / denominator if denominator > 0 else torch.tensor(0.0)

def tpr_sigmoid(y_true, y_prob, threshold=0.5, beta=10):
    y_prob_sigmoid = sigmoid(y_prob - threshold, beta)
    TP = torch.sum(y_prob_sigmoid * y_true)
    P = torch.sum(y_true)
    return TP / P if P > 0 else torch.tensor(0.0)

def fpr_sigmoid(y_true, y_prob, threshold=0.5, beta=10):
    y_prob_sigmoid = sigmoid(y_prob - threshold, beta)
    FP = torch.sum(y_prob_sigmoid * (1 - y_true))
    N = torch.sum(1 - y_true)
    return FP / N if N > 0 else torch.tensor(0.0)

def accuracy_sigmoid(y_true, y_prob, threshold=0.5, beta=10):
    y_prob_sigmoid = sigmoid(y_prob - threshold, beta)
    TP = torch.sum(y_prob_sigmoid * y_true)
    TN = torch.sum((1 - y_prob_sigmoid) * (1 - y_true))
    total = y_true.numel()  # Total number of elements
    return (TP + TN) / total if total > 0 else torch.tensor(0.0)

class ApproxThresholdPytorch(BaseEstimator, ClassifierMixin):
    def __init__(self, metric_functions, lambda_=0.5, global_metric=accuracy_torch, alpha=0.01, lr=0.01, num_iters=1000, patience=20, min_delta=0.001, gamma=0.1, max_error=0, max_total_combinations=1):
        self.metric_functions = metric_functions 
        self.lambda_ = lambda_
        self.thresholds_ = None
        self.global_metric_func = global_metric
        self.global_metric = global_metric
        self.alpha = alpha
        self.gamma = gamma
        self.lr = lr
        self.num_iters = num_iters
        self.patience = patience
        self.min_delta = min_delta
        self.group_metrics = {}
        self.max_epsilon = 0

    def fit(self, y_prob, y, A):
        y_true_tensor = torch.tensor(y, dtype=torch.float32)
        y_prob_tensor = torch.tensor(y_prob, dtype=torch.float32)
        A_tensor = torch.tensor(A, dtype=torch.int64)

        unique_groups = np.unique(A)
        self.thresholds_ = self._find_intersection_gd(y_true_tensor, y_prob_tensor, A_tensor, unique_groups, self.metric_functions)

        return self

    def predict(self, y_prob, A):
        y_prob_tensor = torch.tensor(y_prob, dtype=torch.float32)

        adjusted_labels = np.zeros(A.shape)
        unique_groups = np.unique(A)
        for group in unique_groups:
            mask = A == group
            threshold = self.thresholds_[group]
            adjusted_labels[mask] = np.where(y_prob_tensor[mask].numpy() > threshold, 1, 0)
        
        return adjusted_labels

    def _find_intersection_gd(self, y_true, y_prob, A, unique_groups, metrics_functions, resource_constraint=None, num_initializations=5):
        best_overall_loss = float('inf')
        best_overall_thresholds = None
        best_epsilons = {}  # Store the best epsilons here

        for initialization in range(num_initializations):
            thresholds = torch.rand(len(unique_groups), requires_grad=True)

            optimizer = optim.Adam([thresholds], lr=self.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=self.gamma)

            best_loss = float('inf')
            best_thresholds = None
            best_iteration_epsilons = {}  # Temporary storage for the best epsilons of this initialization
            epochs_no_improve = 0

            for epoch in range(self.num_iters):
                optimizer.zero_grad()

                objective, iteration_epsilons = self._compute_objective_gd_with_epsilons(y_true, y_prob, A, unique_groups, thresholds, metrics_functions, self.lambda_, resource_constraint)
                regularization = self.alpha * torch.sum(thresholds**2)
                objective += regularization

                loss = objective
                loss.backward()
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    thresholds.clamp_(0, 1)

                if best_loss - loss.item() > self.min_delta:
                    best_loss = loss.item()
                    best_thresholds = thresholds.clone()
                    best_iteration_epsilons = iteration_epsilons  # Update the best epsilons for this initialization
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve == self.patience:
                    print(f"Early stopping triggered at initialization {initialization}, epoch {epoch}")
                    break

                if epoch % 10 == 0:
                    print(f'Initialization {initialization}, Epoch {epoch}, Loss: {loss.item()}')

            if best_loss < best_overall_loss:
                best_overall_loss = best_loss
                best_overall_thresholds = best_thresholds
                best_epsilons = best_iteration_epsilons  # Update the global best epsilons

        print(f'Best objective value: {best_overall_loss}')
        print(f'Best thresholds: {best_overall_thresholds}')
        print(f'Epsilon differences: {best_epsilons}')
        self.best_objective_value = best_overall_loss
        self.epsilons_ = best_epsilons  # Store the best epsilons in the class

        optimized_thresholds = best_overall_thresholds.detach().numpy() if best_overall_thresholds is not None else thresholds.detach().numpy()
        return dict(zip(unique_groups, optimized_thresholds))
    
    def _compute_objective_gd_with_epsilons(self, y_true, y_prob, A, unique_groups, thresholds, metrics_functions, lambda_, resource_constraint, distance_type="squared"):
        objective = 0.0
        epsilons = {}  # init a dictionary to store epsilon differences between groups

        unique_groups_indices = {group: idx for idx, group in enumerate(unique_groups)}

        metrics_per_group = {group: [] for group in unique_groups}
        total_size = y_true.shape[0]
        weighted_global_metric = 0.0

        for group in unique_groups:
            group_idx = unique_groups_indices[group]
            y_prob_group = y_prob[A == group]
            y_true_group = y_true[A == group]
            group_size = y_true_group.shape[0]

            group_metrics = []
            for metric_name, metric_func in metrics_functions.items():
                metric_value = metric_func(y_true_group, y_prob_group, thresholds[group_idx])
                group_metrics.append(metric_value)

            metrics_per_group[group] = torch.stack(group_metrics)

            group_global_metric = self.global_metric_func(y_true_group, y_prob_group, thresholds[group_idx])
            weighted_global_metric += group_global_metric * (group_size / total_size)

        for i, group in enumerate(unique_groups):
            epsilons[group] = {}
            for j, other_group in enumerate(unique_groups):
                if i != j:
                    differences = metrics_per_group[group] - metrics_per_group[other_group]
                    # calculate epsilon differences for each metric
                    epsilon_diff = torch.abs(differences)
                    if self.max_epsilon < epsilon_diff.max():
                        self.max_epsilon = epsilon_diff.max()
                    epsilons[group][other_group] = epsilon_diff.numpy()  # Store the epsilon differences

                    if distance_type == "euclidean":
                        distance = torch.sqrt(torch.sum(differences ** 2))
                    elif distance_type == "squared":
                        distance = torch.sum(differences ** 2)
                    else:
                        raise ValueError("Invalid distance type. Choose 'euclidean' or 'squared'")
                    objective += distance

        objective += lambda_ * (1 - weighted_global_metric)
        return objective, epsilons

    
    def _store_metrics_for_epoch(self, y_true, y_prob, A, unique_groups, thresholds, metrics_functions):
        if not hasattr(self, 'group_metrics') or not self.group_metrics:
            self.group_metrics = {group: {} for group in unique_groups}

        unique_groups_indices = {group: idx for idx, group in enumerate(unique_groups)}

        for group in unique_groups:
            group_idx = unique_groups_indices[group]
            y_prob_group = y_prob[A == group]
            y_true_group = y_true[A == group]

            group_metrics = {}
            for metric_name, metric_func in metrics_functions.items():
                metric_value = metric_func(y_true_group, y_prob_group, thresholds[group_idx])
                group_metrics[metric_name] = metric_value.item() 

            self.group_metrics[group][group_idx] = group_metrics
    
    def plot_matplotlib(self, metric_keys=None):
        if metric_keys is None:
            metric_keys = list(self.metric_functions.keys())

        num_metrics = len(metric_keys)
        print(f"Plotting {num_metrics} metrics: {metric_keys}")

        if num_metrics == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            for group, metrics_dict in self.group_metrics.items():
                x_data = [metrics[metric_keys[0]] for metrics in metrics_dict.values()]
                y_data = [metrics[metric_keys[1]] for metrics in metrics_dict.values()]
                z_data = [metrics[metric_keys[2]] for metrics in metrics_dict.values()]

                ax.scatter(x_data, y_data, z_data, label=f'Group {group}')

            ax.set_xlabel(metric_keys[0])
            ax.set_ylabel(metric_keys[1])
            ax.set_zlabel(metric_keys[2])
            ax.set_title('3D Metric Plot')
            ax.legend()
            plt.show()

        elif num_metrics == 2:
            fig, ax = plt.subplots(figsize=(8, 6))

            for group, metrics_dict in self.group_metrics.items():
                x_data = [metrics[metric_keys[0]] for metrics in metrics_dict.values()]
                y_data = [metrics[metric_keys[1]] for metrics in metrics_dict.values()]

                ax.scatter(x_data, y_data, label=f'Group {group}')

            ax.set_xlabel(metric_keys[0])
            ax.set_ylabel(metric_keys[1])
            ax.set_title('2D Metric Plot')
            ax.legend()
            plt.show()

        else:
            raise ValueError("The number of metrics must be 2 or 3 for plotting.")

    def plot_plotly(self, metric_keys=None):
        if metric_keys is None:
            metric_keys = list(self.metric_functions.keys())

        num_metrics = len(metric_keys)

        if num_metrics == 3:
            fig = go.Figure()

            for group, metrics in self.group_metrics.items():
                x_data = [epoch_metrics[metric_keys[0]] for epoch_metrics in metrics.values()]
                y_data = [epoch_metrics[metric_keys[1]] for epoch_metrics in metrics.values()]
                z_data = [epoch_metrics[metric_keys[2]] for epoch_metrics in metrics.values()]

                fig.add_trace(go.Scatter3d(x=x_data, y=y_data, z=z_data, mode='lines', name=f'Group {group} Line'))

                best_metrics = max(metrics.values(), key=lambda x: sum(x.values()))
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

            for group, metrics in self.group_metrics.items():
                x_data = [epoch_metrics[metric_keys[0]] for epoch_metrics in metrics.values()]
                y_data = [epoch_metrics[metric_keys[1]] for epoch_metrics in metrics.values()]

                fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=f'Group {group} Line'))

                best_metrics = max(metrics.values(), key=lambda x: sum(x.values()))  # Example heuristic
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

