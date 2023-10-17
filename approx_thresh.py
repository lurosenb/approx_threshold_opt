import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from scipy.spatial.distance import cdist

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score

class ApproxThreshold(BaseEstimator, ClassifierMixin):
    """
    ApproxThreshold is a classifier that adjusts the decision threshold of a base classifier
    to equalize a number of fairness constraints while optimizing for a global metric.

    Parameters
    ----------
    base_model : object
        A base classifier that implements the scikit-learn estimator interface.
        Must have a predict_proba method.
    
    lambda_ : float, default=0.5
        A parameter that controls the trade-off between a focus on fairness and on global metric.
    """
    def __init__(self, base_model, lambda_=0.5, flag='global_objective'):
        self.base_model = base_model
        self.lambda_ = lambda_
        self.thresholds_ = None
        self.flag = flag

    def fit(self, X, y, A):
        """
        This method fits the model to the data:
        - First, it fits the base model to the data.
        - Then, it computes the ROC and precision curves for each class.
        - Finally, it finds the intersection of the two curves that minimizes the distance between them,
        controlled by the lambda_ parameter.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.
        
        A : array-like of shape (n_samples,)
            The protected attribute values.

        Returns
        -------
        self : object
            Returns self.
        """

        self.base_model.fit(X, y)
        y_prob = self.base_model.predict_proba(X)[:, 1]
        mask_class1 = A == 1
        mask_class2 = A == 2
        self.fpr1_, self.tpr1_, self.precision1_, self.thresholds1_, self.roc_auc1_ = self._compute_roc_and_precision(y[mask_class1], y_prob[mask_class1])
        self.fpr2_, self.tpr2_, self.precision2_, self.thresholds2_, self.roc_auc2_ = self._compute_roc_and_precision(y[mask_class2], y_prob[mask_class2])
        
        # Check whether to use brute force or cdist
        if 'global_objective' in self.flag:
            self.thresholds_ = self._find_intersection(self.fpr1_, 
                                                    self.tpr1_, 
                                                    self.precision1_, 
                                                    self.thresholds1_, 
                                                    self.fpr2_, 
                                                    self.tpr2_, 
                                                    self.precision2_, 
                                                    self.thresholds2_, 
                                                    y, y_prob, 
                                                    mask_class1, 
                                                    mask_class2, 
                                                    lambda_=self.lambda_)
        else:
            self.thresholds_ = self._find_intersection_just_thresholds(self.fpr1_,
                                                                    self.tpr1_,
                                                                    self.precision1_,
                                                                    self.thresholds1_,
                                                                    self.fpr2_,
                                                                    self.tpr2_,
                                                                    self.precision2_,
                                                                    self.thresholds2_)
        
        return self

    def predict(self, X, A):
        """
        This method predicts the class labels for the provided data.
        First, it computes the predicted probabilities for each class.
        Then, it adjusts the decision threshold for each class according to the thresholds found in the fit method.
        Finally, it returns the adjusted class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        A : array-like of shape (n_samples,)
            The protected attribute values.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted class labels.
        """

        y_prob = self.base_model.predict_proba(X)[:, 1]
        mask_class1 = A == 1
        mask_class2 = A == 2
        adjusted_labels1 = np.where(y_prob[mask_class1] > self.thresholds_[0], 1, 0)
        adjusted_labels2 = np.where(y_prob[mask_class2] > self.thresholds_[1], 1, 0)
        adjusted_labels = np.zeros(A.shape)
        adjusted_labels[mask_class1] = adjusted_labels1
        adjusted_labels[mask_class2] = adjusted_labels2
        
        return adjusted_labels
        
    def _compute_roc_and_precision(self, y_true, y_prob):
        """
        This method computes the ROC and precision curves for a given class.
        First it computes the ROC and precision curves.
        Then, it interpolates the precision curve at the points of the ROC curve.
        Finally, it computes the area under the ROC curve.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The target values.
        
        y_prob : array-like of shape (n_samples,)

        Returns
        -------
        fpr : ndarray of shape (n_thresholds + 1,)
            False Positive Rates.
        
        tpr : ndarray of shape (n_thresholds + 1,)
            True Positive Rates.

        precision_interp : ndarray of shape (n_thresholds + 1,)
            Precision values interpolated at the points of the ROC curve.

        thresholds : ndarray of shape (n_thresholds,)
            Threshold values.

        roc_auc : float
            Area under the ROC curve.
        """

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        precision_interp = np.interp(tpr, recall[::-1], precision[::-1])
        return fpr, tpr, precision_interp, thresholds, auc(fpr, tpr)

    def _find_intersection_just_thresholds(self, fpr1, tpr1, precision1, thresholds1, fpr2, tpr2, precision2, thresholds2):
        """
        This method finds the intersection of two curves by minimizing the distance between them.
        It returns the thresholds that correspond to the intersection point, as well as the distances between the curves at that point.
        First, it computes the distances between all points of the two curves.
        Then, it finds the indices of the points that minimize the distance between the curves.
        Finally, it returns the thresholds that correspond to those indices, as well as the distances between the curves at that point.

        Parameters
        ----------
        fpr1 : ndarray of shape (n_thresholds + 1,)
            False Positive Rates for class 1.

        tpr1 : ndarray of shape (n_thresholds + 1,)
            True Positive Rates for class 1.

        precision1 : ndarray of shape (n_thresholds + 1,)
            Precision values for class 1.

        thresholds1 : ndarray of shape (n_thresholds,)
            Threshold values for class 1.

        fpr2 : ndarray of shape (n_thresholds + 1,)
            False Positive Rates for class 2.

        tpr2 : ndarray of shape (n_thresholds + 1,)
            True Positive Rates for class 2.
        
        precision2 : ndarray of shape (n_thresholds + 1,)
            Precision values for class 2.
        
        thresholds2 : ndarray of shape (n_thresholds,)
            Threshold values for class 2.

        Returns
        -------
        threshold1 : float
            Threshold for class 1 that corresponds to the intersection point.

        threshold2 : float
            Threshold for class 2 that corresponds to the intersection point.

        epsilon_fpr : float
            Distance between the two curves at the intersection point.
        
        epsilon_tpr : float
            Distance between the two curves at the intersection point.

        epsilon_precision : float
            Distance between the two curves at the intersection point.
        """

        points1 = np.vstack([fpr1, tpr1, precision1]).T
        points2 = np.vstack([fpr2, tpr2, precision2]).T
        distances = cdist(points1, points2, metric='euclidean')
        i, j = np.unravel_index(distances.argmin(), distances.shape)
        epsilon_fpr = np.abs(fpr1[i] - fpr2[j])
        epsilon_tpr = np.abs(tpr1[i] - tpr2[j])
        epsilon_precision = np.abs(precision1[i] - precision2[j])
        return thresholds1[i], thresholds2[j], epsilon_fpr, epsilon_tpr, epsilon_precision

    def _find_intersection(self, fpr1, tpr1, precision1, thresholds1, fpr2, tpr2, precision2, thresholds2, y_true, y_prob, mask_class1, mask_class2, lambda_=0.5):
        """
        This method finds the intersection of two curves by minimizing the distance between them.
        It returns the thresholds that correspond to the intersection point, as well as the distances between the curves at that point.
        First, it computes the distances between all points of the two curves.
        Then, it finds the indices of the points that minimize the distance between the curves.
        It checks the objective function at those points - lambda_ is used to control the trade-off between fairness and global metric.
        Finally, it returns the thresholds that correspond to those indices, as well as the distances between the curves at that point.

        Parameters
        ----------
        fpr1 : ndarray of shape (n_thresholds + 1,)
            False Positive Rates for class 1.

        tpr1 : ndarray of shape (n_thresholds + 1,)
            True Positive Rates for class 1.

        precision1 : ndarray of shape (n_thresholds + 1,)
            Precision values for class 1.

        thresholds1 : ndarray of shape (n_thresholds,)
            Threshold values for class 1.

        fpr2 : ndarray of shape (n_thresholds + 1,)
            False Positive Rates for class 2.

        tpr2 : ndarray of shape (n_thresholds + 1,)
            True Positive Rates for class 2.
        
        precision2 : ndarray of shape (n_thresholds + 1,)
            Precision values for class 2.
        
        thresholds2 : ndarray of shape (n_thresholds,)
            Threshold values for class 2.

        Returns
        -------
        threshold1 : float
            Threshold for class 1 that corresponds to the intersection point.

        threshold2 : float
            Threshold for class 2 that corresponds to the intersection point.

        epsilon_fpr : float
            Distance between the two curves at the intersection point.
        
        epsilon_tpr : float
            Distance between the two curves at the intersection point.

        epsilon_precision : float
            Distance between the two curves at the intersection point.
        """
        points1 = np.vstack([fpr1, tpr1, precision1]).T
        points2 = np.vstack([fpr2, tpr2, precision2]).T
        distances = cdist(points1, points2, metric='euclidean')
        best_objective_value = float('inf')
        best_indices = (0, 0)

        for i in range(len(thresholds1)):
            for j in range(len(thresholds2)):
                adjusted_labels1 = y_prob[mask_class1] > thresholds1[i]
                adjusted_labels2 = y_prob[mask_class2] > thresholds2[j]
                combined_preds = np.concatenate([adjusted_labels1, adjusted_labels2])
                combined_true = np.concatenate([y_true[mask_class1], y_true[mask_class2]])
                acc = accuracy_score(combined_true, combined_preds)
                objective = distances[i, j] + lambda_ * (1 - acc)
                if objective < best_objective_value:
                    best_objective_value = objective
                    best_indices = (i, j)
                    
        i, j = best_indices
        epsilon_fpr = np.abs(fpr1[i] - fpr2[j])
        epsilon_tpr = np.abs(tpr1[i] - tpr2[j])
        epsilon_precision = np.abs(precision1[i] - precision2[j])
        return thresholds1[i], thresholds2[j], epsilon_fpr, epsilon_tpr, epsilon_precision

    def plot_matplotlib(self):
        """
        This method plots the ROC and precision curves for each class.
        It also plots the intersection point of the two curves.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(self.fpr1_, self.tpr1_, self.precision1_, label=f'Class 1 ROC (area = {self.roc_auc1_:.2f})', color='blue')
        ax.plot(self.fpr2_, self.tpr2_, self.precision2_, label=f'Class 2 ROC (area = {self.roc_auc2_:.2f})', color='orange')
        
        ax.scatter(self.fpr1_[np.argwhere(self.thresholds1_ == self.thresholds_[0])[0]],
                self.tpr1_[np.argwhere(self.thresholds1_ == self.thresholds_[0])[0]],
                self.precision1_[np.argwhere(self.thresholds1_ == self.thresholds_[0])[0]], c='blue', s=100, marker='o')

        ax.scatter(self.fpr2_[np.argwhere(self.thresholds2_ == self.thresholds_[1])[0]],
                self.tpr2_[np.argwhere(self.thresholds2_ == self.thresholds_[1])[0]],
                self.precision2_[np.argwhere(self.thresholds2_ == self.thresholds_[1])[0]], c='orange', s=100, marker='o')

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_zlabel('Precision')
        ax.set_title('3D ROC-Precision curve')
        ax.legend()
        plt.show()

    def plot_plotly(self):
        """
        This method plots the ROC and precision curves for each class.
        It also plots the intersection point of the two curves.
        It uses the plotly library.
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(x=self.fpr1_, y=self.tpr1_, z=self.precision1_, 
                                mode='lines+markers', 
                                name=f'Class 1 ROC (area = {self.roc_auc1_:.2f})',
                                line=dict(color='blue'),
                                marker=dict(size=[10 if th == self.thresholds_[0] else 2 for th in self.thresholds1_])))

        fig.add_trace(go.Scatter3d(x=self.fpr2_, y=self.tpr2_, z=self.precision2_, 
                                mode='lines+markers', 
                                name=f'Class 2 ROC (area = {self.roc_auc2_:.2f})',
                                line=dict(color='orange'),
                                marker=dict(size=[10 if th == self.thresholds_[1] else 2 for th in self.thresholds2_])))

        midpoint = [(self.fpr1_[np.argwhere(self.thresholds1_ == self.thresholds_[0])[0][0]] + \
                     self.fpr2_[np.argwhere(self.thresholds2_ == self.thresholds_[1])[0][0]]) / 2,
                    (self.tpr1_[np.argwhere(self.thresholds1_ == self.thresholds_[0])[0][0]] + \
                     self.tpr2_[np.argwhere(self.thresholds2_ == self.thresholds_[1])[0][0]]) / 2,
                    (self.precision1_[np.argwhere(self.thresholds1_ == self.thresholds_[0])[0][0]] + \
                     self.precision2_[np.argwhere(self.thresholds2_ == self.thresholds_[1])[0][0]]) / 2]

        fig.add_trace(go.Scatter3d(x=[self.fpr1_[np.argwhere(self.thresholds1_ == self.thresholds_[0])[0][0]], \
                                      self.fpr2_[np.argwhere(self.thresholds2_ == self.thresholds_[1])[0][0]]],
                                y=[self.tpr1_[np.argwhere(self.thresholds1_ == self.thresholds_[0])[0][0]], \
                                   self.tpr2_[np.argwhere(self.thresholds2_ == self.thresholds_[1])[0][0]]],
                                z=[self.precision1_[np.argwhere(self.thresholds1_ == self.thresholds_[0])[0][0]], \
                                   self.precision2_[np.argwhere(self.thresholds2_ == self.thresholds_[1])[0][0]]],
                                mode='lines',
                                line=dict(color='red'),
                                showlegend=False))

        annotation_text = f'ΔFPR={self.thresholds_[2]:.2f}<br>ΔTPR={self.thresholds_[3]:.2f}<br>ΔPrecision={self.thresholds_[4]:.2f}'

        fig.add_trace(go.Scatter3d(x=[midpoint[0]], y=[midpoint[1]], z=[midpoint[2]],
                                mode='text',
                                text=[annotation_text],
                                textposition="bottom center",
                                showlegend=False))

        fig.update_layout(scene=dict(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                zaxis_title='Precision'),
            margin=dict(r=20, b=10, l=10, t=10))

        fig.show()
    
    def plot_2d_roc(self):
        """
        This method plots the ROC curve for each class.
        It also plots the diagonal line, which corresponds to a random classifier.
        """
        plt.figure(figsize=(8, 6))

        plt.plot(self.fpr1_, self.tpr1_, color='blue', label=f'Class 1 ROC (area = {self.roc_auc1_:.2f})')
        plt.plot(self.fpr2_, self.tpr2_, color='orange', label=f'Class 2 ROC (area = {self.roc_auc2_:.2f})')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('2D ROC Curve')
        plt.legend(loc='lower right')

        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.show()

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

        fig.update_layout(barmode='group', title='Performance Comparison: Original vs. Adjusted Thresholds')
        fig.show()
