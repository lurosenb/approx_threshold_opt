import numpy as np
import torch

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

def npv(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return TN / (TN + FN) if (TN + FN) > 0 else 0

def accuracy(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return np.sum(y_pred == y_true) / len(y_true)

def f1(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

def selection_rate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    total_predictions = len(y_pred)
    positive_predictions = np.sum(y_pred == 1)
    return positive_predictions / total_predictions if total_predictions > 0 else 0

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

def f1_torch(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).float()
    TP = torch.sum((y_pred == 1) & (y_true == 1)).float()
    FP = torch.sum((y_pred == 1) & (y_true == 0)).float()
    FN = torch.sum((y_pred == 0) & (y_true == 1)).float()
    return 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else torch.tensor(0.0)

def selection_rate_torch(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).float()
    total_predictions = y_pred.size(0)
    positive_predictions = torch.sum(y_pred == 1).float()
    return positive_predictions / total_predictions if total_predictions > 0 else torch.tensor(0.0)

def npv_torch(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).float()
    TN = torch.sum((y_pred == 0) & (y_true == 0)).float()
    FN = torch.sum((y_pred == 0) & (y_true == 1)).float()
    return TN / (TN + FN) if (TN + FN) > 0 else torch.tensor(0.0)
