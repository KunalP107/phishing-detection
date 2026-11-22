"""Shared evaluation script used for both models (owner: Kunal)."""
import time

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from statsmodels.stats.contingency_tables import mcnemar


def classification_metrics(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'accuracy': accuracy_score(y, pred), 'precision': precision_score(y, pred, zero_division=0),
            'recall': recall_score(y, pred, zero_division=0), 'f1': f1_score(y, pred, zero_division=0),
            'roc_auc': roc_auc_score(y, score), 'false_positive_rate': fp / (fp + tn) if (fp + tn) else 0.0,
            'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)}


def mcnemar_test(y, pred_a, pred_b):
    """McNemar's test on paired predictions of two models (Dietterich, 1998)."""
    a_ok, b_ok = pred_a == y, pred_b == y
    table = [[int(np.sum(a_ok & b_ok)), int(np.sum(a_ok & ~b_ok))],
             [int(np.sum(~a_ok & b_ok)), int(np.sum(~a_ok & ~b_ok))]]
    discordant = table[0][1] + table[1][0]
    result = mcnemar(table, exact=discordant < 25, correction=True)
    return {'both_correct': table[0][0], 'a_right_b_wrong': table[0][1], 'a_wrong_b_right': table[1][0],
            'both_wrong': table[1][1], 'statistic': float(result.statistic), 'p_value': float(result.pvalue),
            'exact_test': discordant < 25}


def per_email_latency(predict_one, texts, warmup=10):
    """Deployment simulation: score emails one at a time, like a mail gateway, and time each one (ms)."""
    for t in texts[:warmup]:
        predict_one(t)
    times = []
    for t in texts:
        start = time.perf_counter()
        predict_one(t)
        times.append((time.perf_counter() - start) * 1000)
    return {'median_ms': float(np.median(times)), 'p95_ms': float(np.percentile(times, 95))}
