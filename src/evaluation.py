"""
evaluation.py — Micro and Macro-Averaged Evaluation Metrics

Implements both micro-averaged and macro-averaged Precision, Recall, and F1
for the COLIEE Task 2 evaluation framework.

- Micro-averaged: Global TP / Global Retrieved (official COLIEE ranking metric)
- Macro-averaged: Per-query P/R/F1 averaged across queries (per-query consistency)

The distinction matters: micro-F1 is dominated by high-candidate queries where
aggressive recall is rewarded; macro-F1 weights each query equally, better
reflecting reliability across the full case distribution.
"""

from typing import Callable, Dict, List, Set, Tuple

import numpy as np


def compute_micro_metrics(
    predictions: Dict[str, Dict], 
    threshold_fn: Callable
) -> Tuple[float, float, float]:
    """Compute micro-averaged Precision, Recall, and F1.
    
    Micro-averaging aggregates all true positives, retrievals, and relevant
    items globally before computing the ratios. This is the official COLIEE
    Task 2 ranking metric.
    
    Args:
        predictions: Dict mapping query_id to {"scored": [...], "ground_truth": set}.
        threshold_fn: Function that takes scored candidates and returns predicted filenames.
        
    Returns:
        Tuple of (precision, recall, f1).
    """
    total_tp, total_retrieved, total_relevant = 0, 0, 0
    
    for q_id, data in predictions.items():
        predicted = set(threshold_fn(data["scored"]))
        gt = data["ground_truth"]
        
        total_tp += len(predicted & gt)
        total_retrieved += len(predicted)
        total_relevant += len(gt)
    
    precision = total_tp / total_retrieved if total_retrieved > 0 else 0
    recall = total_tp / total_relevant if total_relevant > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def compute_macro_metrics(
    predictions: Dict[str, Dict], 
    threshold_fn: Callable
) -> Tuple[float, float, float]:
    """Compute macro-averaged Precision, Recall, and F1.
    
    Macro-averaging computes per-query metrics first, then averages across
    all queries. This gives equal weight to each query regardless of its
    number of candidates, better reflecting per-query consistency.
    
    Args:
        predictions: Dict mapping query_id to {"scored": [...], "ground_truth": set}.
        threshold_fn: Function that takes scored candidates and returns predicted filenames.
        
    Returns:
        Tuple of (precision, recall, f1).
    """
    precisions, recalls, f1s = [], [], []
    
    for q_id, data in predictions.items():
        predicted = set(threshold_fn(data["scored"]))
        gt = data["ground_truth"]
        tp = len(predicted & gt)
        
        p = tp / len(predicted) if len(predicted) > 0 else 0
        r = tp / len(gt) if len(gt) > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


def evaluate_all_strategies(
    predictions: Dict[str, Dict],
    thresholds: List[float] = None
) -> Dict[str, Dict[str, float]]:
    """Run a comprehensive evaluation across multiple thresholding strategies.
    
    Evaluates static thresholds, top-1 argmax, and 3-Gate Max-Gap, computing
    both micro and macro metrics for each.
    
    Args:
        predictions: Dict mapping query_id to {"scored": [...], "ground_truth": set}.
        thresholds: List of static threshold values to evaluate.
        
    Returns:
        Dict mapping strategy name to {"micro_p", "micro_r", "micro_f1",
        "macro_p", "macro_r", "macro_f1"}.
    """
    from src.inference import safe_max_gap, static_threshold, top_k_prediction
    
    if thresholds is None:
        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 
                      0.50, 0.55, 0.60, 0.65, 0.70]
    
    results = {}
    
    # Static thresholds
    for t in thresholds:
        fn = lambda scored, _t=t: static_threshold(scored, _t)
        mi_p, mi_r, mi_f1 = compute_micro_metrics(predictions, fn)
        ma_p, ma_r, ma_f1 = compute_macro_metrics(predictions, fn)
        results[f"Static ({t:.2f})"] = {
            "micro_p": mi_p, "micro_r": mi_r, "micro_f1": mi_f1,
            "macro_p": ma_p, "macro_r": ma_r, "macro_f1": ma_f1
        }
    
    # Top-1 Argmax
    fn = lambda scored: top_k_prediction(scored, k=1)
    mi_p, mi_r, mi_f1 = compute_micro_metrics(predictions, fn)
    ma_p, ma_r, ma_f1 = compute_macro_metrics(predictions, fn)
    results["Top-1 (Argmax)"] = {
        "micro_p": mi_p, "micro_r": mi_r, "micro_f1": mi_f1,
        "macro_p": ma_p, "macro_r": ma_r, "macro_f1": ma_f1
    }
    
    # 3-Gate Max-Gap
    mi_p, mi_r, mi_f1 = compute_micro_metrics(predictions, safe_max_gap)
    ma_p, ma_r, ma_f1 = compute_macro_metrics(predictions, safe_max_gap)
    results["3-Gate Max-Gap"] = {
        "micro_p": mi_p, "micro_r": mi_r, "micro_f1": mi_f1,
        "macro_p": ma_p, "macro_r": ma_r, "macro_f1": ma_f1
    }
    
    return results
