#!/usr/bin/env python3
"""
evaluate.py — Standalone evaluation script for the Decoupled Architecture.

Runs comprehensive threshold analysis (static grid + 3-Gate Max-Gap + Top-1)
on pre-computed prediction scores, computing both micro and macro metrics.

Usage:
    python evaluate.py --predictions path/to/val_predictions.json
    python evaluate.py --predictions path/to/val_predictions.json --output results/threshold_analysis.csv

Input format (JSON):
    {
        "query_001": {
            "scored": [["para_12.txt", 0.87], ["para_05.txt", 0.43], ...],
            "ground_truth": ["para_12.txt"]
        },
        ...
    }
"""

import argparse
import json
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.inference import safe_max_gap, static_threshold, top_k_prediction
from src.evaluation import compute_micro_metrics, compute_macro_metrics


def main():
    parser = argparse.ArgumentParser(
        description="COLIEE Task 2 — Threshold Analysis & Evaluation"
    )
    parser.add_argument(
        "--predictions", type=str, required=True,
        help="Path to JSON file with prediction scores and ground truth."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional CSV output path for results table."
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 
                 0.50, 0.55, 0.60, 0.65, 0.70],
        help="Static threshold values to evaluate."
    )
    args = parser.parse_args()
    
    # Load predictions
    with open(args.predictions, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    
    # Convert ground_truth lists to sets for evaluation
    for q_id in predictions:
        if isinstance(predictions[q_id]["ground_truth"], list):
            predictions[q_id]["ground_truth"] = set(predictions[q_id]["ground_truth"])
        # Convert scored back to list of tuples
        predictions[q_id]["scored"] = [
            (item[0], item[1]) for item in predictions[q_id]["scored"]
        ]
    
    print(f"Loaded predictions for {len(predictions)} queries.\n")
    
    # Header
    print("=" * 95)
    print("  COMPREHENSIVE THRESHOLD ANALYSIS")
    print("=" * 95)
    print(f"{'Strategy':<22} | {'Micro-P':>8} {'Micro-R':>8} {'Micro-F1':>9} | "
          f"{'Macro-P':>8} {'Macro-R':>8} {'Macro-F1':>9}")
    print("-" * 95)
    
    results = []
    
    # Static thresholds
    for t in args.thresholds:
        fn = lambda scored, _t=t: static_threshold(scored, _t)
        mi_p, mi_r, mi_f1 = compute_micro_metrics(predictions, fn)
        ma_p, ma_r, ma_f1 = compute_macro_metrics(predictions, fn)
        name = f"Static ({t:.2f})"
        print(f"{name:<22} | {mi_p:>8.4f} {mi_r:>8.4f} {mi_f1:>9.4f} | "
              f"{ma_p:>8.4f} {ma_r:>8.4f} {ma_f1:>9.4f}")
        results.append({
            "strategy": name, "threshold": t,
            "micro_p": mi_p, "micro_r": mi_r, "micro_f1": mi_f1,
            "macro_p": ma_p, "macro_r": ma_r, "macro_f1": ma_f1
        })
    
    print("-" * 95)
    
    # Top-1 Argmax
    fn = lambda scored: top_k_prediction(scored, k=1)
    mi_p, mi_r, mi_f1 = compute_micro_metrics(predictions, fn)
    ma_p, ma_r, ma_f1 = compute_macro_metrics(predictions, fn)
    print(f"{'Top-1 (Argmax)':<22} | {mi_p:>8.4f} {mi_r:>8.4f} {mi_f1:>9.4f} | "
          f"{ma_p:>8.4f} {ma_r:>8.4f} {ma_f1:>9.4f}")
    results.append({
        "strategy": "Top-1 (Argmax)", "threshold": None,
        "micro_p": mi_p, "micro_r": mi_r, "micro_f1": mi_f1,
        "macro_p": ma_p, "macro_r": ma_r, "macro_f1": ma_f1
    })
    
    # 3-Gate Max-Gap
    mi_p, mi_r, mi_f1 = compute_micro_metrics(predictions, safe_max_gap)
    ma_p, ma_r, ma_f1 = compute_macro_metrics(predictions, safe_max_gap)
    print(f"{'3-Gate Max-Gap':<22} | {mi_p:>8.4f} {mi_r:>8.4f} {mi_f1:>9.4f} | "
          f"{ma_p:>8.4f} {ma_r:>8.4f} {ma_f1:>9.4f}")
    results.append({
        "strategy": "3-Gate Max-Gap", "threshold": "dynamic",
        "micro_p": mi_p, "micro_r": mi_r, "micro_f1": mi_f1,
        "macro_p": ma_p, "macro_r": ma_r, "macro_f1": ma_f1
    })
    
    print("=" * 95)
    
    # Save CSV if requested
    if args.output:
        import pandas as pd
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\n✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()
