"""
inference.py — Stage 3: 3-Gate Safe Max-Gap Variance Reduction Algorithm

Implements the deterministic gating mechanism that stabilizes the final 
decision boundary by identifying natural probability gaps in the ranked 
candidate list. This replaces brittle static thresholds with a dynamic, 
per-query acceptance criterion.

The algorithm operates as a defensive cascade:
  Gate 1 (Noise Filter): Catches total entailment failure
  Gate 2 (Plateau Guard): Detects low-confidence plateaus  
  Gate 3 (Rigorous Acceptance): Multi-paragraph acceptance only when a
    mathematically significant, high-confidence boundary is detected
"""

from typing import List, Tuple


def safe_max_gap(
    scored_candidates: List[Tuple[str, float]],
    tau_none: float = 0.10,
    tau_low: float = 0.20,
    tau_accept: float = 0.35,
    gap_min: float = 0.06,
    ratio_min: float = 1.30,
    search_depth: int = 8
) -> List[str]:
    """3-Gate Safe Max-Gap algorithm for dynamic threshold selection.
    
    Given a sorted list of (filename, probability) candidates, determines
    the optimal cut-off point using a three-gate defensive cascade.
    
    Args:
        scored_candidates: List of (filename, probability) tuples, sorted
                          by probability in descending order.
        tau_none: Gate 1 threshold — below this, the model has no signal.
        tau_low: Gate 2 threshold — low confidence plateau detection.
        tau_accept: Gate 3 minimum probability for acceptance.
        gap_min: Gate 3 minimum probability gap for acceptance.
        ratio_min: Gate 3 minimum decay ratio for acceptance.
        search_depth: Maximum number of top candidates to scan for gaps.
        
    Returns:
        List of predicted entailing paragraph filenames.
    """
    if not scored_candidates:
        return []
    
    p1 = scored_candidates[0][1]
    
    # Gate 1: Noise Filter — total entailment failure
    if p1 < tau_none:
        return [scored_candidates[0][0]]
    
    # Gate 2: Plateau Guard — low-confidence guessing
    p2 = scored_candidates[1][1] if len(scored_candidates) > 1 else 0
    if p1 < tau_low and (p1 - p2) < 0.03:
        return [scored_candidates[0][0]]
    
    # Find the maximum probability gap in the top candidates
    max_gap = 0
    best_cut = 1
    limit = min(len(scored_candidates), search_depth)
    
    for i in range(limit - 1):
        gap = scored_candidates[i][1] - scored_candidates[i + 1][1]
        if gap > max_gap:
            max_gap = gap
            best_cut = i + 1
    
    # Gate 3: Rigorous Acceptance — multi-paragraph only with strong evidence
    cut_prob = scored_candidates[best_cut - 1][1]
    next_prob = scored_candidates[best_cut][1] if best_cut < len(scored_candidates) else 0
    
    c_acc = (
        (cut_prob >= tau_accept) and 
        (max_gap >= gap_min) and 
        ((cut_prob / (next_prob + 1e-9)) >= ratio_min)
    )
    
    if c_acc:
        return [f for f, p in scored_candidates[:best_cut]]
    else:
        return [scored_candidates[0][0]]


def static_threshold(
    scored_candidates: List[Tuple[str, float]], 
    threshold: float
) -> List[str]:
    """Simple static threshold baseline for ablation comparison.
    
    Selects all candidates with probability >= threshold.
    Falls back to top-1 if no candidate exceeds the threshold.
    
    Args:
        scored_candidates: List of (filename, probability) tuples, sorted descending.
        threshold: Static probability threshold.
        
    Returns:
        List of predicted entailing paragraph filenames.
    """
    if not scored_candidates:
        return []
    selected = [f for f, p in scored_candidates if p >= threshold]
    return selected if selected else [scored_candidates[0][0]]


def top_k_prediction(
    scored_candidates: List[Tuple[str, float]], 
    k: int = 1
) -> List[str]:
    """Top-k prediction baseline.
    
    Args:
        scored_candidates: List of (filename, probability) tuples, sorted descending.
        k: Number of top candidates to return.
        
    Returns:
        List of top-k predicted filenames.
    """
    if not scored_candidates:
        return []
    return [f for f, p in scored_candidates[:k]]
