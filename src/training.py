"""
training.py — Focal Loss and Bucketed Negative Sampling for Legal Entailment

Implements the training components for the PMA Cross-Encoder:
  1. Focal Loss — down-weights easy examples to focus learning on hard negatives
     (lexical traps) that the model confidently misclassifies.
  2. Bucketed Negative Sampling — stratified 6:4:2 (Hard:Semi-Hard:Easy) curriculum
     that forces the model to construct strict decision boundaries around lexical
     traps without forgetting basic entailment features.

References:
    - Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
"""

import random
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance and hard negative mining.
    
    Applies a modulating factor (1 - p_t)^gamma to the standard cross-entropy
    loss, down-weighting well-classified examples and focusing training on
    hard negatives (lexical traps).
    
    Args:
        alpha: Weighting factor for the positive class (default: 0.75).
        gamma: Focusing parameter — higher values increase focus on hard 
               examples (default: 2.0).
        reduction: Reduction method ('mean' or 'none').
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return torch.mean(focal_loss) if self.reduction == 'mean' else focal_loss


class FocalLossTrainer(Trainer):
    """HuggingFace Trainer with Focal Loss instead of standard cross-entropy."""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = FocalLoss(alpha=0.75, gamma=2.0)
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


class CrossEntropyTrainer(Trainer):
    """HuggingFace Trainer with standard cross-entropy (for ablation baselines)."""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def create_bucketed_training_data(
    dataset: List[Dict],
    hybrid_map: Dict[str, List[str]],
    hard_count: int = 6,
    semi_hard_count: int = 4,
    easy_count: int = 2,
    query_ids: Optional[set] = None
) -> pd.DataFrame:
    """Create training pairs with bucketed negative sampling (6:4:2).
    
    Negatives are stratified by their wRRF retrieval rank:
      - Hard Negatives (ranks 1-20): Lexical traps with high surface overlap
      - Semi-Hard Negatives (ranks 21-60): Moderate difficulty
      - Easy Negatives (ranks 60+): Clearly non-entailing
    
    This curriculum forces the model to construct strict decision boundaries
    around lexical traps without forgetting basic entailment features.
    
    Args:
        dataset: List of case dictionaries with query, candidates, and ground truth.
        hybrid_map: Mapping from query_id to wRRF-ranked candidate filenames.
        hard_count: Number of hard negatives per positive (default: 6).
        semi_hard_count: Number of semi-hard negatives per positive (default: 4).
        easy_count: Number of easy negatives per positive (default: 2).
        query_ids: Optional set of query IDs to filter (None = use all).
        
    Returns:
        DataFrame with columns ['query', 'paragraph', 'label'].
    """
    records = []
    
    for case in dataset:
        if query_ids is not None and case["query_id"] not in query_ids:
            continue
        
        # CRITICAL DECOUPLING: Use plain fragment only (no base case context)
        query = case["entailed_fragment"]
        ground_truth = set(case["ground_truth"])
        candidates = case["candidates"]
        hybrid_top = hybrid_map.get(case["query_id"], [])
        
        # Positives
        for gt_file in ground_truth:
            if gt_file in candidates:
                records.append({"query": query, "paragraph": candidates[gt_file], "label": 1})
        
        # Bucketed negatives from wRRF rankings
        negatives = [f for f in hybrid_top if f not in ground_truth and f in candidates]
        
        if len(negatives) > 0:
            # Hard Negatives (Ranks 1-20) — Lexical Traps
            hard = negatives[:20]
            for n in hard[:hard_count]:
                records.append({"query": query, "paragraph": candidates[n], "label": 0})
            
            # Semi-Hard Negatives (Ranks 21-60)
            semi_hard = negatives[20:60]
            if semi_hard:
                sample_size = min(semi_hard_count, len(semi_hard))
                for n in random.sample(semi_hard, sample_size):
                    records.append({"query": query, "paragraph": candidates[n], "label": 0})
            
            # Easy Negatives (Ranks 60+)
            easy = negatives[60:]
            if easy:
                sample_size = min(easy_count, len(easy))
                for n in random.sample(easy, sample_size):
                    records.append({"query": query, "paragraph": candidates[n], "label": 0})
    
    return pd.DataFrame(records)


def create_standard_training_data(
    dataset: List[Dict],
    hybrid_map: Dict[str, List[str]],
    neg_per_positive: int = 12,
    query_ids: Optional[set] = None
) -> pd.DataFrame:
    """Create training pairs with uniform random negative sampling (for ablation).
    
    Args:
        dataset: List of case dictionaries.
        hybrid_map: Mapping from query_id to wRRF-ranked candidate filenames.
        neg_per_positive: Total negatives per positive example.
        query_ids: Optional set of query IDs to filter.
        
    Returns:
        DataFrame with columns ['query', 'paragraph', 'label'].
    """
    records = []
    
    for case in dataset:
        if query_ids is not None and case["query_id"] not in query_ids:
            continue
        
        query = case["entailed_fragment"]
        ground_truth = set(case["ground_truth"])
        candidates = case["candidates"]
        hybrid_top = hybrid_map.get(case["query_id"], [])
        
        for gt_file in ground_truth:
            if gt_file in candidates:
                records.append({"query": query, "paragraph": candidates[gt_file], "label": 1})
        
        negatives = [f for f in hybrid_top if f not in ground_truth and f in candidates]
        if negatives:
            sample_size = min(neg_per_positive, len(negatives))
            for n in random.sample(negatives, sample_size):
                records.append({"query": query, "paragraph": candidates[n], "label": 0})
    
    return pd.DataFrame(records)
