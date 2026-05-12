"""
retrieval.py — Stage 1: Weighted Reciprocal Rank Fusion (wRRF) Retrieval

Implements the hybrid retrieval pipeline combining sparse lexical (BM25) and
dense semantic (BGE-m3) retrieval branches, fused via asymmetrically weighted
Reciprocal Rank Fusion.

The key insight is that in the legal domain, dense embeddings should be the
primary retrieval driver (w_bge=1.0) while sparse retrieval serves only as a 
supplementary exact-match safety net (w_bm25=0.2), preventing lexical metrics
from overwhelming true semantic intent.

References:
    - Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and
      individual Rank Learning Methods" (SIGIR 2009)
    - Chen et al., "BGE M3-Embedding" (2024)
"""

import re
import string
import numpy as np
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi


def tokenize_text(text: str) -> List[str]:
    """Simple whitespace tokenizer with punctuation removal for BM25."""
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text.split()


def compress_base_case(base_case: str, fragment: str, top_k: int = 2) -> str:
    """Extract the top-k most relevant sentences from the base case.
    
    Uses BM25 scoring at the sentence level to identify the most topically 
    relevant fragments of the base case for query enrichment, without
    introducing excessive noise.
    
    Args:
        base_case: Full text of the base case document.
        fragment: The entailed fragment (query).
        top_k: Number of top-scoring sentences to retain.
        
    Returns:
        Concatenated top-k sentences as compressed context.
    """
    if not base_case:
        return ""
    
    # Split by sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', base_case)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) > 5]
    
    if not sentences:
        return ""
    
    tokenized = [s.lower().split() for s in sentences]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(fragment.lower().split())
    
    # Get top_k highest scoring sentences, maintain original order
    top_indices = scores.argsort()[-top_k:][::-1]
    return " ".join([sentences[i] for i in sorted(top_indices)])


def build_enriched_query(fragment: str, base_case: str) -> str:
    """Construct an enriched query by combining context with the fragment.
    
    The enriched query is used ONLY for Stage 1 retrieval. The context is
    aggressively stripped before Stage 2 cross-encoding (the "decoupling").
    
    Args:
        fragment: The entailed fragment (query).
        base_case: Full text of the base case document.
        
    Returns:
        Enriched query string with context prefix.
    """
    context = compress_base_case(base_case, fragment)
    if context:
        return f"[CONTEXT] {context} [FRAGMENT] {fragment}"
    return fragment


def weighted_reciprocal_rank_fusion(
    bge_ranked_list: List[str], 
    bm25_ranked_list: List[str], 
    k: int = 60, 
    w_bge: float = 1.0, 
    w_bm25: float = 0.2
) -> List[str]:
    """Merge sparse and dense retrieval rankings via weighted RRF.
    
    The asymmetric weighting (w_bge >> w_bm25) ensures dense embeddings 
    remain the primary retrieval driver while BM25 serves as a supplementary
    exact-match safety net.
    
    Args:
        bge_ranked_list: Documents ranked by BGE-m3 cosine similarity.
        bm25_ranked_list: Documents ranked by BM25 score.
        k: RRF smoothing constant (default: 60, per original literature).
        w_bge: Weight for dense retrieval branch.
        w_bm25: Weight for sparse retrieval branch.
        
    Returns:
        Fused document ranking (highest score first).
    """
    scores = {}
    for rank, doc in enumerate(bge_ranked_list):
        scores[doc] = scores.get(doc, 0.0) + w_bge * (1.0 / (k + rank + 1))
    for rank, doc in enumerate(bm25_ranked_list):
        scores[doc] = scores.get(doc, 0.0) + w_bm25 * (1.0 / (k + rank + 1))
    
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def run_bm25_retrieval(
    query: str, 
    para_filenames: List[str], 
    para_texts: List[str]
) -> List[str]:
    """Run BM25 sparse retrieval and return ranked filenames.
    
    Args:
        query: The (enriched) query string.
        para_filenames: List of candidate paragraph filenames.
        para_texts: List of corresponding paragraph texts.
        
    Returns:
        Paragraph filenames ranked by BM25 score (descending).
    """
    tokenized_corpus = [tokenize_text(doc) for doc in para_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenize_text(query))
    scored = sorted(zip(para_filenames, bm25_scores), key=lambda x: x[1], reverse=True)
    return [f for f, s in scored]
