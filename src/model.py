"""
model.py — PMA Cross-Encoder Architecture for Legal Case Entailment

Implements the Pooling-based Multi-Head Attention (PMA) mechanism adapted from
Lee et al. (2019) "Set Transformer" for permutation-invariant set processing,
applied here to aggregate distributed entailment signals across legal text.

Architecture:
    LegalBERT → Full Sequence Hidden States → PMA (4-head) → Flatten → Linear Classifier

References:
    - Lee et al., "Set Transformer: A Framework for Attention-Based
      Permutation-Invariant Input" (ICML 2019)
    - Chalkidis et al., "LEGAL-BERT: The Muppets straight out of Law School" (2020)
"""

import math
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


class PMAPooling(nn.Module):
    """Pooling by Multi-Head Attention (PMA).
    
    Instead of relying on the [CLS] token bottleneck, PMA uses learnable 
    query vectors to dynamically attend to different semantic regions of 
    the input sequence. This allows the model to anchor on multiple 
    distant entailment signals within lengthy legal paragraphs.
    
    Args:
        hidden_size: Dimensionality of the transformer hidden states.
        num_heads: Number of independent pooling attention heads.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = hidden_size
        
        # Learnable pooling queries — independent of input sequence
        self.q = nn.Parameter(torch.randn(num_heads, hidden_size))
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size) from transformer encoder.
            attention_mask: (batch, seq_len) binary mask for padding tokens.
            
        Returns:
            Flattened pooled representation: (batch, num_heads * hidden_size)
        """
        k = self.k(hidden_states)
        v = self.v(hidden_states)
        
        # Compute attention scores: (batch, num_heads, seq_len)
        scores = torch.einsum('hd,bsd->bhs', self.q, k) / math.sqrt(self.d_model)
        
        # Apply padding mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = self.softmax(scores)
        
        # Weighted sum over values: (batch, num_heads, hidden_size)
        pooled_output = torch.matmul(attn_weights, v)
        
        # Flatten to single vector: (batch, num_heads * hidden_size)
        return pooled_output.view(hidden_states.size(0), -1)


class PMALegalCrossEncoder(nn.Module):
    """Context-Safe PMA Cross-Encoder for legal case entailment.
    
    Replaces the standard [CLS] pooling bottleneck with a multi-head PMA
    layer that dynamically routes attention across the full sequence. This
    enables the model to capture entailment signals scattered across lengthy
    legal paragraphs while suppressing recurrent boilerplate.
    
    Args:
        model_name: HuggingFace model identifier (e.g., 'nlpaueb/legal-bert-base-uncased').
        num_labels: Number of output classes (default: 2 for entailment/not-entailment).
        num_heads: Number of PMA attention heads (default: 4).
    """
    
    def __init__(self, model_name: str, num_labels: int = 2, num_heads: int = 4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config
        self.pma = PMAPooling(hidden_size=self.config.hidden_size, num_heads=num_heads)
        self.classifier = nn.Linear(self.config.hidden_size * num_heads, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
                labels=None, **kwargs):
        outputs = self.bert(
            input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        pooled = self.pma(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(pooled)
        return SequenceClassifierOutput(logits=logits)


class CLSCrossEncoder(nn.Module):
    """Standard [CLS]-pooled cross-encoder baseline.
    
    Uses the standard BERT [CLS] token output with dropout for classification.
    Serves as the ablation baseline against the PMA architecture.
    
    Args:
        model_name: HuggingFace model identifier.
        num_labels: Number of output classes (default: 2).
    """
    
    def __init__(self, model_name: str, num_labels: int = 2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
                labels=None, **kwargs):
        outputs = self.bert(
            input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        pooled = self.dropout(outputs.last_hidden_state[:, 0, :])  # [CLS] token
        logits = self.classifier(pooled)
        return SequenceClassifierOutput(logits=logits)
