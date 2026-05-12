#!/usr/bin/env python3
"""
train.py — Standalone training script for the PMA Cross-Encoder.

Trains the LegalBERT + PMA cross-encoder with focal loss and bucketed 
negative sampling on the COLIEE 2026 Task 2 training data.

Usage:
    python train.py --config configs/default.yaml --data_dir /path/to/cases --labels data/task2_train_labels_2026.json
    python train.py --config configs/default.yaml --data_dir /path/to/cases --labels data/task2_train_labels_2026.json --seed 42

Prerequisites:
    - wRRF retrieval must be run first (requires GPU for BGE-m3 embeddings)
    - The notebook (notebooks/coliee-task2.ipynb) handles the full end-to-end pipeline
    - This script is for standalone training once retrieval results exist
"""

import argparse
import json
import os
import sys
import random

import numpy as np
import torch
import yaml

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import PMALegalCrossEncoder, CLSCrossEncoder
from src.training import (
    FocalLossTrainer, CrossEntropyTrainer,
    create_bucketed_training_data, create_standard_training_data
)


def load_dataset(data_dir, labels_file):
    """Load COLIEE case data from disk."""
    from tqdm import tqdm
    
    with open(labels_file, "r", encoding="utf-8") as f:
        labels = json.load(f)
    
    dataset = []
    for query_id, positive_paras in tqdm(labels.items(), desc="Loading cases"):
        case_folder = os.path.join(data_dir, query_id)
        if not os.path.exists(case_folder):
            continue
        
        fragment_path = os.path.join(case_folder, "entailed_fragment.txt")
        base_case_path = os.path.join(case_folder, "base_case.txt")
        paragraphs_dir = os.path.join(case_folder, "paragraphs")
        
        try:
            with open(fragment_path, "r", encoding="utf-8") as f:
                entailed_fragment = f.read().strip()
        except FileNotFoundError:
            continue
        
        try:
            with open(base_case_path, "r", encoding="utf-8") as f:
                base_case = f.read().strip()
        except FileNotFoundError:
            base_case = ""
        
        candidates = {}
        if os.path.exists(paragraphs_dir):
            for para_file in os.listdir(paragraphs_dir):
                if para_file.endswith(".txt"):
                    with open(os.path.join(paragraphs_dir, para_file), "r", encoding="utf-8") as f:
                        candidates[para_file] = f.read().strip()
        
        dataset.append({
            "query_id": query_id,
            "entailed_fragment": entailed_fragment,
            "base_case": base_case,
            "candidates": candidates,
            "ground_truth": positive_paras
        })
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="COLIEE Task 2 — Train PMA Cross-Encoder"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML configuration file.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to directory containing case folders.")
    parser.add_argument("--labels", type=str, required=True,
                        help="Path to training labels JSON file.")
    parser.add_argument("--hybrid_map", type=str, default=None,
                        help="Path to wRRF retrieval results JSON.")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save trained model.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--use_pma", action="store_true", default=True,
                        help="Use PMA pooling (default: True).")
    parser.add_argument("--no_pma", action="store_true",
                        help="Disable PMA pooling (use [CLS] baseline).")
    parser.add_argument("--use_focal", action="store_true", default=True,
                        help="Use Focal Loss (default: True).")
    parser.add_argument("--no_focal", action="store_true",
                        help="Disable Focal Loss (use CE baseline).")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load data
    print(f"\n📂 Loading dataset from {args.data_dir}...")
    dataset = load_dataset(args.data_dir, args.labels)
    print(f"✅ Loaded {len(dataset)} query cases.")
    
    # Load hybrid map
    if args.hybrid_map:
        with open(args.hybrid_map, "r") as f:
            hybrid_map = json.load(f)
        print(f"✅ Loaded wRRF results for {len(hybrid_map)} queries.")
    else:
        print("⚠️  No hybrid_map provided. Using all candidates as negatives.")
        hybrid_map = {
            case["query_id"]: list(case["candidates"].keys()) 
            for case in dataset
        }
    
    # Build training data
    use_pma = args.use_pma and not args.no_pma
    use_focal = args.use_focal and not args.no_focal
    
    neg_config = config.get("negative_sampling", {})
    if use_focal and neg_config.get("strategy") == "bucketed":
        print("📊 Creating bucketed training data (6:4:2)...")
        train_df = create_bucketed_training_data(
            dataset, hybrid_map,
            hard_count=neg_config.get("hard_count", 6),
            semi_hard_count=neg_config.get("semi_hard_count", 4),
            easy_count=neg_config.get("easy_count", 2)
        )
    else:
        print("📊 Creating standard training data...")
        train_df = create_standard_training_data(
            dataset, hybrid_map,
            neg_per_positive=neg_config.get("total_per_positive", 12)
        )
    
    print(f"✅ Generated {len(train_df)} training pairs "
          f"(pos: {(train_df['label']==1).sum()}, neg: {(train_df['label']==0).sum()})")
    
    # Tokenize
    from datasets import Dataset as HFDataset
    from transformers import AutoTokenizer, TrainingArguments
    
    model_config = config.get("model", {})
    model_name = model_config.get("encoder", "nlpaueb/legal-bert-base-uncased")
    max_length = model_config.get("max_seq_length", 512)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_dataset = HFDataset.from_pandas(train_df)
    
    def tokenize_fn(ex):
        return tokenizer(ex["query"], ex["paragraph"], 
                        padding="max_length", truncation=True, max_length=max_length)
    
    tokenized = hf_dataset.map(tokenize_fn, batched=True).remove_columns(["query", "paragraph"])
    
    # Build model
    if use_pma:
        print(f"🧠 Building PMA Cross-Encoder ({model_name}, {model_config.get('pma_heads', 4)} heads)...")
        model = PMALegalCrossEncoder(
            model_name, 
            num_labels=model_config.get("num_labels", 2),
            num_heads=model_config.get("pma_heads", 4)
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        print(f"🧠 Building [CLS] Cross-Encoder ({model_name})...")
        model = CLSCrossEncoder(
            model_name, 
            num_labels=model_config.get("num_labels", 2)
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training arguments
    train_config = config.get("training", {})
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=train_config.get("epochs", 4),
        per_device_train_batch_size=train_config.get("batch_size", 8),
        gradient_accumulation_steps=train_config.get("gradient_accumulation", 2),
        learning_rate=train_config.get("learning_rate", 2e-5),
        warmup_ratio=train_config.get("warmup_ratio", 0.1),
        fp16=train_config.get("fp16", True),
        weight_decay=train_config.get("weight_decay", 0.01),
        logging_steps=50,
        report_to="none",
        save_strategy="epoch",
        seed=args.seed,
    )
    
    # Train
    TrainerCls = FocalLossTrainer if use_focal else CrossEntropyTrainer
    trainer = TrainerCls(model=model, args=training_args, train_dataset=tokenized)
    
    pma_str = "PMA" if use_pma else "[CLS]"
    loss_str = "Focal" if use_focal else "CE"
    print(f"\n🚀 Training {pma_str} + {loss_str} (seed={args.seed})...")
    trainer.train()
    
    # Save
    save_path = os.path.join(args.output_dir, f"final_model_s{args.seed}")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n✅ Model saved to {save_path}")


if __name__ == "__main__":
    main()
