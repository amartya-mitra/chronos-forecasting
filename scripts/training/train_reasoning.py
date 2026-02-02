"""
Reasoning SFT Training Script for Chronos-2

This script implements the "Reasoning" mode training for Chronos-2 models.
It adds two control tokens at the END of the vocabulary (safe append):
- <|fast_mode|>: ID 4096 - Standard forecasting
- <|reasoning_mode|>: ID 4097 - Decomposition + forecasting

CRITICAL FIXES (v2):
1. Tokens are appended at vocab END (4096, 4097) - NOT overwriting bins at 2, 3
2. max_new_tokens set to 300 to accommodate Reasoning Mode output (257 tokens)
3. n_special_tokens remains at 2 (PAD=0, EOS=1) - bins stay at 2-4095
"""

import logging
import os
import random
import sys
import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np
import yaml
from fire import Fire
from gluonts.dataset.common import ProcessDataEntry
from itertools import cycle

# Simple Cyclic replacement
def Cyclic(dataset):
    return cycle(dataset)

from gluonts.transform import (
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    FilterTransformation,
)
from transformers import AutoConfig, AutoModelForSeq2SeqLM, set_seed, GenerationConfig

import chronos
from chronos import ChronosConfig, ChronosTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS - Control Token IDs (Appended at vocabulary END)
# =============================================================================
# Original vocab: 4096 tokens (0=PAD, 1=EOS, 2-4095=bins)
# New vocab: 4098 tokens (0=PAD, 1=EOS, 2-4095=bins, 4096=fast, 4097=reasoning)
ORIGINAL_VOCAB_SIZE = 4096
FAST_MODE_TOKEN_ID = 4096      # Appended at end
REASONING_MODE_TOKEN_ID = 4097  # Appended at end
NEW_VOCAB_SIZE = 4098

# Reasoning mode output length: Trend(64) + Season(64) + Vol(64) + Forecast(64) + EOS(1) = 257
# Set buffer for safety
MAX_NEW_TOKENS_REASONING = 300


# =============================================================================
# DATASET CLASS
# =============================================================================
class ReasoningChronosDataset(torch.utils.data.IterableDataset):
    """
    Custom dataset that produces Fast Mode and Reasoning Mode samples.
    
    Fast Mode: [<|fast_mode|> + History] -> [Forecast + EOS]
    Reasoning Mode: [<|reasoning_mode|> + History] -> [Trend + Season + Vol + Forecast + EOS]
    """
    
    def __init__(
        self,
        datasets: list,
        probabilities: List[float],
        tokenizer: ChronosTokenizer,
        context_length: int = 512,
        prediction_length: int = 64,
        drop_prob: float = 0.2,
        min_past: Optional[int] = None,
        model_type: str = "seq2seq",
        mode: str = "training",
        np_dtype=np.float32,
    ):
        super().__init__()
        self.datasets = datasets
        self.probabilities = probabilities
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.drop_prob = drop_prob
        self.min_past = min_past or prediction_length
        self.model_type = model_type
        self.mode = mode
        self.np_dtype = np_dtype
        
        # Token IDs - APPENDED AT VOCAB END (not overwriting bins!)
        self.fast_token_id = FAST_MODE_TOKEN_ID       # 4096
        self.reason_token_id = REASONING_MODE_TOKEN_ID # 4097

    def _create_instance_splitter(self, mode: str):
        instance_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_instances=1,
            min_past=self.min_past,
            min_future=self.prediction_length,
        )
        if mode == "validation":
            instance_sampler = ValidationSplitSampler(min_future=self.prediction_length)
            
        return InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            dummy_value=np.nan,
            time_series_fields=["trend", "seasonal", "volatility"],
        )

    def preprocess_entry(self, entry: dict, mode: str) -> dict:
        needed_keys = ["start", "target", "trend", "seasonal", "volatility"]
        new_entry = {f: entry[f] for f in needed_keys if f in entry}
        
        # Convert start to pd.Period for GluonTS splitting
        import pandas as pd
        if "start" in new_entry:
            start = new_entry["start"]
            if not isinstance(start, pd.Period):
                new_entry["start"] = pd.Period(start, freq="h")  # Use lowercase 'h'

        for k in ["target", "trend", "seasonal", "volatility"]:
            if k in new_entry:
                new_entry[k] = np.asarray(new_entry[k], dtype=self.np_dtype)
        
        # Simple drop out logic
        if mode == "training" and self.drop_prob > 0:
            target = new_entry["target"].copy()
            drop_p = np.random.uniform(low=0.0, high=self.drop_prob)
            mask = np.random.choice([True, False], size=len(target), p=[drop_p, 1 - drop_p])
            target[mask] = np.nan
            new_entry["target"] = target
            
        return new_entry

    def to_hf_format(self, entry: dict) -> dict:
        mode = np.random.choice(["fast", "reasoning"], p=[0.5, 0.5])
        
        # Context (Encoder Input)
        past_target = torch.tensor(entry["past_target"]).unsqueeze(0)
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(past_target)
        
        # Prepend control token
        token_to_add = self.fast_token_id if mode == "fast" else self.reason_token_id
        control_token = torch.tensor([[token_to_add]], dtype=input_ids.dtype, device=input_ids.device)
        control_mask = torch.tensor([[True]], dtype=attention_mask.dtype, device=attention_mask.device)
        
        input_ids = torch.cat([control_token, input_ids], dim=1)
        attention_mask = torch.cat([control_mask, attention_mask], dim=1)
        
        # Target (Decoder Output)
        future_target = torch.tensor(entry["future_target"]).unsqueeze(0)
        
        if mode == "fast":
            labels, labels_mask = self.tokenizer.label_input_transform(future_target, scale)
        else:
            # Reasoning mode: Trend + Season + Vol + Forecast
            def tokenize_component(comp_name):
                data = torch.tensor(entry[f"future_{comp_name}"]).unsqueeze(0)
                ids, mask, _ = self.tokenizer._input_transform(data, scale=scale)
                return ids, mask

            trend_ids, trend_mask = tokenize_component("trend")
            season_ids, season_mask = tokenize_component("seasonal")
            vol_ids, vol_mask = tokenize_component("volatility")
            forecast_ids, forecast_mask = self.tokenizer.label_input_transform(future_target, scale)
            
            labels = torch.cat([trend_ids, season_ids, vol_ids, forecast_ids], dim=1)
            labels_mask = torch.cat([trend_mask, season_mask, vol_mask, forecast_mask], dim=1)

        labels[~labels_mask] = -100  # Ignore in loss
        
        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        split_transform = self._create_instance_splitter(self.mode) + FilterTransformation(
             condition=lambda entry: (~np.isnan(entry["past_target"])).sum() > 0
        )
        
        dataset = self.datasets[0]
        iter_data = iter(dataset)
        
        # For validation/test, iterate once through the dataset (finite)
        # For training, iterate infinitely
        samples_yielded = 0
        max_samples = len(dataset) * 2 if self.mode != "training" else float('inf')
        
        while samples_yielded < max_samples:
            try:
                entry = next(iter_data)
            except StopIteration:
                if self.mode != "training":
                    # End iteration for validation after one pass
                    break
                iter_data = iter(dataset)
                entry = next(iter_data)
                
            entry = self.preprocess_entry(entry, self.mode)
            
            for split_entry in split_transform([entry], is_train=(self.mode=="training")):
                yield self.to_hf_format(split_entry)
                samples_yielded += 1
                if samples_yielded >= max_samples:
                    break


# =============================================================================
# EMBEDDING RESIZE FUNCTION (Safe Append - No Bin Shift!)
# =============================================================================
def append_control_tokens(model):
    """
    Safely append control tokens to the END of vocabulary.
    
    Original: 4096 tokens (0=PAD, 1=EOS, 2-4095=bins)
    New:      4098 tokens (0=PAD, 1=EOS, 2-4095=bins, 4096=fast, 4097=reasoning)
    
    This does NOT modify existing token embeddings - bins remain at their original positions!
    """
    old_vocab_size = model.model.shared.num_embeddings
    
    logger.info(f"Original vocab size: {old_vocab_size}")
    logger.info(f"Appending 2 control tokens at positions {old_vocab_size} and {old_vocab_size + 1}")
    
    # Resize embeddings (HF method initializes new tokens with mean+cov of existing)
    model.model.resize_token_embeddings(NEW_VOCAB_SIZE)
    
    # Optionally re-initialize the new token embeddings with small random values
    # (The default HF initialization is also reasonable)
    with torch.no_grad():
        model.model.shared.weight.data[FAST_MODE_TOKEN_ID] = torch.randn(
            model.model.shared.weight.shape[1]
        ) * 0.02
        model.model.shared.weight.data[REASONING_MODE_TOKEN_ID] = torch.randn(
            model.model.shared.weight.shape[1]
        ) * 0.02
    
    logger.info(f"New vocab size: {model.model.shared.num_embeddings}")
    logger.info(f"Control tokens: fast={FAST_MODE_TOKEN_ID}, reasoning={REASONING_MODE_TOKEN_ID}")
    
    return model


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================
def train(
    config_path: str,
    output_dir: str = "./output/reasoning-v4-from-amazon/",
    dataset_path: str = "gifteval-reasoning.arrow",
):
    """Main training function with critical fixes applied."""
    
    # Load config
    with open(config_path) as fp:
        config = yaml.safe_load(fp)
    
    # DO NOT modify n_special_tokens - keep it at 2!
    # Bins remain at positions 2-4095, control tokens are APPENDED at 4096-4097
    logger.info(f"Config n_special_tokens: {config.get('n_special_tokens', 2)} (keeping unchanged)")
    logger.info(f"Config n_tokens: {config.get('n_tokens', 4096)} -> Will resize to {NEW_VOCAB_SIZE}")
    
    # Filter config for ChronosConfig validation
    chronos_config_keys = [
        "tokenizer_class", "tokenizer_kwargs", "context_length", "prediction_length",
        "n_tokens", "n_special_tokens", "pad_token_id", "eos_token_id", "use_eos_token",
        "model_type", "num_samples", "temperature", "top_k", "top_p"
    ]
    chronos_config_dict = {k: v for k, v in config.items() if k in chronos_config_keys}
    chronos_config = ChronosConfig(**chronos_config_dict)
    
    # Tokenizer (uses original config - bins at 2-4095)
    tokenizer = chronos_config.create_tokenizer()
    
    # Load pretrained model - use original Amazon Chronos (not any local finetuned checkpoints)
    model_id = config.get("model_id", "amazon/chronos-t5-small")
    logger.info(f"Loading model from: {model_id}")
    
    from chronos import ChronosPipeline
    pipeline = ChronosPipeline.from_pretrained(model_id)
    model = pipeline.model
    
    # =========================================================================
    # CRITICAL FIX 1: Append control tokens at vocab END
    # =========================================================================
    model = append_control_tokens(model)
    
    # Update model config for new vocab size (but NOT n_special_tokens!)
    model.model.config.vocab_size = NEW_VOCAB_SIZE
    
    # =========================================================================
    # CRITICAL FIX 2: Set max_new_tokens for Reasoning Mode output length
    # =========================================================================
    logger.info(f"Setting max_new_tokens = {MAX_NEW_TOKENS_REASONING} for Reasoning Mode")
    
    # Update generation config
    model.model.generation_config.max_new_tokens = MAX_NEW_TOKENS_REASONING
    model.model.generation_config.min_new_tokens = 1
    
    # Also update model config if it has max_length
    if hasattr(model.model.config, 'max_length'):
        original_max = model.model.config.max_length
        model.model.config.max_length = MAX_NEW_TOKENS_REASONING
        logger.info(f"Updated model.config.max_length: {original_max} -> {MAX_NEW_TOKENS_REASONING}")
    
    # =========================================================================
    # Dataset with Train/Val Split
    # =========================================================================
    import datasets
    arrow_ds = datasets.load_dataset("arrow", data_files=dataset_path, split="train")
    
    # Split 80% train, 20% validation
    split_ds = arrow_ds.train_test_split(test_size=0.2, seed=42)
    logger.info(f"Dataset split: Train={len(split_ds['train'])}, Val={len(split_ds['test'])}")
    
    train_dataset = ReasoningChronosDataset(
        datasets=[split_ds["train"]],
        probabilities=[1.0],
        tokenizer=tokenizer,
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        model_type=config["model_type"],
        mode="training",
    )
    
    eval_dataset = ReasoningChronosDataset(
        datasets=[split_ds["test"]],
        probabilities=[1.0],
        tokenizer=tokenizer,
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        model_type=config["model_type"],
        mode="validation",
    )
    
    # =========================================================================
    # Training - Extended Configuration
    # =========================================================================
    from transformers import TrainingArguments, Trainer
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=1e-5,              # Lower LR for fine-tuning
        lr_scheduler_type="cosine",      # Cosine annealing
        warmup_ratio=0.1,                # 10% warmup
        max_steps=1000,                  # Extended training
        save_steps=200,
        logging_steps=50,
        eval_strategy="steps",           # Enable evaluation
        eval_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",                # Disable wandb/tensorboard for now
    )
    
    def collate_fn(batch):
        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]
        masks = [b["attention_mask"] for b in batch]
        
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        masks_padded = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": masks_padded,
            "labels": labels_padded
        }

    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )
    
    logger.info("=" * 60)
    logger.info("Starting Reasoning SFT Training (v2 - with critical fixes)")
    logger.info("=" * 60)
    trainer.train()
    trainer.save_model(output_dir)
    
    # =========================================================================
    # Verification
    # =========================================================================
    run_verification(train_dataset, config)
    
    logger.info("=" * 60)
    logger.info("Training and verification complete!")
    logger.info("=" * 60)


# =============================================================================
# VERIFICATION FUNCTION
# =============================================================================
def run_verification(train_dataset, config):
    """Verify the pipeline is producing correct output."""
    
    logger.info("=" * 60)
    logger.info("Running Verification Tests")
    logger.info("=" * 60)
    
    pl = config["prediction_length"]
    expected_reasoning_length = 4 * pl + 1  # 257 for pl=64
    expected_fast_length = pl + 1  # 65 for pl=64
    
    max_samples = 50
    fast_count = 0
    reasoning_count = 0
    token_id_issues = []
    
    for i, batch in enumerate(train_dataset):
        if i >= max_samples:
            break
            
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        # Check control token position and value
        control_token = input_ids[0].item()
        
        if control_token == FAST_MODE_TOKEN_ID:
            fast_count += 1
            if labels.shape[0] != expected_fast_length:
                logger.warning(f"Sample {i}: Fast mode but label length {labels.shape[0]} != {expected_fast_length}")
        elif control_token == REASONING_MODE_TOKEN_ID:
            reasoning_count += 1
            if labels.shape[0] != expected_reasoning_length:
                logger.warning(f"Sample {i}: Reasoning mode but label length {labels.shape[0]} != {expected_reasoning_length}")
        else:
            # CRITICAL: Control token should NOT be in bin range (2-4095)
            if 2 <= control_token <= 4095:
                token_id_issues.append(f"Sample {i}: Token {control_token} is in BIN RANGE - collision detected!")
            else:
                token_id_issues.append(f"Sample {i}: Unexpected control token {control_token}")
    
    # Report results
    logger.info("-" * 40)
    logger.info("Verification Results:")
    logger.info(f"  Samples checked: {max_samples}")
    logger.info(f"  Fast mode samples: {fast_count}")
    logger.info(f"  Reasoning mode samples: {reasoning_count}")
    logger.info(f"  Expected fast token ID: {FAST_MODE_TOKEN_ID}")
    logger.info(f"  Expected reasoning token ID: {REASONING_MODE_TOKEN_ID}")
    
    if token_id_issues:
        logger.error("TOKEN ID COLLISION DETECTED!")
        for issue in token_id_issues[:5]:
            logger.error(f"  {issue}")
        return False
    else:
        logger.info("✓ No token ID collisions detected")
        
    if fast_count > 0 and reasoning_count > 0:
        logger.info("✓ Both Fast and Reasoning modes are being generated")
    else:
        logger.warning(f"Mode distribution issue: fast={fast_count}, reasoning={reasoning_count}")
    
    logger.info("-" * 40)
    return True


if __name__ == "__main__":
    Fire(train)
