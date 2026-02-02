
import logging
from typing import List, Optional
import numpy as np
import torch
from gluonts.dataset.common import ProcessDataEntry
from gluonts.transform import (
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    Cyclic,
    FilterTransformation,
)
from scripts.training.train import ChronosDataset

class ReasoningChronosDataset(ChronosDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We need the control token IDs. 
        # Assuming we updated config:
        # pad=0, eos=1, fast=2, reason=3
        # n_special_tokens=4
        self.fast_token_id = 2
        self.reason_token_id = 3
    
    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "test", "validation"]

        instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=self.min_past,
                min_future=self.prediction_length,
            ),
            "test": TestSplitSampler(),
            "validation": ValidationSplitSampler(min_future=self.prediction_length),
        }[mode]

        return InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            dummy_value=np.nan,
            # CRITICAL: We want to split these extra fields too
            time_series_fields=["trend", "seasonal", "volatility"],
        )

    def preprocess_entry(self, entry: dict, mode: str) -> dict:
        # Start with standard preprocessing
        # But standard preprocess_entry (in train.py) filters keys to just ["start", "target"]!!
        # We MUST Override it to keep our new fields.
        
        # We need to manually handle what train.py did but for all fields.
        # train.py:331
        
        # 1. Select subset of keys
        needed_keys = ["start", "target", "trend", "seasonal", "volatility"]
        # Only keep if they exist (datasets might not have them if not augmented, but we assume yes)
        new_entry = {f: entry[f] for f in needed_keys if f in entry}
        
        # 2. Cast to numpy
        for k in ["target", "trend", "seasonal", "volatility"]:
            if k in new_entry:
                new_entry[k] = np.asarray(new_entry[k], dtype=self.np_dtype)
                if self.model_type == "causal" and k == "target":
                     new_entry[k] = self.imputation_method(new_entry[k])

        # 3. Drop probability (only for target?)
        # If we drop target, should we drop reasoning? 
        # The reasoning is derived from target. If target is masked, reasoning should probably be masked too??
        # For simplicity, let's just stick to target masking logic from base class if we wanted, 
        # but let's assume we implement simple logic here.
        
        if mode == "training" and self.drop_prob > 0:
            target = new_entry["target"].copy()
            drop_p = np.random.uniform(low=0.0, high=self.drop_prob)
            mask = np.random.choice(
                [True, False], size=len(target), p=[drop_p, 1 - drop_p]
            )
            target[mask] = np.nan
            new_entry["target"] = target
            # Ideally apply same mask to trend/season which are derived from it?
            # Or just leave them?
            # Let's leave them for now.
            
        return new_entry

    def to_hf_format(self, entry: dict) -> dict:
        # 50/50 split
        # We want to support mixed training.
        # Randomly choose mode.
        mode = np.random.choice(["fast", "reasoning"], p=[0.5, 0.5])
        
        # 1. Prepare Context (History)
        past_target = torch.tensor(entry["past_target"]).unsqueeze(0)
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(
            past_target
        )
        
        # PREPEND Control Token
        # input_ids shape: (batch=1, seq_len)
        if mode == "fast":
            token_to_add = self.fast_token_id
        else:
            token_to_add = self.reason_token_id
            
        control_token = torch.tensor([[token_to_add]], dtype=input_ids.dtype, device=input_ids.device)
        control_mask = torch.tensor([[True]], dtype=attention_mask.dtype, device=attention_mask.device)
        
        # Concat: [Control] + [History]
        # input_ids = torch.cat([control_token, input_ids], dim=1) # NO, usually prepended
        # But wait, ChronosTokenizer buckets might need to be shifted?
        # If we resized embeddings and shifted weights, the tokenizer logic (which adds n_special_tokens)
        # will map 0 -> 4. 
        # So "bin 0" becomes index 4.
        # Our tokens are index 2, 3.
        # So we can just prepend raw IDs 2 or 3.
        
        input_ids = torch.cat([control_token, input_ids], dim=1)
        attention_mask = torch.cat([control_mask, attention_mask], dim=1)
        
        # 2. Prepare Target (Labels)
        future_target = torch.tensor(entry["future_target"]).unsqueeze(0)
        
        if mode == "fast":
            # Target = [Forecast]
            # Standard formatting
            labels, labels_mask = self.tokenizer.label_input_transform(future_target, scale)
        else:
            # Target = [Trend] + [Season] + [Vol] + [Forecast]
            # We need to tokenize each.
            # Use same 'scale' from context? Yes, keeping consistency.
            
            # Helper to tokenize a component
            def tokenize_component(comp_name):
                # entry["future_trend"] etc.
                data = torch.tensor(entry[f"future_{comp_name}"]).unsqueeze(0)
                # We typically don't append EOS for intermediate segments?
                # Or do we? 
                # "Concatenate them... to generate the tokens."
                # If we use `label_input_transform`, it might append EOS if configured.
                # We probably want EOS only at the very end.
                # So we use `_input_transform` directly?
                # Tokenizer `_input_transform` returns (ids, mask, scale). We want ids.
                ids, mask, _ = self.tokenizer._input_transform(data, scale=scale)
                return ids, mask

            trend_ids, trend_mask = tokenize_component("trend")
            season_ids, season_mask = tokenize_component("seasonal")
            vol_ids, vol_mask = tokenize_component("volatility")
            
            # Forecast (final) - Use standard label transform which might add EOS
            forecast_ids, forecast_mask = self.tokenizer.label_input_transform(future_target, scale)
            
            # Concatenate
            labels = torch.cat([trend_ids, season_ids, vol_ids, forecast_ids], dim=1)
            labels_mask = torch.cat([trend_mask, season_mask, vol_mask, forecast_mask], dim=1)

        labels[~labels_mask] = -100  # PyTorch ignore index
        
        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
            "mode": mode # Optional debugging
        }
