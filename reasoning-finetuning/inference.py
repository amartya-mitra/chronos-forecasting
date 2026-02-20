#!/usr/bin/env python3
"""
Chronos-2 Reasoning Mode: Standalone Inference Script

This script provides a clean interface to use the reasoning-enabled Chronos model.
It handles control tokens, tokenization, generation, and de-amplification.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from typing import Dict, Union, List

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chronos import ChronosPipeline

# Constants (Matching the training and preparation scripts)
FAST_MODE_TOKEN_ID = 4096
REASONING_MODE_TOKEN_ID = 4097
PREDICTION_LENGTH = 64
DECOMPOSITION_LENGTH = 64
SEASONAL_AMP = 10.0
VOLATILITY_AMP = 50.0

class ChronosReasoningInference:
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the finetuned model directory.
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading Reasoning Model from: {model_path} on {device}")
        self.pipeline = ChronosPipeline.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float32,
        )
        self.tokenizer = self.pipeline.tokenizer
        self.device = device

    def forecast(
        self, 
        series: Union[np.ndarray, List[float]], 
        mode: str = "reasoning"
    ) -> Dict[str, np.ndarray]:
        """
        Generate forecast and (optional) decomposition.
        
        Args:
            series: Input time series values.
            mode: "reasoning" or "fast".
            
        Returns:
            Dictionary containing 'forecast' and decomposition components if in reasoning mode.
        """
        series_np = np.asarray(series, dtype=np.float32)
        context_tensor = torch.tensor(series_np).float().unsqueeze(0)
        
        # Transform to tokens
        token_ids, attention_mask, scale = self.tokenizer.context_input_transform(context_tensor)
        
        # Prepend appropriate control token
        control_id = REASONING_MODE_TOKEN_ID if mode == "reasoning" else FAST_MODE_TOKEN_ID
        
        control_token = torch.tensor([[control_id]], dtype=token_ids.dtype)
        control_mask = torch.tensor([[True]], dtype=attention_mask.dtype)
        
        input_ids = torch.cat([control_token, token_ids], dim=1).to(self.device)
        attention_mask = torch.cat([control_mask, attention_mask], dim=1).to(self.device)
        
        # Generation length
        if mode == "reasoning":
            max_new = 3 * DECOMPOSITION_LENGTH + PREDICTION_LENGTH
        else:
            max_new = PREDICTION_LENGTH
            
        # Generate
        with torch.no_grad():
            generated = self.pipeline.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prediction_length=max_new,
                num_samples=1,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
            )
            
        # Decode
        decoded = self.tokenizer.output_transform(generated.cpu(), scale)
        values = decoded[0, 0, :].numpy()
        
        if mode == "fast":
            return {"forecast": values[:PREDICTION_LENGTH]}
        
        # Reasoning mode: Split and de-amplify
        # Trend (no amp), Seasonal (10x), Volatility (50x)
        trend = values[0:DECOMPOSITION_LENGTH]
        seasonal = values[DECOMPOSITION_LENGTH:2*DECOMPOSITION_LENGTH] / SEASONAL_AMP
        volatility = values[2*DECOMPOSITION_LENGTH:3*DECOMPOSITION_LENGTH] / VOLATILITY_AMP
        forecast = values[3*DECOMPOSITION_LENGTH:3*DECOMPOSITION_LENGTH+PREDICTION_LENGTH]
        
        return {
            "trend": trend,
            "seasonal": seasonal,
            "volatility": volatility,
            "forecast": forecast
        }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chronos-2 Reasoning Inference")
    parser.add_argument("--model", type=str, default="output/reasoning-v4-from-amazon/", help="Path to model")
    parser.add_argument("--mode", type=str, choices=["reasoning", "fast"], default="reasoning", help="Inference mode")
    args = parser.parse_args()
    
    # Example usage with synthetic data
    t = np.linspace(0, 10 * np.pi, 200)
    series = np.sin(t) + 0.1 * t + np.random.normal(0, 0.05, 200)
    
    engine = ChronosReasoningInference(args.model)
    results = engine.forecast(series, mode=args.mode)
    
    print(f"\nResults for mode: {args.mode}")
    print("-" * 30)
    for key, val in results.items():
        print(f"{key:10} | length: {len(val)} | mean: {np.mean(val):.4f}")

if __name__ == "__main__":
    main()
