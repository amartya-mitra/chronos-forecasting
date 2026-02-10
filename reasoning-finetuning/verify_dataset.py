#!/usr/bin/env python3
"""
Verify Reasoning Dataset Integrity

Comprehensive checks on the reasoning finetuning dataset:
1. Component ordering: trend → seasonal → volatility → forecast
2. Tokenize→detokenize roundtrip fidelity
3. Decomposition matches independent STL on the context
4. Token count structure (64+64+64+64=256 for reasoning, 64 for fast)
5. Scale consistency: context_scale matches recomputed scale

Usage:
    python verify_dataset.py                                  # verify GiftEval
    python verify_dataset.py --dataset sarsim0-reasoning.arrow  # verify SarSim0
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import datasets
from statsmodels.tsa.seasonal import STL
import pandas as pd

from chronos import ChronosConfig
from prepare_dataset import (
    CONTEXT_LENGTH,
    PREDICTION_LENGTH,
    DECOMPOSITION_LENGTH,
    SEASONAL_AMP,
    VOLATILITY_AMP,
    compute_decomposition,
)

# Thresholds for violation detection
ROUNDTRIP_CORR_THRESHOLD = 0.95   # tokenize→detokenize should be high fidelity
DECOMP_CORR_THRESHOLD = 0.85      # vs independent STL (allowing for quantization)
VOLATILITY_CORR_THRESHOLD = 0.60  # volatility is noisier, more lenient
SCALE_REL_TOL = 0.01              # context scale relative tolerance


def create_tokenizer():
    """Create the same tokenizer used during dataset preparation."""
    config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},
        n_tokens=4096,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=True,
        model_type="seq2seq",
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    return config.create_tokenizer()


def detokenize(token_ids: list, scale: float, tokenizer) -> np.ndarray:
    """Convert token IDs back to values using the tokenizer's output_transform."""
    tokens_tensor = torch.tensor([token_ids]).long()
    scale_tensor = torch.tensor([scale]).float()
    values = tokenizer.output_transform(tokens_tensor, scale_tensor)
    return values[0, 0].numpy()  # shape: (num_samples=1, seq_len)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute correlation, returning 0 if either array is constant."""
    if len(a) != len(b):
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


class DatasetVerifier:
    """Runs all verification checks on a reasoning dataset."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.tokenizer = create_tokenizer()
        self.violations = []
        self.source = None  # auto-detected: "gifteval" or "sarsim0"
        self.stats = {
            "total": 0,
            "fast_mode": 0,
            "reasoning_mode": 0,
        }
        self.check_results = {
            "token_count": {"pass": 0, "fail": 0, "details": []},
            "roundtrip": {"pass": 0, "fail": 0, "details": []},
            "component_order": {"pass": 0, "fail": 0, "details": []},
            "decomp_match": {"pass": 0, "fail": 0, "details": []},
            "scale_consistency": {"pass": 0, "fail": 0, "details": []},
        }

    def _detect_source(self, ds):
        """Auto-detect dataset source by checking if samples have stored components."""
        sample = ds[0]
        has_stored = all(k in sample for k in ("trend", "seasonal", "volatility"))
        self.source = "sarsim0" if has_stored else "gifteval"
        print(f"  Detected source: {self.source}"
              f" ({'stored components' if has_stored else 'STL recomputation'})")

    def _get_ground_truth(self, sample: dict, context: np.ndarray) -> dict:
        """
        Get ground truth decomposition based on dataset source.
        
        For GiftEval: recompute via STL from context
        For SarSim0 : use stored components from the sample
        """
        if self.source == "sarsim0":
            return {
                "trend": np.array(sample["trend"]),
                "seasonal": np.array(sample["seasonal"]),
                "volatility": np.array(sample["volatility"]),
            }
        else:
            return compute_decomposition(context, period=7)

    def load_dataset(self):
        """Load the arrow dataset."""
        print(f"Loading dataset: {self.dataset_path}")
        self.ds = datasets.load_dataset(
            "arrow", data_files=str(self.dataset_path), split="train"
        )
        print(f"  Loaded {len(self.ds)} samples")
        self._detect_source(self.ds)
        return self.ds

    def verify_sample(self, idx: int, sample: dict):
        """Run all checks on a single sample."""
        self.stats["total"] += 1
        mode = sample.get("mode", "unknown")

        if mode == "fast":
            self.stats["fast_mode"] += 1
            self._check_fast_sample(idx, sample)
        elif mode == "reasoning":
            self.stats["reasoning_mode"] += 1
            self._check_reasoning_sample(idx, sample)
        else:
            self._add_violation(idx, "unknown_mode", f"Unknown mode: {mode}")

    def _check_fast_sample(self, idx: int, sample: dict):
        """Verify a fast-mode sample."""
        tokens = sample["reasoning_tokens"]

        # Check 1: Token count = 64
        if len(tokens) != PREDICTION_LENGTH:
            self._add_violation(
                idx, "token_count",
                f"Fast mode: expected {PREDICTION_LENGTH} tokens, got {len(tokens)}"
            )
        else:
            self.check_results["token_count"]["pass"] += 1

    def _check_reasoning_sample(self, idx: int, sample: dict):
        """Verify a reasoning-mode sample with all checks."""
        tokens = sample["reasoning_tokens"]
        target = np.array(sample["target"])

        # --- Check 1: Token count = 256 ---
        expected_len = DECOMPOSITION_LENGTH * 3 + PREDICTION_LENGTH  # 256
        if len(tokens) != expected_len:
            self._add_check_fail(
                idx, "token_count",
                f"Expected {expected_len} tokens, got {len(tokens)}"
            )
            return  # Can't proceed with wrong token count
        else:
            self.check_results["token_count"]["pass"] += 1

        # --- Extract context and future from target ---
        total_needed = CONTEXT_LENGTH + PREDICTION_LENGTH
        if len(target) >= total_needed:
            context = target[-(total_needed):-PREDICTION_LENGTH]
            future = target[-PREDICTION_LENGTH:]
        else:
            split = len(target) - PREDICTION_LENGTH
            context = target[:split]
            future = target[split:]

        # --- Check 2: Scale consistency ---
        context_tensor = torch.tensor(context).float().unsqueeze(0)
        _, _, recomputed_scale = self.tokenizer.context_input_transform(context_tensor)
        recomputed_scale_val = float(recomputed_scale.item())

        stored_scale = sample.get("context_scale", None)
        if stored_scale is not None:
            rel_diff = abs(stored_scale - recomputed_scale_val) / max(recomputed_scale_val, 1e-10)
            if rel_diff > SCALE_REL_TOL:
                self._add_check_fail(
                    idx, "scale_consistency",
                    f"Stored scale={stored_scale:.6f}, recomputed={recomputed_scale_val:.6f}, diff={rel_diff:.4f}"
                )
            else:
                self.check_results["scale_consistency"]["pass"] += 1
        else:
            self.check_results["scale_consistency"]["pass"] += 1

        # Use recomputed scale for de-tokenization
        scale = recomputed_scale_val

        # --- De-tokenize and split into components ---
        detok_values = detokenize(tokens, scale, self.tokenizer)

        trend_detok = detok_values[0:DECOMPOSITION_LENGTH]
        seasonal_detok = detok_values[DECOMPOSITION_LENGTH:2*DECOMPOSITION_LENGTH] / SEASONAL_AMP
        volatility_detok = detok_values[2*DECOMPOSITION_LENGTH:3*DECOMPOSITION_LENGTH] / VOLATILITY_AMP
        forecast_detok = detok_values[3*DECOMPOSITION_LENGTH:]

        # --- Get ground truth (source-aware) ---
        gt = self._get_ground_truth(sample, context)
        trend_gt = gt["trend"]
        seasonal_gt = gt["seasonal"]
        volatility_gt = gt["volatility"]

        # --- Check 3: Roundtrip fidelity ---
        # Re-tokenize using ground truth + context scale, compare tokens
        trend_amp_gt = trend_gt
        seasonal_amp_gt = seasonal_gt * SEASONAL_AMP
        volatility_amp_gt = volatility_gt * VOLATILITY_AMP
        combined_gt = np.concatenate([trend_amp_gt, seasonal_amp_gt, volatility_amp_gt, future])
        combined_tensor = torch.tensor(combined_gt).float().unsqueeze(0)
        retokens, _, _ = self.tokenizer._input_transform(combined_tensor, scale=recomputed_scale)
        retokens_list = retokens[0].numpy().tolist()

        token_match_count = sum(1 for a, b in zip(tokens, retokens_list) if a == b)
        token_match_pct = token_match_count / len(tokens) * 100

        if token_match_pct < 95:
            self._add_check_fail(
                idx, "roundtrip",
                f"Token match: {token_match_pct:.1f}% ({token_match_count}/{len(tokens)})"
            )
        else:
            self.check_results["roundtrip"]["pass"] += 1

        # --- Check 4: Component ordering ---
        corr_trend_trend = safe_corr(trend_detok, trend_gt)
        corr_trend_seasonal = safe_corr(trend_detok, seasonal_gt)
        corr_seasonal_seasonal = safe_corr(seasonal_detok, seasonal_gt)
        corr_seasonal_trend = safe_corr(seasonal_detok, trend_gt)

        order_ok = True
        if corr_trend_seasonal > corr_trend_trend + 0.1:
            order_ok = False
            self._add_check_fail(
                idx, "component_order",
                f"Slot 0 matches seasonal better ({corr_trend_seasonal:+.3f}) than trend ({corr_trend_trend:+.3f}) → likely swapped"
            )
        if corr_seasonal_trend > corr_seasonal_seasonal + 0.1:
            order_ok = False
            self._add_check_fail(
                idx, "component_order",
                f"Slot 1 matches trend better ({corr_seasonal_trend:+.3f}) than seasonal ({corr_seasonal_seasonal:+.3f}) → likely swapped"
            )
        if order_ok:
            self.check_results["component_order"]["pass"] += 1

        # --- Check 5: Decomposition match ---
        corr_forecast = safe_corr(forecast_detok, future)

        decomp_ok = True
        if corr_trend_trend < DECOMP_CORR_THRESHOLD:
            decomp_ok = False
            self._add_check_fail(
                idx, "decomp_match",
                f"Trend corr={corr_trend_trend:+.3f} < {DECOMP_CORR_THRESHOLD}"
            )
        if corr_seasonal_seasonal < DECOMP_CORR_THRESHOLD:
            decomp_ok = False
            self._add_check_fail(
                idx, "decomp_match",
                f"Seasonal corr={corr_seasonal_seasonal:+.3f} < {DECOMP_CORR_THRESHOLD}"
            )
        corr_vol = safe_corr(volatility_detok, volatility_gt)
        if corr_vol < VOLATILITY_CORR_THRESHOLD:
            decomp_ok = False
            self._add_check_fail(
                idx, "decomp_match",
                f"Volatility corr={corr_vol:+.3f} < {VOLATILITY_CORR_THRESHOLD}"
            )
        if corr_forecast < DECOMP_CORR_THRESHOLD:
            decomp_ok = False
            self._add_check_fail(
                idx, "decomp_match",
                f"Forecast corr={corr_forecast:+.3f} < {DECOMP_CORR_THRESHOLD}"
            )
        if decomp_ok:
            self.check_results["decomp_match"]["pass"] += 1

    def _add_check_fail(self, idx: int, check_name: str, detail: str):
        """Record a check failure."""
        self.check_results[check_name]["fail"] += 1
        self.check_results[check_name]["details"].append(f"  Sample {idx}: {detail}")
        self.violations.append((idx, check_name, detail))

    def _add_violation(self, idx: int, check_name: str, detail: str):
        """Legacy violation adder."""
        self._add_check_fail(idx, check_name, detail)

    def run_all(self):
        """Run verification on all samples."""
        ds = self.load_dataset()

        print(f"\nRunning {5} checks on {len(ds)} samples...")
        print("-" * 60)

        for i, sample in enumerate(ds):
            if i % 50 == 0 and i > 0:
                print(f"  Processed {i}/{len(ds)} samples...")
            self.verify_sample(i, sample)

        self._print_report()

    def _print_report(self):
        """Print comprehensive verification report."""
        print("\n" + "=" * 60)
        print("DATASET VERIFICATION REPORT")
        print("=" * 60)

        print(f"\nDataset: {self.dataset_path.name}")
        print(f"Total samples: {self.stats['total']}")
        print(f"  Fast mode:      {self.stats['fast_mode']}")
        print(f"  Reasoning mode: {self.stats['reasoning_mode']}")

        print(f"\n{'Check':<25} {'Pass':>8} {'Fail':>8} {'Rate':>8}")
        print("-" * 53)

        all_pass = True
        for check_name, results in self.check_results.items():
            total = results["pass"] + results["fail"]
            rate = results["pass"] / total * 100 if total > 0 else 0
            status = "✅" if results["fail"] == 0 else "❌"
            if results["fail"] > 0:
                all_pass = False
            print(f"  {status} {check_name:<22} {results['pass']:>6} {results['fail']:>6} {rate:>7.1f}%")

        # Print violation details
        if self.violations:
            print(f"\n{'='*60}")
            print(f"VIOLATIONS ({len(self.violations)} total)")
            print(f"{'='*60}")

            for check_name, results in self.check_results.items():
                if results["fail"] > 0:
                    print(f"\n[{check_name}] — {results['fail']} failures:")
                    # Show first 5 details max
                    for detail in results["details"][:5]:
                        print(detail)
                    if len(results["details"]) > 5:
                        print(f"  ... and {len(results['details']) - 5} more")

        print(f"\n{'='*60}")
        if all_pass:
            print("✅ ALL CHECKS PASSED")
        else:
            print(f"❌ {len(self.violations)} VIOLATIONS FOUND")
        print(f"{'='*60}")


def main(dataset: str = "gifteval-reasoning.arrow"):
    """
    Verify a reasoning dataset.

    Args:
        dataset: Filename of the dataset in data/ directory
    """
    data_dir = Path(__file__).parent / "data"
    dataset_path = data_dir / dataset

    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        print(f"Available files in {data_dir}:")
        if data_dir.exists():
            for f in data_dir.iterdir():
                print(f"  {f.name}")
        return

    verifier = DatasetVerifier(str(dataset_path))
    verifier.run_all()


if __name__ == "__main__":
    import fire
    fire.Fire(main)
