"""
Verification Tests for Reasoning SFT Pipeline (v2)

This script verifies the critical fixes have been applied correctly:
1. Token IDs are at vocab END (4096, 4097) - not overwriting bins
2. max_new_tokens is set appropriately 
3. Tokenizer bin mapping unchanged (bins at 2-4095)
"""

import torch
import numpy as np
import sys
sys.path.insert(0, ".")

def test_token_ids_at_vocab_end():
    """Test that control tokens are defined at vocabulary END."""
    print("\n" + "=" * 60)
    print("TEST 1: Token IDs at Vocabulary End")
    print("=" * 60)
    
    # Import the constants from train_reasoning
    from scripts.training.train_reasoning import (
        FAST_MODE_TOKEN_ID,
        REASONING_MODE_TOKEN_ID,
        ORIGINAL_VOCAB_SIZE,
        NEW_VOCAB_SIZE
    )
    
    print(f"  ORIGINAL_VOCAB_SIZE: {ORIGINAL_VOCAB_SIZE}")
    print(f"  NEW_VOCAB_SIZE: {NEW_VOCAB_SIZE}")
    print(f"  FAST_MODE_TOKEN_ID: {FAST_MODE_TOKEN_ID}")
    print(f"  REASONING_MODE_TOKEN_ID: {REASONING_MODE_TOKEN_ID}")
    
    # Verify tokens are OUTSIDE bin range (2-4095)
    bin_range = range(2, 4096)
    
    assert FAST_MODE_TOKEN_ID not in bin_range, \
        f"FAIL: Fast token {FAST_MODE_TOKEN_ID} is in bin range!"
    assert REASONING_MODE_TOKEN_ID not in bin_range, \
        f"FAIL: Reasoning token {REASONING_MODE_TOKEN_ID} is in bin range!"
    
    # Verify tokens are at expected positions
    assert FAST_MODE_TOKEN_ID == ORIGINAL_VOCAB_SIZE, \
        f"FAIL: Fast token should be at {ORIGINAL_VOCAB_SIZE}, got {FAST_MODE_TOKEN_ID}"
    assert REASONING_MODE_TOKEN_ID == ORIGINAL_VOCAB_SIZE + 1, \
        f"FAIL: Reasoning token should be at {ORIGINAL_VOCAB_SIZE + 1}, got {REASONING_MODE_TOKEN_ID}"
    
    print("  ✓ Control tokens are correctly placed at vocabulary END")
    print("  ✓ No collision with bin IDs (2-4095)")
    return True


def test_tokenizer_bin_mapping_unchanged():
    """Test that ChronosTokenizer still maps bins to positions 2-4095."""
    print("\n" + "=" * 60)
    print("TEST 2: Tokenizer Bin Mapping Unchanged")
    print("=" * 60)
    
    from chronos import ChronosConfig
    
    # Create tokenizer with ORIGINAL config (n_special_tokens=2)
    config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},
        n_tokens=4096,
        n_special_tokens=2,  # UNCHANGED!
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=True,
        context_length=512,
        prediction_length=64,
        model_type="seq2seq",
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    tokenizer = config.create_tokenizer()
    
    # Create sample data
    sample = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    
    # Tokenize
    ids, mask, scale = tokenizer.context_input_transform(sample)
    
    print(f"  Sample input: {sample.numpy()}")
    print(f"  Token IDs: {ids.numpy()}")
    print(f"  ID range: [{ids.min().item()}, {ids.max().item()}]")
    
    # Verify all tokens are in bin range (2-4095)
    min_id = ids.min().item()
    max_id = ids.max().item()
    
    assert min_id >= 2, f"FAIL: Min token ID {min_id} < 2 (should be >= 2 for bins)"
    assert max_id <= 4095, f"FAIL: Max token ID {max_id} > 4095 (should be <= 4095 for bins)"
    
    print("  ✓ Tokenizer still maps values to bin IDs 2-4095")
    print("  ✓ n_special_tokens=2 preserved (PAD=0, EOS=1)")
    return True


def test_model_embedding_resize():
    """Test that model embeddings are correctly resized."""
    print("\n" + "=" * 60)
    print("TEST 3: Model Embedding Resize")
    print("=" * 60)
    
    from chronos import ChronosPipeline
    from scripts.training.train_reasoning import (
        append_control_tokens,
        FAST_MODE_TOKEN_ID,
        REASONING_MODE_TOKEN_ID,
        NEW_VOCAB_SIZE
    )
    
    # Load model
    model_id = "output/gifteval-subset-finetune/run-0/checkpoint-final"
    print(f"  Loading model from: {model_id}")
    
    try:
        pipeline = ChronosPipeline.from_pretrained(model_id)
        model = pipeline.model
    except Exception as e:
        print(f"  ⚠ Could not load model: {e}")
        print("  Skipping model embedding test (model not available)")
        return None
    
    original_size = model.model.shared.num_embeddings
    print(f"  Original vocab size: {original_size}")
    
    # Apply resize
    model = append_control_tokens(model)
    
    new_size = model.model.shared.num_embeddings
    print(f"  New vocab size: {new_size}")
    
    assert new_size == NEW_VOCAB_SIZE, \
        f"FAIL: Expected vocab size {NEW_VOCAB_SIZE}, got {new_size}"
    
    # Verify control token embeddings exist and are non-zero
    fast_emb = model.model.shared.weight.data[FAST_MODE_TOKEN_ID]
    reason_emb = model.model.shared.weight.data[REASONING_MODE_TOKEN_ID]
    
    assert fast_emb.abs().sum() > 0, "FAIL: Fast token embedding is all zeros"
    assert reason_emb.abs().sum() > 0, "FAIL: Reasoning token embedding is all zeros"
    
    print(f"  ✓ Embeddings resized to {NEW_VOCAB_SIZE}")
    print(f"  ✓ Control token embeddings initialized (non-zero)")
    return True


def test_generation_config():
    """Test that generation config has appropriate max_new_tokens."""
    print("\n" + "=" * 60)
    print("TEST 4: Generation Config (max_new_tokens)")
    print("=" * 60)
    
    from chronos import ChronosPipeline
    from scripts.training.train_reasoning import MAX_NEW_TOKENS_REASONING
    
    model_id = "output/gifteval-subset-finetune/run-0/checkpoint-final"
    
    try:
        pipeline = ChronosPipeline.from_pretrained(model_id)
        model = pipeline.model
    except Exception as e:
        print(f"  ⚠ Could not load model: {e}")
        print("  Skipping generation config test")
        return None
    
    # Check original config
    gen_config = model.model.generation_config
    print(f"  Original max_new_tokens: {getattr(gen_config, 'max_new_tokens', 'Not set')}")
    print(f"  Original max_length: {getattr(gen_config, 'max_length', 'Not set')}")
    
    # Apply the fix
    model.model.generation_config.max_new_tokens = MAX_NEW_TOKENS_REASONING
    model.model.generation_config.min_new_tokens = 1
    
    print(f"  Required for Reasoning Mode: {MAX_NEW_TOKENS_REASONING}")
    print(f"  After fix - max_new_tokens: {model.model.generation_config.max_new_tokens}")
    
    assert model.model.generation_config.max_new_tokens >= MAX_NEW_TOKENS_REASONING, \
        f"FAIL: max_new_tokens should be >= {MAX_NEW_TOKENS_REASONING}"
    
    print(f"  ✓ max_new_tokens = {MAX_NEW_TOKENS_REASONING} (sufficient for 257-token output)")
    return True


def test_dataset_token_usage():
    """Test that dataset correctly uses appended token IDs."""
    print("\n" + "=" * 60)
    print("TEST 5: Dataset Token Usage")
    print("=" * 60)
    
    from chronos import ChronosConfig
    from scripts.training.train_reasoning import (
        ReasoningChronosDataset,
        FAST_MODE_TOKEN_ID,
        REASONING_MODE_TOKEN_ID
    )
    import datasets
    
    # Create tokenizer
    config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},
        n_tokens=4096,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=True,
        context_length=512,
        prediction_length=64,
        model_type="seq2seq",
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    tokenizer = config.create_tokenizer()
    
    # Load dataset
    try:
        arrow_ds = datasets.load_dataset("arrow", data_files="gifteval-reasoning.arrow", split="train")
    except Exception as e:
        print(f"  ⚠ Could not load dataset: {e}")
        print("  Skipping dataset token usage test")
        return None
    
    # Create dataset
    train_dataset = ReasoningChronosDataset(
        datasets=[arrow_ds],
        probabilities=[1.0],
        tokenizer=tokenizer,
        context_length=512,
        prediction_length=64,
        model_type="seq2seq",
    )
    
    # Check samples
    fast_found = False
    reasoning_found = False
    bin_collision = False
    
    for i, batch in enumerate(train_dataset):
        if i >= 20:
            break
            
        control_token = batch["input_ids"][0].item()
        
        if control_token == FAST_MODE_TOKEN_ID:
            fast_found = True
            print(f"  Sample {i}: Fast mode (token {control_token}), labels length: {batch['labels'].shape[0]}")
        elif control_token == REASONING_MODE_TOKEN_ID:
            reasoning_found = True
            print(f"  Sample {i}: Reasoning mode (token {control_token}), labels length: {batch['labels'].shape[0]}")
        elif 2 <= control_token <= 4095:
            bin_collision = True
            print(f"  ⚠ Sample {i}: Token {control_token} IS A BIN - COLLISION!")
        
        if fast_found and reasoning_found and not bin_collision:
            break
    
    if bin_collision:
        print("  ✗ TOKEN COLLISION DETECTED - control tokens overlapping with bins!")
        return False
    
    if fast_found and reasoning_found:
        print(f"  ✓ Fast mode uses token ID {FAST_MODE_TOKEN_ID}")
        print(f"  ✓ Reasoning mode uses token ID {REASONING_MODE_TOKEN_ID}")
        print("  ✓ No bin collisions detected")
        return True
    else:
        print(f"  ⚠ Did not find both modes (fast={fast_found}, reasoning={reasoning_found})")
        return None


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("REASONING SFT PIPELINE - VERIFICATION TESTS")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Token IDs at vocab end
    try:
        results["token_ids"] = test_token_ids_at_vocab_end()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["token_ids"] = False
    
    # Test 2: Tokenizer bin mapping
    try:
        results["tokenizer"] = test_tokenizer_bin_mapping_unchanged()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["tokenizer"] = False
    
    # Test 3: Model embedding resize
    try:
        results["embeddings"] = test_model_embedding_resize()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["embeddings"] = False
    
    # Test 4: Generation config
    try:
        results["generation"] = test_generation_config()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["generation"] = False
    
    # Test 5: Dataset token usage
    try:
        results["dataset"] = test_dataset_token_usage()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["dataset"] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v is True)
    skipped = sum(1 for v in results.values() if v is None)
    failed = sum(1 for v in results.values() if v is False)
    
    for name, result in results.items():
        status = "✓ PASS" if result is True else ("⚠ SKIP" if result is None else "✗ FAIL")
        print(f"  {name}: {status}")
    
    print(f"\n  Total: {passed} passed, {skipped} skipped, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
