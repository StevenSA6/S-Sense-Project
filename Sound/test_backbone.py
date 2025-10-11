"""
Test script for PANN backbone integration.
This script tests loading of pretrained backbones without full training.
"""
import torch
from omegaconf import OmegaConf
from src.models import build_model


def test_backbone_loading(backbone_type):
  """Test loading a specific backbone type."""
  print(f"\n{'='*60}")
  print(f"Testing backbone: {backbone_type}")
  print(f"{'='*60}")

  # Create minimal config with backbone enabled
  cfg = OmegaConf.create({
      "features": {
          "type": "mel",
          "deltas": {"enabled": False},
          "aux_channels": {"enabled": False}
      },
      "model": {
          "variant": "crnn",
          "in_channels": 1,
          "hidden": 128,
          "dropout": 0.1,
          "norm": "batch",
          "se_blocks": False,
          "backbone": {
              "enabled": True,
              "type": backbone_type,
              "checkpoint_path": "./backbones/",
              "freeze": True
          },
          "rnn": {
              "type": "lstm",
              "layers": 1,
              "bidirectional": True,
              "hidden": 128
          },
          "heads": {
              "sed": {"enabled": True, "activation": "sigmoid"},
              "count": {"enabled": False}
          }
      }
  })

  # Build model
  try:
    model, backbone = build_model(cfg, return_backbone=True)
    print(f"✓ Model built successfully")

    # Check if backbone was loaded
    if backbone is not None:
      print(f"✓ Backbone loaded: {type(backbone).__name__}")

      # Check that model actually uses the backbone
      if hasattr(model, 'use_backbone') and model.use_backbone:
        print(f"✓ Model is configured to use backbone")
      else:
        raise ValueError("Model was built but not configured to use backbone!")

      # Check frozen parameters
      frozen_params = sum(1 for p in backbone.parameters()
                          if not p.requires_grad)
      total_params = sum(1 for p in backbone.parameters())
      print(f"✓ Frozen parameters: {frozen_params}/{total_params}")

      # Test forward pass through the full model with waveform
      print("\nTesting model forward pass with backbone...")
      dummy_wave = torch.randn(2, 16000)  # batch=2, 1 second at 16kHz
      dummy_spec = torch.randn(2, 1, 64, 101)  # dummy spec (not used)
      with torch.no_grad():
        output = model(dummy_spec, waveform=dummy_wave)
      print(f"✓ Model output keys: {output.keys()}")
      print(f"✓ Model logits shape: {output['logits'].shape}")

    print(f"\n{'='*60}")
    print(f"✓ {backbone_type} test PASSED")
    print(f"{'='*60}\n")
    return True

  except Exception as e:
    print(f"\n{'='*60}")
    print(f"✗ {backbone_type} test FAILED")
    print(f"Error: {e}")
    print(f"{'='*60}\n")
    import traceback
    traceback.print_exc()
    return False


def test_no_backbone():
  """Test that model works without backbone (baseline)."""
  print(f"\n{'='*60}")
  print(f"Testing baseline model (no backbone)")
  print(f"{'='*60}")

  cfg = OmegaConf.create({
      "features": {
          "type": "mel",
          "deltas": {"enabled": False},
          "aux_channels": {"enabled": False}
      },
      "model": {
          "variant": "crnn",
          "in_channels": 1,
          "hidden": 128,
          "dropout": 0.1,
          "norm": "batch",
          "se_blocks": False,
          "backbone": {"enabled": False},
          "rnn": {
              "type": "lstm",
              "layers": 1,
              "bidirectional": True,
              "hidden": 128
          },
          "heads": {
              "sed": {"enabled": True, "activation": "sigmoid"},
              "count": {"enabled": False}
          }
      }
  })

  try:
    model = build_model(cfg)
    print(f"✓ Baseline model built successfully")
    print(f"✓ No backbone loaded (as expected)")
    print(f"\n{'='*60}")
    print(f"✓ Baseline test PASSED")
    print(f"{'='*60}\n")
    return True

  except Exception as e:
    print(f"\n{'='*60}")
    print(f"✗ Baseline test FAILED")
    print(f"Error: {e}")
    print(f"{'='*60}\n")
    import traceback
    traceback.print_exc()
    return False


if __name__ == "__main__":
  print("\n" + "="*60)
  print("PANN BACKBONE INTEGRATION TEST")
  print("="*60)

  results = {}

  # Test baseline (no backbone)
  results["baseline"] = test_no_backbone()

  # Test Wavegram_Cnn14
  results["wavegram_cnn14"] = test_backbone_loading("wavegram_cnn14")

  # Test Wavegram_Logmel_Cnn14
  results["wavegram_logmel_cnn14"] = test_backbone_loading(
      "wavegram_logmel_cnn14")

  # Summary
  print("\n" + "="*60)
  print("TEST SUMMARY")
  print("="*60)
  for name, passed in results.items():
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"{name:30s}: {status}")
  print("="*60)

  all_passed = all(results.values())
  if all_passed:
    print("\n✓ All tests passed!")
  else:
    print("\n✗ Some tests failed")
    exit(1)
