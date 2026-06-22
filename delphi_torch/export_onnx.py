#!/usr/bin/env python3
"""Export Delphi model to ONNX format.

Usage:
    python export_onnx.py --checkpoint Delphi-2M/ckpt.pt --output delphi.onnx

The exported model takes two inputs:
    - idx: int64 tensor of shape (batch_size, seq_len) - token IDs
    - age: float32 tensor of shape (batch_size, seq_len) - age in days

And produces:
    - logits: float32 tensor of shape (batch_size, seq_len, vocab_size)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from delphi_torch.config import ModelConfig
from delphi_torch.models.delphi import DelphiModel


class DelphiForExport(nn.Module):
    """Wrapper for ONNX export that simplifies the forward signature."""

    def __init__(self, model: DelphiModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        """Forward pass for inference (no targets_age, no attention return)."""
        logits, _, _ = self.model(idx, age, targets_age=None, return_attn=False)
        return logits


def load_checkpoint(ckpt_path: Path) -> tuple[DelphiModel, dict]:
    """Load model from checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Extract config from checkpoint
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        # Handle both old-style flat config and new-style nested config
        if "model" in config_dict:
            model_cfg = ModelConfig(**config_dict["model"])
        else:
            # Old-style config - extract model parameters
            model_cfg = ModelConfig(
                block_size=config_dict.get("block_size", 48),
                n_layer=config_dict.get("n_layer", 12),
                n_head=config_dict.get("n_head", 12),
                d_model=config_dict.get("n_embd", 120),
                dropout=0.0,  # No dropout for inference
                token_dropout=0.0,
                bias=config_dict.get("bias", False),
                vocab_size=config_dict.get("vocab_size", 1270),
                t_min=config_dict.get("t_min", 0.1),
                mask_ties=config_dict.get("mask_ties", True),
                ignore_tokens=config_dict.get("ignore_tokens", [0]),
            )
    elif "model_args" in checkpoint:
        # Very old style checkpoint
        args = checkpoint["model_args"]
        model_cfg = ModelConfig(
            block_size=args.get("block_size", 48),
            n_layer=args.get("n_layer", 12),
            n_head=args.get("n_head", 12),
            d_model=args.get("n_embd", 120),
            dropout=0.0,
            token_dropout=0.0,
            bias=args.get("bias", False),
            vocab_size=args.get("vocab_size", 1270),
            t_min=args.get("t_min", 0.1),
            mask_ties=args.get("mask_ties", True),
            ignore_tokens=args.get("ignore_tokens", [0]),
        )
    else:
        raise ValueError("Cannot find config in checkpoint")

    # Create and load model
    model = DelphiModel(model_cfg)
    
    # Load state dict, handling potential prefix issues
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, {"config": model_cfg}


def export_onnx(
    model: DelphiModel,
    output_path: Path,
    *,
    batch_size: int = 1,
    seq_len: int = 48,
    opset_version: int = 17,
    dynamic_axes: bool = True,
) -> None:
    """Export model to ONNX format.
    
    Args:
        model: The Delphi model to export
        output_path: Path for the output .onnx file
        batch_size: Batch size for the example input
        seq_len: Sequence length for the example input
        opset_version: ONNX opset version
        dynamic_axes: Whether to enable dynamic batch/sequence dimensions
    """
    # Wrap model for simpler export signature
    export_model = DelphiForExport(model)
    export_model.eval()
    
    # Create example inputs
    idx = torch.randint(0, model.cfg.vocab_size, (batch_size, seq_len), dtype=torch.int64)
    age = torch.rand(batch_size, seq_len, dtype=torch.float32) * 36525  # 0-100 years in days
    
    # Define dynamic axes for variable batch/seq length
    if dynamic_axes:
        dynamic_axes_dict = {
            "idx": {0: "batch_size", 1: "seq_len"},
            "age": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        }
    else:
        dynamic_axes_dict = None
    
    # Export
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        export_model,
        (idx, age),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["idx", "age"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes_dict,
    )
    print(f"✓ Exported to {output_path}")
    
    # Verify the export
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validation passed")
    except ImportError:
        print("⚠ Install 'onnx' package to validate: pip install onnx")
    except Exception as e:
        print(f"⚠ ONNX validation warning: {e}")
    
    # Print model info
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\nModel info:")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  Vocab size: {model.cfg.vocab_size}")
    print(f"  Embedding dim: {model.cfg.d_model}")
    print(f"  Layers: {model.cfg.n_layer}")
    print(f"  Heads: {model.cfg.n_head}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Delphi model to ONNX")
    parser.add_argument(
        "--checkpoint", "-c",
        type=Path,
        required=True,
        help="Path to checkpoint file (ckpt.pt)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("delphi.onnx"),
        help="Output ONNX file path (default: delphi.onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--no-dynamic",
        action="store_true",
        help="Disable dynamic axes (fixed batch/seq dimensions)",
    )
    args = parser.parse_args()
    
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Load model
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, metadata = load_checkpoint(args.checkpoint)
    print(f"✓ Loaded model: {metadata['config'].n_layer}L-{metadata['config'].n_head}H-{metadata['config'].d_model}E")
    
    # Export
    export_onnx(
        model,
        args.output,
        opset_version=args.opset,
        dynamic_axes=not args.no_dynamic,
    )
    
    print("\n" + "=" * 50)
    print("ONNX export complete!")
    print("=" * 50)
    print(f"\nTo use in Python with ONNX Runtime:")
    print("""
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("delphi.onnx")

# Prepare inputs (example)
idx = np.array([[1, 5, 10, 15]], dtype=np.int64)  # token IDs
age = np.array([[0, 365, 730, 1095]], dtype=np.float32)  # ages in days

# Run inference
logits = session.run(["logits"], {"idx": idx, "age": age})[0]
print(logits.shape)  # (1, 4, vocab_size)
""")


if __name__ == "__main__":
    main()
