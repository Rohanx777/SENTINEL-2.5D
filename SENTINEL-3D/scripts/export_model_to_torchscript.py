#!/usr/bin/env python3
"""
export_model_to_torchscript.py
-------------------------------
Export the Cylinder3D PyTorch model to TorchScript format for C++ LibTorch.

Usage:
    python export_model_to_torchscript.py

Output:
    weights/model_load.pt (TorchScript model)
"""

import sys
import torch
from pathlib import Path

# Import the model definition from cylinder3d_net.py
from cylinder3d_net import build_model, GRID_SIZE

WEIGHTS_IN = "weights/model_load.pth"
WEIGHTS_OUT = "weights/model_load.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def export_to_torchscript():
    print(f"[1/3] Loading PyTorch model from {WEIGHTS_IN}...")
    model = build_model(WEIGHTS_IN, DEVICE)
    model.eval()

    print(f"[2/3] Tracing model with example inputs...")
    # Create dummy inputs matching the model's expected format
    dummy_pt_fea = torch.randn(1000, 9, device=DEVICE)      # (N, 9) features
    dummy_grid_ind = torch.randint(0, 100, (1000, 3), device=DEVICE).int()  # (N, 3) indices

    # Trace the model
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(
                model,
                ([dummy_pt_fea], [dummy_grid_ind], 1)  # batch_size=1
            )
        print("[2/3] ✓ Model traced successfully")
    except Exception as e:
        print(f"[ERROR] Tracing failed: {e}")
        print("Trying torch.jit.script instead...")
        try:
            traced_model = torch.jit.script(model)
            print("[2/3] ✓ Model scripted successfully")
        except Exception as e2:
            print(f"[ERROR] Scripting also failed: {e2}")
            print("\nThe Cylinder3D model uses spconv, which may not support TorchScript directly.")
            print("Alternative approaches:")
            print("  1. Use ONNX export → TensorRT")
            print("  2. Keep PyTorch inference in Python and only migrate voxelizer + grid builder to C++/CUDA")
            print("  3. Reimplement the spconv operations in LibTorch manually")
            sys.exit(1)

    print(f"[3/3] Saving TorchScript model to {WEIGHTS_OUT}...")
    traced_model.save(WEIGHTS_OUT)
    print(f"[3/3] ✓ Model saved successfully")

    # Verify the saved model loads correctly
    print("\n[Verification] Loading saved model...")
    loaded_model = torch.jit.load(WEIGHTS_OUT, map_location=DEVICE)
    print("[Verification] ✓ Model loads successfully in LibTorch-compatible format")

    # Test inference
    print("[Verification] Running test inference...")
    with torch.no_grad():
        output = loaded_model([dummy_pt_fea], [dummy_grid_ind], 1)
    print(f"[Verification] ✓ Output shape: {output.shape}")

    print("\n" + "="*60)
    print("SUCCESS! TorchScript model ready for C++ LibTorch backend.")
    print(f"Model saved at: {Path(WEIGHTS_OUT).resolve()}")
    print("="*60)

if __name__ == "__main__":
    # Check if input weights exist
    if not Path(WEIGHTS_IN).exists():
        print(f"[ERROR] Weights file not found: {WEIGHTS_IN}")
        print("Make sure you have downloaded the Cylinder3D checkpoint.")
        sys.exit(1)

    export_to_torchscript()
