#!/usr/bin/env python3
"""
Debug script to analyze prediction quality and identify issues.
"""

import torch
import numpy as np
from pathlib import Path

# Load saved predictions
pred_path = Path("./evaluation_results/predictions.pt")
data = torch.load(pred_path, weights_only=False)

predictions = data["predictions"]
targets = data["targets"]
inputs = data["inputs"]

print("=" * 60)
print("PREDICTION ANALYSIS")
print("=" * 60)

print(f"\nShapes:")
print(f"  Predictions: {predictions.shape}")
print(f"  Targets: {targets.shape}")
print(f"  Inputs: {inputs.shape}")

# Check value ranges per channel
field_names = ["density", "pressure", "velocity_x", "velocity_y"]

print(f"\n{'Field':<15} {'Pred Min':>12} {'Pred Max':>12} {'Pred Mean':>12} {'Pred Std':>12}")
print("-" * 65)
for i, name in enumerate(field_names):
    pred_ch = predictions[..., i]
    print(f"{name:<15} {pred_ch.min().item():>12.4f} {pred_ch.max().item():>12.4f} {pred_ch.mean().item():>12.4f} {pred_ch.std().item():>12.4f}")

print(f"\n{'Field':<15} {'Target Min':>12} {'Target Max':>12} {'Target Mean':>12} {'Target Std':>12}")
print("-" * 65)
for i, name in enumerate(field_names):
    target_ch = targets[..., i]
    print(f"{name:<15} {target_ch.min().item():>12.4f} {target_ch.max().item():>12.4f} {target_ch.mean().item():>12.4f} {target_ch.std().item():>12.4f}")

print(f"\n{'Field':<15} {'Input Min':>12} {'Input Max':>12} {'Input Mean':>12} {'Input Std':>12}")
print("-" * 65)
for i, name in enumerate(field_names):
    input_ch = inputs[..., i]
    print(f"{name:<15} {input_ch.min().item():>12.4f} {input_ch.max().item():>12.4f} {input_ch.mean().item():>12.4f} {input_ch.std().item():>12.4f}")

# Check per-field MSE
print(f"\n{'Field':<15} {'MSE':>15} {'% of Total':>12}")
print("-" * 45)
total_mse = 0
field_mses = []
for i, name in enumerate(field_names):
    mse = ((predictions[..., i] - targets[..., i]) ** 2).mean().item()
    field_mses.append(mse)
    total_mse += mse

for i, name in enumerate(field_names):
    pct = (field_mses[i] / total_mse) * 100 if total_mse > 0 else 0
    print(f"{name:<15} {field_mses[i]:>15.4f} {pct:>11.1f}%")

print(f"\n{'Total':<15} {total_mse:>15.4f}")

# Check if predictions are just noise or have structure
print("\n" + "=" * 60)
print("TEMPORAL CONSISTENCY CHECK")
print("=" * 60)

# Check if consecutive predicted frames are similar
if predictions.shape[1] > 1:
    frame_diffs = []
    for t in range(predictions.shape[1] - 1):
        diff = ((predictions[:, t] - predictions[:, t+1]) ** 2).mean().item()
        frame_diffs.append(diff)
    print(f"\nMean frame-to-frame difference in predictions: {np.mean(frame_diffs):.4f}")
    
    # Same for targets
    target_diffs = []
    for t in range(targets.shape[1] - 1):
        diff = ((targets[:, t] - targets[:, t+1]) ** 2).mean().item()
        target_diffs.append(diff)
    print(f"Mean frame-to-frame difference in targets: {np.mean(target_diffs):.4f}")

# Check correlation between predictions and targets
print("\n" + "=" * 60)
print("CORRELATION CHECK")
print("=" * 60)
for i, name in enumerate(field_names):
    pred_flat = predictions[..., i].flatten().numpy()
    target_flat = targets[..., i].flatten().numpy()
    
    # Compute correlation
    if pred_flat.std() > 0 and target_flat.std() > 0:
        corr = np.corrcoef(pred_flat, target_flat)[0, 1]
        print(f"{name}: correlation = {corr:.4f}")
    else:
        print(f"{name}: unable to compute correlation (constant values)")
