#!/usr/bin/env python3
"""
Compute per-frame VRMSE from saved single-shot predictions.
No additional denoising required - uses cached predictions.
"""

import torch
import numpy as np
from pathlib import Path
import sys

def compute_vrmse_per_channel(pred, target):
    """Compute VRMSE for each channel."""
    # pred, target: (B, T, H, W, C)
    errors = []
    for c in range(pred.shape[-1]):
        p = pred[..., c].flatten()
        t = target[..., c].flatten()
        var = t.var()
        if var > 1e-8:
            rmse = ((p - t) ** 2).mean().sqrt()
            vrmse = rmse / var.sqrt()
        else:
            vrmse = torch.tensor(float('nan'))
        errors.append(vrmse.item())
    return errors

# Load predictions
pred_path = Path("./evaluation_results/predictions.pt")
if not pred_path.exists():
    print(f"Error: {pred_path} not found. Run evaluate.py first.")
    sys.exit(1)

data = torch.load(pred_path, weights_only=False)
predictions = data["predictions"]  # (B, T, H, W, C)
targets = data["targets"]          # (B, T, H, W, C)

B, T_pred, H, W, C = predictions.shape
T_target = targets.shape[1]
T = min(T_pred, T_target)

print("=" * 70)
print("PER-FRAME VRMSE (from single-shot prediction, no autoregressive)")
print("=" * 70)
print(f"\nSamples: {B}, Frames evaluated: {T}")
print(f"Prediction shape: {predictions.shape}")
print(f"Target shape: {targets.shape}")

field_names = ["density", "pressure", "velocity_x", "velocity_y"]

# Per-frame VRMSE
print("\n" + "-" * 70)
print(f"{'Frame':<8}", end="")
for name in field_names:
    print(f"{name:<12}", end="")
print(f"{'Mean':<12}")
print("-" * 70)

all_frame_vrmse = []
for t in range(T):
    pred_t = predictions[:, t]
    target_t = targets[:, t]
    
    vrmse_per_ch = compute_vrmse_per_channel(pred_t, target_t)
    mean_vrmse = np.nanmean(vrmse_per_ch)
    all_frame_vrmse.append(mean_vrmse)
    
    print(f"{t:<8}", end="")
    for v in vrmse_per_ch:
        print(f"{v:<12.4f}", end="")
    print(f"{mean_vrmse:<12.4f}")

print("-" * 70)
print(f"{'Average':<8}", end="")
avg_per_ch = []
for c in range(C):
    ch_vrmse = []
    for t in range(T):
        pred_t = predictions[:, t, ..., c]
        target_t = targets[:, t, ..., c]
        var = target_t.flatten().var()
        if var > 1e-8:
            rmse = ((pred_t - target_t) ** 2).mean().sqrt()
            vrmse = rmse / var.sqrt()
        else:
            vrmse = float('nan')
        ch_vrmse.append(vrmse if isinstance(vrmse, float) else vrmse.item())
    avg_per_ch.append(np.nanmean(ch_vrmse))
    print(f"{np.nanmean(ch_vrmse):<12.4f}", end="")
print(f"{np.nanmean(all_frame_vrmse):<12.4f}")

# Also show baseline comparison
print("\n" + "=" * 70)
print("BASELINE COMPARISON (Repeat Last Frame)")
print("=" * 70)

inputs = data["inputs"]  # (B, 1, H, W, C)
last_frame = inputs[:, -1]  # The conditioning frame

print(f"\n{'Frame':<8}", end="")
for name in field_names:
    print(f"{name:<12}", end="")
print(f"{'Mean':<12}")
print("-" * 70)

baseline_frame_vrmse = []
for t in range(T):
    target_t = targets[:, t]
    # Baseline: repeat the input frame
    baseline_t = last_frame
    
    vrmse_per_ch = compute_vrmse_per_channel(baseline_t, target_t)
    mean_vrmse = np.nanmean(vrmse_per_ch)
    baseline_frame_vrmse.append(mean_vrmse)
    
    print(f"{t:<8}", end="")
    for v in vrmse_per_ch:
        print(f"{v:<12.4f}", end="")
    print(f"{mean_vrmse:<12.4f}")

print("-" * 70)
print(f"{'Average':<8}", end="")
for c in range(C):
    ch_vrmse = []
    for t in range(T):
        target_t = targets[:, t, ..., c]
        baseline_t = last_frame[..., c]
        var = target_t.flatten().var()
        if var > 1e-8:
            rmse = ((baseline_t - target_t) ** 2).mean().sqrt()
            vrmse = rmse / var.sqrt()
        else:
            vrmse = float('nan')
        ch_vrmse.append(vrmse if isinstance(vrmse, float) else vrmse.item())
    print(f"{np.nanmean(ch_vrmse):<12.4f}", end="")
print(f"{np.nanmean(baseline_frame_vrmse):<12.4f}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Model average VRMSE: {np.nanmean(all_frame_vrmse):.4f}")
print(f"Baseline average VRMSE: {np.nanmean(baseline_frame_vrmse):.4f}")
if np.nanmean(all_frame_vrmse) < np.nanmean(baseline_frame_vrmse):
    improvement = (1 - np.nanmean(all_frame_vrmse) / np.nanmean(baseline_frame_vrmse)) * 100
    print(f"Model is {improvement:.1f}% BETTER than baseline")
else:
    degradation = (np.nanmean(all_frame_vrmse) / np.nanmean(baseline_frame_vrmse) - 1) * 100
    print(f"Model is {degradation:.1f}% WORSE than baseline")
