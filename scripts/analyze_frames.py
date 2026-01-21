#!/usr/bin/env python3
"""
Analyze I2V output frame structure to determine which frames are 
conditioning vs generated.
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
print("I2V FRAME STRUCTURE ANALYSIS")
print("=" * 60)

print(f"\nShapes:")
print(f"  Inputs: {inputs.shape}")      # (B, 1, H, W, C)
print(f"  Predictions: {predictions.shape}")  # (B, 5, H, W, C)
print(f"  Targets: {targets.shape}")     # (B, 5, H, W, C) - truncated to match

# Question: Is prediction frame 0 similar to the input frame?
# If so, frame 0 is the conditioning frame (preserved), frames 1-4 are generated

print("\n" + "=" * 60)
print("SIMILARITY: Input frame vs Prediction frames")
print("=" * 60)

input_frame = inputs[:, 0]  # (B, H, W, C) - the conditioning frame

for t in range(predictions.shape[1]):
    pred_frame = predictions[:, t]  # (B, H, W, C)
    
    # MSE between input and this prediction frame
    mse = ((input_frame - pred_frame) ** 2).mean().item()
    
    # Correlation per channel
    corr_per_ch = []
    for c in range(4):
        in_ch = input_frame[..., c].flatten().numpy()
        pr_ch = pred_frame[..., c].flatten().numpy()
        if in_ch.std() > 0 and pr_ch.std() > 0:
            corr = np.corrcoef(in_ch, pr_ch)[0, 1]
        else:
            corr = 0
        corr_per_ch.append(corr)
    
    mean_corr = np.mean(corr_per_ch)
    
    print(f"Pred frame {t}: MSE to input = {mse:.4f}, Correlation = {mean_corr:.4f}")

print("\n" + "=" * 60)
print("SIMILARITY: Prediction frames vs Target frames")
print("=" * 60)

# Also check how prediction frames relate to target frames
# Target frame 0 should be the first frame AFTER the conditioning
for t in range(min(predictions.shape[1], targets.shape[1])):
    pred_frame = predictions[:, t]
    target_frame = targets[:, t]
    
    mse = ((pred_frame - target_frame) ** 2).mean().item()
    
    corr_per_ch = []
    for c in range(4):
        pr_ch = pred_frame[..., c].flatten().numpy()
        tg_ch = target_frame[..., c].flatten().numpy()
        if pr_ch.std() > 0 and tg_ch.std() > 0:
            corr = np.corrcoef(pr_ch, tg_ch)[0, 1]
        else:
            corr = 0
        corr_per_ch.append(corr)
    
    mean_corr = np.mean(corr_per_ch)
    
    print(f"Frame {t}: Pred vs Target MSE = {mse:.4f}, Correlation = {mean_corr:.4f}")

print("\n" + "=" * 60)
print("EXPECTED FRAME MAPPING")
print("=" * 60)
print("""
For I2V with 8 target frames (encoded to 2 latent frames):

Input:   frame 0 (conditioning)
Targets: frames 1-8 (to predict)

Latent space:
  Latent frame 0 = encodes input frame 0 (mask=1, kept)
  Latent frame 1 = should be generated (mask=0)

VAE decode (2 latent → 5 video):
  Video frame 0 ≈ latent frame 0 start → should match input
  Video frames 1-4 ≈ interpolated/generated

So prediction frame 0 should be ~= input, and we should compare:
  - Pred frames 1-4 vs Target frames 0-3 (shift by 1)
  OR
  - Pred frames 0-4 vs Target frames 0-4 (if targets start at conditioning)

Let's check what the targets actually contain...
""")

print("First 10 pixels of input[0, 0] density:")
print(inputs[0, 0, 0, :10, 0].numpy()[:10])

print("\nFirst 10 pixels of target[0, 0] density:")
print(targets[0, 0, 0, :10, 0].numpy()[:10])

print("\nAre input and target[0] the same frame?")
diff = (inputs[0, 0] - targets[0, 0]).abs().max().item()
print(f"Max difference: {diff:.6f}")
if diff < 1e-5:
    print("YES - target frame 0 is the same as input frame 0 (conditioning)")
else:
    print("NO - target starts after the conditioning frame")
