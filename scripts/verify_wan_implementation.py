#!/usr/bin/env python3
"""
Verify our I2V implementation against the official diffusers WanImageToVideoPipeline.
This script extracts and prints the key methods to compare.
"""

import inspect
import sys

print("=" * 80)
print("DIFFUSERS WAN I2V PIPELINE REFERENCE")
print("=" * 80)

try:
    from diffusers import WanImageToVideoPipeline
except ImportError:
    print("ERROR: diffusers not installed or WanImageToVideoPipeline not found")
    sys.exit(1)

# Get source file
source_file = inspect.getsourcefile(WanImageToVideoPipeline)
print(f"\nSource file: {source_file}\n")

# === 1. Look for prepare_latents ===
print("=" * 80)
print("PREPARE_LATENTS METHOD")
print("=" * 80)
if hasattr(WanImageToVideoPipeline, 'prepare_latents'):
    try:
        source = inspect.getsource(WanImageToVideoPipeline.prepare_latents)
        print(source[:3000])  # First 3000 chars
    except Exception as e:
        print(f"Could not get source: {e}")
else:
    print("prepare_latents not found directly, checking parent classes...")
    for cls in WanImageToVideoPipeline.__mro__:
        if hasattr(cls, 'prepare_latents') and 'prepare_latents' in cls.__dict__:
            print(f"Found in {cls.__name__}")
            try:
                source = inspect.getsource(cls.prepare_latents)
                print(source[:3000])
            except:
                pass
            break

# === 2. Look for encode_image if exists ===
print("\n" + "=" * 80)
print("ENCODE_IMAGE METHOD (if exists)")
print("=" * 80)
if hasattr(WanImageToVideoPipeline, 'encode_image'):
    try:
        source = inspect.getsource(WanImageToVideoPipeline.encode_image)
        print(source[:2000])
    except Exception as e:
        print(f"Could not get source: {e}")
else:
    print("encode_image not found")

# === 3. Look at __call__ to understand the flow ===
print("\n" + "=" * 80)
print("__CALL__ METHOD (main pipeline)")
print("=" * 80)
try:
    source = inspect.getsource(WanImageToVideoPipeline.__call__)
    # Find key sections
    lines = source.split('\n')
    
    # Print the docstring and first 100 lines
    print("First 150 lines of __call__:")
    print("-" * 40)
    for i, line in enumerate(lines[:150]):
        print(f"{i+1:4d}: {line}")
    
    # Find sections about latent preparation
    print("\n" + "-" * 40)
    print("Looking for 'latent' mentions...")
    for i, line in enumerate(lines):
        if 'latent' in line.lower() and 'prepare' in line.lower():
            start = max(0, i-2)
            end = min(len(lines), i+10)
            print(f"\n--- Around line {i+1} ---")
            for j in range(start, end):
                print(f"{j+1:4d}: {lines[j]}")
                
    # Find sections about conditioning/mask
    print("\n" + "-" * 40)
    print("Looking for 'mask' mentions...")
    for i, line in enumerate(lines):
        if 'mask' in line.lower() and ('lat' in line.lower() or 'cond' in line.lower()):
            start = max(0, i-2)
            end = min(len(lines), i+8)
            print(f"\n--- Around line {i+1} ---")
            for j in range(start, end):
                print(f"{j+1:4d}: {lines[j]}")
            
except Exception as e:
    print(f"Could not get __call__ source: {e}")

# === 4. Summary of what we're looking for ===
print("\n" + "=" * 80)
print("KEY QUESTIONS TO VERIFY")
print("=" * 80)
print("""
1. How is the conditioning image encoded?
   - Is it part of the full video latent, or separate?
   
2. How is the mask structured?
   - mask=1 for conditioning frame, mask=0 for generated?
   
3. What does the output video contain?
   - Frame 0 = conditioning, frames 1+ = generated?
   - Or all frames are treated as output?
   
4. How is the latent concatenated?
   - hidden_states = [noisy_latent (16ch), conditioning (20ch)] = 36ch?

Compare these findings with our implementation in:
  src/models/wan_wrapper.py - forward_i2v_diffusion() and predict_i2v_diffusion()
""")
