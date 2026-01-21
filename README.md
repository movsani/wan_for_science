# Wan2.2-I2V Fine-tuning for Physics Simulation Prediction

Fine-tune the **Wan2.2-I2V-14B** video generative model on physics simulation data from **The Well** dataset to predict temporal evolution of turbulent radiative layer dynamics.

## Overview

Given an initial physics state (density, pressure, velocity fields), the model predicts future states using Wan2.2's native image-to-video diffusion capabilities.

### Approaches

**Native I2V Diffusion (Recommended):**
```
Physics Frame  →  Channel  →  Spatial  →  VAE  →  Wan2.2 I2V  →  VAE  →  Spatial  →  Channel  →  Future
(4 channels)      Encode      Upscale     Encode   Diffusion     Decode   Downscale   Decode     Frames
                 (learnable) (learnable) (frozen) (LoRA tuned)  (frozen) (learnable) (learnable)
```

The model conditions on 1 physics frame and generates 8 future frames using Wan2.2's native I2V conditioning mechanism (36-channel input with mask).

**Legacy: Temporal Predictor (ConvLSTM):**
```
Physics Input  →  Adapters  →  VAE Encode  →  ConvLSTM  →  VAE Decode  →  Adapters  →  Physics Output
                              (frozen)     (learnable)   (frozen)
```

**Legacy: Adapter-only Training:**
```
Physics Data  →  Channel Encoder  →  Wan2.2-I2V  →  Channel Decoder  →  Physics Prediction
                  (learnable)       (LoRA tuned)     (learnable)
```

### Key Features

- **Native I2V Diffusion**: Leverages Wan2.2's video generation for physics prediction
- **LoRA Fine-tuning**: Efficient parameter adaptation (~0.7% trainable parameters)
- **Learnable Adapters**: Channel (4↔3) and spatial (physics↔video resolution) adapters
- **VAE Reconstruction Loss**: Trains decoder on actual inference path
- **Physics Metrics**: VRMSE evaluation standard in physics ML benchmarks
- **Multi-GPU Training**: Distributed training on 4xH100 GPUs

## Quick Start

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd video_for_science
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install torch torchvision
pip install -r requirements.txt
pip install git+https://github.com/huggingface/diffusers
```

### Training (Native I2V Diffusion)

```bash
# Multi-GPU training (recommended)
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/i2v_native.yaml \
    --checkpoint_dir ./checkpoints/

# Single GPU
python scripts/train.py --config configs/i2v_native.yaml
```

### Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint ./checkpoints/best_model_adapters.pt \
    --config configs/i2v_native.yaml \
    --output_dir ./evaluation_results \
    --num_samples 50
```

## Configuration

### Native I2V Diffusion (`configs/i2v_native.yaml`)

```yaml
model:
  use_native_i2v: true
  lora:
    enabled: true
    rank: 32
    target_modules: ["to_q", "to_k", "to_v", "to_out.0"]

data:
  n_steps_input: 1    # Conditioning frame
  n_steps_output: 8   # Frames to predict

training:
  batch_size: 1
  lr: 1.0e-5
  num_epochs: 20
```

### Legacy Temporal Predictor (`configs/default.yaml`)

```yaml
model:
  temporal_predictor:
    enabled: true
    type: "convlstm"
    hidden_channels: 64
```

## Data

Uses the [turbulent_radiative_layer_2D](https://polymathic-ai.org/the_well/datasets/turbulent_radiative_layer_2D/) dataset:
- **4 Fields**: density, pressure, velocity_x, velocity_y
- **Resolution**: 128 × 384

```bash
python scripts/download_data.py --output_dir ./datasets
```

## Architecture Details

### Native I2V Conditioning

Wan2.2 I2V uses a 36-channel input to the transformer:
- **16 channels**: Noisy latent (to denoise)
- **20 channels**: Conditioning (4ch mask + 16ch VAE-encoded condition)

The mask indicates frame 0 as conditioning (preserved), frames 1+ as generated.

### Training Losses

| Loss | Weight | Purpose |
|------|--------|---------|
| Diffusion | 1.0 | Noise prediction (denoising objective) |
| VAE Reconstruction | 0.5 | Train decoder on VAE-decoded outputs |
| Adapter | 0.1 | Encode-decode cycle consistency |

### Evaluation

Predictions are compared against:
- **Repeat Last Frame Baseline**: Naive persistence forecast
- **Mean Prediction Baseline**: VRMSE = 1.0 by definition

## Project Structure

```
├── configs/
│   ├── i2v_native.yaml      # Native I2V diffusion config
│   └── default.yaml         # Legacy temporal predictor config
├── scripts/
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   └── download_data.py     # Dataset download
├── src/
│   ├── models/
│   │   ├── wan_wrapper.py       # Wan2.2 wrapper with I2V diffusion
│   │   ├── channel_adapter.py   # 4ch ↔ 3ch adapters
│   │   └── temporal_predictor.py
│   ├── training/trainer.py
│   └── evaluation/metrics.py
```

## Hardware Requirements

- **Minimum**: 1x 80GB GPU (H100/A100), 64GB RAM
- **Recommended**: 4x H100 80GB for distributed training

## Citation

```bibtex
@article{wan2025,
  title={Wan: Open and Advanced Large-Scale Video Generative Models},
  author={Team Wan},
  journal={arXiv preprint arXiv:2503.20314},
  year={2025}
}

@article{thewell2024,
  title={The Well: A Large-Scale Collection of Diverse Physics Simulations for ML},
  author={The Well Team},
  year={2024}
}
```

## License

Apache 2.0 License

## Acknowledgments

- [Wan-AI](https://github.com/Wan-Video/Wan2.2) for Wan2.2
- [Polymathic AI](https://polymathic-ai.org/) for The Well dataset
- [Hugging Face](https://huggingface.co/) for Diffusers
