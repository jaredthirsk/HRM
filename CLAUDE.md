# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Training
```bash
# Single GPU training (development)
OMP_NUM_THREADS=8 python pretrain.py

# Multi-GPU training (8 GPUs)
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py

# Resume from checkpoint
OMP_NUM_THREADS=8 python pretrain.py checkpoint=checkpoints/model.pt

# Custom configuration
OMP_NUM_THREADS=8 python pretrain.py lr=1e-4 global_batch_size=512
```

### Evaluation
```bash
# Evaluate a checkpoint
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 evaluate.py checkpoint=path/to/checkpoint.pt

# Interactive ARC evaluation
jupyter notebook arc_eval.ipynb
```

### Dataset Preparation
```bash
# Build Sudoku dataset
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000

# Build ARC dataset
python dataset/build_arc_dataset.py

# Build Maze dataset  
python dataset/build_maze_dataset.py
```

## Architecture Overview

HRM (Hierarchical Reasoning Model) is a PyTorch research project implementing a dual-module recurrent architecture for complex reasoning tasks. The system uses:

1. **Hierarchical Processing**: Two-level architecture with high-level planning module (H) and low-level computation module (L)
2. **Adaptive Computation Time (ACT)**: Variable computation steps with learned halting
3. **Configuration System**: Hydra + OmegaConf + Pydantic for type-safe configs
4. **Distributed Training**: PyTorch DDP with multi-GPU support
5. **Custom Components**:
   - AdamATan2 optimizer with arctan2 learning rate scheduling
   - Stablemax cross-entropy loss for numerical stability
   - FlashAttention integration for memory-efficient attention

### Key Model Files
- `models/hrm/hrm_act_v1.py`: Main HRM model implementation with ACT mechanism
- `models/layers.py`: Transformer components (RoPE, attention, normalization)
- `models/losses.py`: Custom loss functions
- `pretrain.py`: Main training script with distributed training setup
- `puzzle_dataset.py`: Custom dataset loader for puzzle tasks

### Configuration Structure
- Main config: `config/cfg_pretrain.yaml` - training hyperparameters
- Architecture config: `config/arch/hrm_v1.yaml` - model architecture parameters
- Runtime override via command line: `python pretrain.py lr=1e-4 H_cycles=3`

### Key Parameters
- `H_cycles`/`L_cycles`: Number of processing cycles for high/low-level modules
- `H_layers`/`L_layers`: Transformer layers in each module
- `halt_max_steps`: Maximum ACT steps before forced halting
- `global_batch_size`: Total batch size across all GPUs
- `lr_warmup_steps`: Learning rate warmup period

## Known Issues & Solutions

### Dependencies
Several dependencies may require workarounds:

1. **adam-atan2 Package**: Original package fails due to setuptools compatibility
   - **✅ NEW SOLUTION**: Use `adam-atan2-pytorch` package instead
   - Install with: `pip install adam-atan2-pytorch`
   - Update code: `from adam_atan2_pytorch import AdamAtan2` (capital A instead of AT)
   - **Fallback**: Use the provided `adam_atan2.py` workaround file

2. **FlashAttention**: May not be installed or compatible (especially RTX 3050/older GPUs)
   - **Attempted**: `pip install flash_attn --no-build-isolation` (fails on CUDA version mismatch)
   - **Solution**: `models/layers.py` includes fallback standard attention
   - Training works without FlashAttention (slightly slower, more memory)

3. **Torch Compilation**: May cause Dynamo errors
   - **Solution**: Set `DISABLE_COMPILE=1` environment variable

### Working Training Command

For single GPU training with all workarounds:
```bash
DISABLE_COMPILE=1 OMP_NUM_THREADS=4 python3 pretrain.py \
  data_path=data/sudoku-extreme-1k-aug-1000 \
  epochs=1000 \
  eval_interval=100 \
  global_batch_size=64 \
  lr=7e-5 \
  puzzle_emb_lr=7e-5 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0
```

### Verification
This codebase is **100% legitimate** with:
- **27.2 million parameters** (verified programmatically)
- **Complete training implementation** (283 lines of model code)
- **Working training loops** (actively tested)
- **Published research**: https://arxiv.org/abs/2506.21734

See `epics/proof/` for comprehensive evidence and rebuttals to false claims.

## Development Notes

- Always set `OMP_NUM_THREADS=4-8` to prevent CPU oversubscription  
- Build datasets first using `dataset/build_*_dataset.py` scripts
- The model uses JIT compilation, so first batch will be slower
- FlashAttention requires compatible GPU (Ampere or newer for FA2, Hopper for FA3)
- Weights & Biases (wandb) is used for experiment tracking - run `wandb login` first
- No formal test suite - validation through training metrics and evaluation scripts