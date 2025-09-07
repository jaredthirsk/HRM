# HRM Sudoku Training Plan

## Overview
This document outlines the plan for training the Hierarchical Reasoning Model (HRM) on Sudoku puzzles. The HRM is a 27M parameter dual-module recurrent architecture designed for complex reasoning tasks.

## Current Status

### Dependencies Installed
- ✅ PyTorch 2.7.1+cu118
- ✅ einops
- ✅ wandb
- ✅ hydra-core
- ✅ omegaconf
- ✅ pydantic
- ✅ argdantic
- ✅ huggingface_hub
- ✅ coolname
- ✅ tqdm

### Dependencies Skipped (Workarounds Needed)
- ❌ **adam-atan2**: Installation failing due to setuptools compatibility issue
  - **Impact**: Custom optimizer with arctan2 learning rate scheduling
  - **Workaround**: Will need to modify pretrain.py to use standard Adam optimizer or implement AdamATan2 locally
  
- ❌ **FlashAttention**: Not yet installed
  - **Impact**: Memory-efficient attention mechanism
  - **Workaround**: Can run without it but may need smaller batch sizes

## Training Pipeline Steps

### Phase 1: Environment Setup ✅
1. Clone repository
2. Install Python dependencies
3. Configure CUDA environment

### Phase 2: Dataset Preparation (Current)
1. Build Sudoku dataset using the provided script
   ```bash
   python3 dataset/build_sudoku_dataset.py \
     --output-dir data/sudoku-extreme-1k-aug-1000 \
     --subsample-size 1000 \
     --num-aug 1000
   ```
2. Verify dataset structure and format
3. Explore dataset with puzzle_visualizer.html

### Phase 3: Code Modifications
1. **Optimizer Workaround**:
   - Option A: Implement local AdamATan2 class
   - Option B: Modify pretrain.py to use standard Adam optimizer
   - Option C: Try to fix adam-atan2 installation

2. **Configuration Updates**:
   - Review config/cfg_pretrain.yaml
   - Review config/arch/hrm_v1.yaml
   - Adjust hyperparameters for single GPU if needed

### Phase 4: Initial Training
1. **Single GPU Training** (for development):
   ```bash
   OMP_NUM_THREADS=8 python pretrain.py \
     data_path=data/sudoku-extreme-1k-aug-1000 \
     epochs=20000 \
     eval_interval=2000 \
     global_batch_size=384 \
     lr=7e-5 \
     puzzle_emb_lr=7e-5 \
     weight_decay=1.0 \
     puzzle_emb_weight_decay=1.0
   ```
   - Expected runtime: ~10 hours on RTX 4070

2. **Multi-GPU Training** (if available):
   ```bash
   OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py \
     data_path=data/sudoku-extreme-1k-aug-1000 \
     epochs=20000 \
     eval_interval=2000 \
     lr=1e-4 \
     puzzle_emb_lr=1e-4 \
     weight_decay=1.0 \
     puzzle_emb_weight_decay=1.0
   ```
   - Expected runtime: ~10 minutes on 8 GPUs

### Phase 5: Monitoring & Evaluation
1. Set up Weights & Biases tracking
2. Monitor training metrics:
   - Loss curves
   - Exact accuracy
   - Validation performance
3. Early stopping if accuracy approaches 100% (to avoid overfitting)

### Phase 6: Evaluation & Analysis
1. Run evaluation script on checkpoint
2. Test on held-out Sudoku puzzles
3. Analyze failure cases
4. Document results

## Key Model Parameters

### Architecture (HRM v1)
- **H_cycles**: Number of high-level processing cycles (planning)
- **L_cycles**: Number of low-level processing cycles (computation)
- **H_layers**: Transformer layers in high-level module
- **L_layers**: Transformer layers in low-level module
- **halt_max_steps**: Maximum ACT steps before forced halting
- **model_dim**: Hidden dimension size

### Training
- **global_batch_size**: Total batch size across all GPUs
- **lr**: Learning rate for model parameters
- **puzzle_emb_lr**: Learning rate for puzzle embeddings
- **lr_warmup_steps**: Warmup period for learning rate
- **weight_decay**: L2 regularization
- **epochs**: Total training epochs
- **eval_interval**: Evaluation frequency

## Risks & Mitigations

1. **adam-atan2 installation failure**
   - Risk: Cannot use specialized optimizer
   - Mitigation: Implement workaround or use standard Adam

2. **CUDA/GPU compatibility**
   - Risk: FlashAttention requires specific GPU architecture
   - Mitigation: Run without FlashAttention, adjust batch size

3. **Memory constraints**
   - Risk: OOM errors on single GPU
   - Mitigation: Reduce batch size, use gradient accumulation

4. **Training instability**
   - Risk: Numerical instability in late training
   - Mitigation: Early stopping, use Stablemax loss

## Success Criteria
- Dataset successfully built with 1000 augmented examples
- Training script runs without errors
- Model achieves >90% exact accuracy on validation set
- Training completes within reasonable time (<24 hours)

## Next Steps
1. Build the Sudoku dataset
2. Implement optimizer workaround
3. Configure W&B logging
4. Start initial training run
5. Monitor and adjust hyperparameters as needed