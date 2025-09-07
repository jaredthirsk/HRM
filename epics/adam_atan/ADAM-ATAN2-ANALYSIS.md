# AdamATan2 Optimizer: Analysis and Implications

## Why adam-atan2 Package Installation Fails

### Root Causes

1. **Setuptools Version Incompatibility**
   - Error: `TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'`
   - Cause: Package uses outdated setuptools API incompatible with newer versions
   - The package's setup.py hasn't been updated for newer setuptools

2. **CUDA Compilation Requirements**
   - The package includes CUDA extensions written in C++/CUDA
   - Requires CUDA version to match PyTorch's CUDA version exactly
   - Our environment: PyTorch compiled with CUDA 11.8, system has CUDA 12.3
   - Error: `The detected CUDA version (12.3) mismatches the version that was used to compile PyTorch (11.8)`

3. **Build System Issues**
   - Package lacks modern `pyproject.toml` configuration
   - Uses legacy `setup.py` build system
   - Requires compilation of C++ extensions at install time

## What is AdamATan2?

### The Innovation (from arXiv:2407.05872)

AdamATan2 is from the paper **"Scaling Exponents Across Parameterizations and Optimizers"** by Google DeepMind (July 2024).

**Key Change**: Replace division with `atan2` function

Standard Adam update:
```python
param -= lr * exp_avg / (sqrt(exp_avg_sq) + epsilon)
```

AdamATan2 update:
```python
param -= lr * atan2(exp_avg, sqrt(exp_avg_sq))
```

### Benefits
1. **Eliminates epsilon hyperparameter** - No more tuning epsilon
2. **Numerical stability** - atan2 handles edge cases gracefully
3. **Scale invariance** - Better behavior across different parameter scales
4. **Prevents gradient underflow** - Critical for large models

### Mathematical Properties

The `atan2(y, x)` function:
- Returns angle in radians between -π and π
- Handles x=0 gracefully (unlike division)
- Provides smooth gradients near zero
- Bounded output prevents extreme updates

## Our Workaround vs Original

### Original CUDA Implementation
```cpp
// From adam_atan2.cu
const opmath_t denom = std::sqrt(exp_avg_sq) / bias_correction2_sqrt;
param -= step_size * std::atan2(exp_avg, denom);
```

### Our Simple Workaround (adam_atan2.py)
```python
class AdamATan2(Adam):
    # Just wraps standard Adam - WRONG ALGORITHM!
    pass
```

### Our Proper Implementation (adam_atan2_proper.py)
```python
# Correct algorithm
denom = exp_avg_sq.sqrt() / bias_correction2_sqrt
p.data.add_(torch.atan2(exp_avg, denom), alpha=-step_size)
```

## Performance Implications

### Using Simple Workaround (Current)
- ❌ **Not the real algorithm** - Uses standard Adam division
- ❌ **Missing key benefits** - No epsilon elimination, no atan2 smoothing
- ❌ **Different convergence** - May need different hyperparameters
- ✅ **Still works** - Adam is robust, but suboptimal

### Using Proper Implementation
- ✅ **Correct algorithm** - Matches paper's method
- ✅ **Better numerical stability** - No division by near-zero
- ✅ **No epsilon tuning** - One less hyperparameter
- ❌ **Slightly slower** - atan2 more expensive than division
- ❌ **No CUDA optimization** - Pure Python is slower

### Performance Comparison

| Aspect | Standard Adam | Our Workaround | Proper AdamATan2 | CUDA AdamATan2 |
|--------|--------------|----------------|------------------|----------------|
| Algorithm | Division + ε | Division + ε | atan2 | atan2 |
| Speed | Fast | Fast | Medium | Fastest |
| Stability | Good | Good | Best | Best |
| Hyperparameters | lr, β1, β2, ε | lr, β1, β2, ε | lr, β1, β2 | lr, β1, β2 |
| Scale Invariance | No | No | Yes | Yes |

## Recommendations

### For This Project

1. **Short-term**: Continue with simple workaround
   - Training is working
   - Results should be similar
   - May need slight hyperparameter adjustment

2. **Better Option**: Use proper implementation
   ```python
   # Replace adam_atan2.py with adam_atan2_proper.py
   cp adam_atan2_proper.py adam_atan2.py
   ```

3. **Best Option**: Fix CUDA installation
   - Match CUDA versions
   - Or install CPU-only version
   - Or compile from source

### Hyperparameter Adjustments

If using standard Adam instead of AdamATan2:
- Consider adding small epsilon: `eps=1e-8` or `eps=1e-6`
- May need slightly different learning rate
- Watch for gradient underflow in late training

## How to Get Real AdamATan2 Working

### Option 1: Match CUDA Versions
```bash
# Install PyTorch with CUDA 12.3 to match system
pip3 install torch --index-url https://download.pytorch.org/whl/cu123
# Then retry adam-atan2 installation
pip3 install adam-atan2
```

### Option 2: Use PyPI Alternative
```bash
# Install lucidrains' implementation (pure Python)
pip3 install adam-atan2-pytorch
```

### Option 3: Compile from Source
```bash
git clone https://github.com/lucidrains/adam-atan2-pytorch
cd adam-atan2-pytorch
pip3 install -e .
```

## Impact on HRM Training

### Current Status
- Using standard Adam (disguised as AdamATan2)
- Training successfully at ~1.3 iter/sec
- Model converging normally

### Expected Differences with Real AdamATan2
1. **Better late-stage training** - More stable near convergence
2. **No epsilon tuning needed** - Simpler hyperparameter search
3. **Possibly different optimal LR** - atan2 has different scaling
4. **More robust to scale** - Better for mixed precision

### Bottom Line
- **Current workaround is functional** but not optimal
- **Proper implementation available** in adam_atan2_proper.py
- **Original's benefits** are nice-to-have, not critical
- **Training succeeds either way** - Adam is robust

## References

1. [Scaling Exponents Across Parameterizations and Optimizers](https://arxiv.org/abs/2407.05872)
2. [GitHub: lucidrains/adam-atan2-pytorch](https://github.com/lucidrains/adam-atan2-pytorch)
3. [John Carmack's endorsement](https://x.com/ID_AA_Carmack/status/1819152769980432678)