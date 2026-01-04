# How to Understand Your Evaluation Results

This guide explains how to interpret the metrics from your STDiff model evaluation.

---

## Your Results Summary

Based on your test run, here are your metrics:

```
PSNR: 22.66 dB (per-frame: [24.85, 22.23, 20.90])
SSIM: 0.575 (per-frame: [0.650, 0.559, 0.518])
LPIPS: 0.222 (per-frame: [0.165, 0.237, 0.264])
InterLPIPS: 0.047 (per-frame: [0.000, 0.042, 0.052])
FVD: 144.93
```

---

## Metric Explanations

### 1. **PSNR (Peak Signal-to-Noise Ratio)** = 22.66 dB

**What it measures:** Pixel-level similarity between predicted and ground truth frames.

**How to interpret:**
- **Higher is better** (typically ranges from 15-40 dB)
- **20-25 dB**: Moderate quality (some visible differences)
- **25-30 dB**: Good quality (minor differences)
- **30+ dB**: Excellent quality (nearly identical)
- **< 20 dB**: Poor quality (significant differences)

**Your result: 22.66 dB**
- ✅ **Moderate to good quality**
- Predictions are reasonably close to ground truth at pixel level
- Some visible differences but generally acceptable

**Per-frame breakdown:**
- Frame 0: 24.85 dB (best)
- Frame 1: 22.23 dB (middle)
- Frame 2: 20.90 dB (worst - quality degrades over time)

**Insight:** Quality decreases for later predicted frames (common in video prediction).

---

### 2. **SSIM (Structural Similarity Index)** = 0.575

**What it measures:** Perceptual similarity considering luminance, contrast, and structure.

**How to interpret:**
- **Range: 0 to 1** (1 = identical, 0 = completely different)
- **Higher is better**
- **0.9-1.0**: Excellent similarity
- **0.7-0.9**: Good similarity
- **0.5-0.7**: Moderate similarity
- **< 0.5**: Poor similarity

**Your result: 0.575**
- ✅ **Moderate perceptual similarity**
- Structures and patterns are somewhat preserved
- Some perceptual differences exist

**Per-frame breakdown:**
- Frame 0: 0.650 (best)
- Frame 1: 0.559 (middle)
- Frame 2: 0.518 (worst)

**Insight:** Consistent with PSNR - quality degrades over time.

---

### 3. **LPIPS (Learned Perceptual Image Patch Similarity)** = 0.222

**What it measures:** Perceptual similarity using deep features (AlexNet).

**How to interpret:**
- **Range: 0 to ~1** (0 = identical, higher = more different)
- **Lower is better**
- **0.0-0.1**: Very similar perceptually
- **0.1-0.3**: Moderately similar
- **0.3-0.5**: Noticeably different
- **> 0.5**: Very different

**Your result: 0.222**
- ✅ **Moderate perceptual similarity**
- Predictions are perceptually reasonable
- Some differences in high-level features

**Per-frame breakdown:**
- Frame 0: 0.165 (best)
- Frame 1: 0.237 (middle)
- Frame 2: 0.264 (worst)

**Insight:** Perceptual quality also degrades over time.

---

### 4. **InterLPIPS (Inter-frame LPIPS)** = 0.047

**What it measures:** Temporal consistency - how similar consecutive predicted frames are.

**How to interpret:**
- **Range: 0 to ~1** (0 = perfectly smooth, higher = more jittery)
- **Lower is better**
- **0.0-0.05**: Very smooth motion
- **0.05-0.15**: Smooth motion
- **0.15-0.3**: Some jitter
- **> 0.3**: Jittery motion

**Your result: 0.047**
- ✅ **Very smooth temporal consistency**
- Excellent frame-to-frame smoothness
- Motion is temporally coherent

**Per-frame breakdown:**
- Frame 0-1: 0.000 (perfect - no transition, this is the first frame)
- Frame 1-2: 0.042 (very smooth)
- Frame 2-3: 0.052 (still smooth)

**Insight:** Your model produces temporally smooth predictions! This is a strength.

---

### 5. **FVD (Fréchet Video Distance)** = 144.93

**What it measures:** Video-level quality comparing distributions of real vs predicted videos.

**How to interpret:**
- **Lower is better** (typically ranges from 0 to 1000+)
- **0-50**: Excellent video quality
- **50-150**: Good video quality
- **150-300**: Moderate video quality
- **300-500**: Poor video quality
- **> 500**: Very poor quality

**Your result: 144.93**
- ✅ **Good video-level quality**
- Predicted videos are reasonably realistic
- Motion and appearance are generally plausible

**Note:** This metric considers the entire video (observed + predicted frames) and measures how "realistic" the generated videos look compared to real videos.

---

## Overall Assessment

### ✅ **Strengths:**
1. **Excellent temporal consistency** (InterLPIPS = 0.047)
   - Smooth, coherent motion
   - No jitter or flickering

2. **Good video-level quality** (FVD = 144.93)
   - Videos look realistic overall
   - Motion patterns are plausible

3. **Moderate pixel-level accuracy** (PSNR = 22.66 dB)
   - Reasonable similarity to ground truth

### ⚠️ **Areas for Improvement:**
1. **Quality degradation over time**
   - Frame 0: Best quality
   - Frame 2: Worst quality
   - Common in video prediction - errors accumulate

2. **Perceptual similarity could be higher** (SSIM = 0.575)
   - Structures could be better preserved
   - Some perceptual differences exist

3. **Pixel-level accuracy** (PSNR = 22.66 dB)
   - Could be improved for better detail preservation

---

## How to Compare Results

### Compare with:
1. **Baseline methods** (if available)
   - Lower FVD, higher PSNR/SSIM = better
   - Lower LPIPS = better

2. **Different checkpoints**
   - Track metrics across training to see improvement

3. **Different configurations**
   - Compare different model settings

### Typical Ranges for Video Prediction:
- **PSNR**: 20-30 dB (yours: 22.66 ✅)
- **SSIM**: 0.5-0.8 (yours: 0.575 ✅)
- **LPIPS**: 0.1-0.3 (yours: 0.222 ✅)
- **FVD**: 50-200 (yours: 144.93 ✅)

**Your results are within typical ranges for video prediction tasks!**

---

## Understanding Per-Frame Metrics

The per-frame values show how quality changes over time:

```
Frame 0: PSNR=24.85, SSIM=0.650, LPIPS=0.165  ← Best
Frame 1: PSNR=22.23, SSIM=0.559, LPIPS=0.237  ← Middle
Frame 2: PSNR=20.90, SSIM=0.518, LPIPS=0.264  ← Worst
```

**Pattern:** Quality decreases for later frames
- **Why?** Errors accumulate in autoregressive prediction
- **Normal?** Yes, this is expected in video prediction
- **Solution?** Better motion modeling, longer training, or different architectures

---

## What Each Metric Tells You

| Metric | What It Measures | Your Score | Interpretation |
|--------|-----------------|------------|----------------|
| **PSNR** | Pixel accuracy | 22.66 dB | Moderate pixel-level similarity |
| **SSIM** | Structural similarity | 0.575 | Moderate perceptual similarity |
| **LPIPS** | Perceptual similarity | 0.222 | Moderate high-level similarity |
| **InterLPIPS** | Temporal smoothness | 0.047 | **Excellent** temporal consistency |
| **FVD** | Video realism | 144.93 | Good video-level quality |

---

## Key Takeaways

1. **Temporal consistency is excellent** - Your model produces smooth videos
2. **Overall quality is good** - Videos are realistic and plausible
3. **Pixel accuracy is moderate** - Some detail loss, but acceptable
4. **Quality degrades over time** - Expected behavior in video prediction

---

## Next Steps

1. **Visual inspection**: Check the generated GIFs/videos in `test_examples_*` folders
2. **Compare checkpoints**: Run evaluation on different checkpoints to track progress
3. **Ablation studies**: Test different configurations to improve metrics
4. **Focus on weak points**: If PSNR/SSIM need improvement, consider:
   - Longer training
   - Better loss functions
   - Architecture improvements

---

## Summary

Your model performs **well** for video prediction:
- ✅ Smooth, temporally consistent videos
- ✅ Realistic motion and appearance
- ✅ Good overall video quality
- ⚠️ Some quality degradation over time (normal)
- ⚠️ Room for improvement in pixel-level accuracy

**Overall assessment: Good performance for a video prediction model!** 🎉

