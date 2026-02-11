# ROI Extraction Research Implementation

This repository implements the statistical-indicator-based ROI extraction pipeline described in `PROJECT(roi).md` for the Severstal Steel Defect Detection dataset.

## Overview

The core insight from `PROJECT(roi).md` is that **high-quality augmented data depends on matching "which defect subclass" with "which background context"**. This implementation evaluates not just where defects are located, but whether the background is suitable for synthetic defect generation.

## Research Pipeline

### 1. Defect Characterization (4 Statistical Indicators)

**Module**: `src/analysis/defect_characterization.py`

Computes geometric properties of each defect:

- **Linearity**: Measures elongation using eigenvalue analysis (0-1)
- **Solidity**: Ratio of defect area to convex hull area (0-1)
- **Extent**: Ratio of defect area to bounding box area (0-1)
- **Aspect Ratio**: Major axis / minor axis length (≥1)

Based on these indicators, defects are classified into subtypes:
- `linear_scratch`: High linearity + high aspect ratio
- `elongated`: High aspect ratio + medium linearity
- `compact_blob`: Low aspect ratio + high solidity
- `irregular`: Low solidity
- `general`: Default classification

### 2. Background Characterization (Grid-Based Analysis)

**Module**: `src/analysis/background_characterization.py`

Divides images into 64×64 grids and classifies each patch:

**Step 1: Variance Analysis**
- Low variance → `smooth` (flat metal surface)
- High variance → Proceed to edge analysis

**Step 2: Edge Direction Analysis (Sobel filters)**
- Dominant vertical edges → `vertical_stripe`
- Dominant horizontal edges → `horizontal_stripe`
- Multi-directional edges → `complex_pattern`
- No dominant direction → `textured`

**Step 3: Frequency Analysis (FFT)**
- High-frequency ratio indicates texture complexity

Each grid cell receives:
- Background type classification
- Stability score (0-1): uniformity/consistency of the background

### 3. ROI Suitability Evaluation

**Module**: `src/analysis/roi_suitability.py`

Evaluates defect-background matching using predefined rules:

| Defect Subtype | Best Background | Score |
|----------------|-----------------|-------|
| `linear_scratch` | `vertical_stripe`, `horizontal_stripe` | 1.0 |
| `compact_blob` | `smooth` | 1.0 |
| `irregular` | `complex_pattern` | 1.0 |

**Suitability Score** = 0.5×matching + 0.3×continuity + 0.2×stability

- **Matching Score**: How well defect type fits background type
- **Continuity Score**: Background uniformity within ROI bbox
- **Stability Score**: Average background stability

**Recommendations**:
- `suitable`: suitability ≥ 0.7
- `acceptable`: 0.5 ≤ suitability < 0.7
- `unsuitable`: suitability < 0.5

### 4. ROI Position Optimization

If the defect centroid is near a background boundary (discontinuity), the ROI window is shifted (up to 32 pixels) to maximize background continuity while keeping the defect within bounds.

### 5. Data Packaging

**Module**: `src/preprocessing/roi_extraction.py`

Final output for each ROI:
- ROI image patch (512×512)
- ROI mask patch (512×512)
- Metadata CSV with:
  - Image ID, class ID, region ID
  - Defect metrics (linearity, solidity, extent, aspect ratio)
  - Background type and stability
  - Suitability scores
  - Text prompt for ControlNet training

## Project Structure

```
severstal-steel-defect-detection/
├── src/
│   ├── utils/
│   │   └── rle_utils.py              # RLE encoding/decoding
│   ├── analysis/
│   │   ├── defect_characterization.py    # Step 1: Defect analysis
│   │   ├── background_characterization.py # Step 2: Background analysis
│   │   └── roi_suitability.py           # Step 3: Matching evaluation
│   └── preprocessing/
│       └── roi_extraction.py           # Step 4-5: ROI extraction pipeline
├── scripts/
│   └── extract_rois.py                # Main execution script
├── data/
│   └── processed/
│       └── roi_patches/               # Output directory
│           ├── images/                # ROI image patches
│           ├── masks/                 # ROI mask patches
│           ├── roi_metadata.csv       # Complete metadata
│           └── statistics.txt         # Summary statistics
├── train_images/                      # Training images (1600×256)
├── test_images/                       # Test images
├── train.csv                          # RLE-encoded annotations
└── PROJECT(roi).md                    # Research documentation

```

## Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Extract ROIs from Dataset

**Test with 10 images:**
```bash
python scripts/extract_rois.py --max_images 10
```

**Process all images:**
```bash
python scripts/extract_rois.py
```

**Custom parameters:**
```bash
python scripts/extract_rois.py \
    --image_dir train_images \
    --train_csv train.csv \
    --output_dir data/processed/roi_patches \
    --roi_size 512 \
    --grid_size 64 \
    --min_suitability 0.5 \
    --max_images 100
```

**Metadata only (no patches):**
```bash
python scripts/extract_rois.py --no_save_patches --max_images 100
```

## Output

### 1. ROI Metadata CSV (`roi_metadata.csv`)

Columns:
- `image_id`, `class_id`, `region_id`: Identifiers
- `roi_bbox`: (x1, y1, x2, y2) optimized ROI position
- `defect_bbox`: (x1, y1, x2, y2) original defect bounding box
- `centroid`: (x, y) defect center
- `area`: defect area in pixels
- `linearity`, `solidity`, `extent`, `aspect_ratio`: 4 indicators
- `defect_subtype`: Defect classification
- `background_type`: Background classification
- `suitability_score`: Overall match quality (0-1)
- `matching_score`: Defect-background matching (0-1)
- `continuity_score`: Background uniformity (0-1)
- `stability_score`: Background stability (0-1)
- `recommendation`: `suitable` / `acceptable` / `unsuitable`
- `prompt`: Text description for ControlNet
- `roi_image_path`: Path to saved image patch
- `roi_mask_path`: Path to saved mask patch

### 2. Image/Mask Patches

Saved as PNG files:
- Images: `{image_id}_class{class_id}_region{region_id}.png`
- Masks: Same naming convention

### 3. Statistics Summary

Example output:
```
Total ROIs extracted: 1234

ROIs per class:
  Class 1: 456
  Class 2: 321
  Class 3: 289
  Class 4: 168

ROIs per defect subtype:
  linear_scratch: 512
  compact_blob: 387
  elongated: 213
  irregular: 122

ROIs per background type:
  smooth: 445
  vertical_stripe: 356
  textured: 289
  horizontal_stripe: 144

Average scores:
  Suitability: 0.712
  Matching: 0.823
  Continuity: 0.687
```

## Key Insights from PROJECT(roi).md

1. **Quality over Quantity**: Not all defect regions are suitable for augmentation. Only use ROIs with good defect-background matching.

2. **Context Matters**: A linear scratch on a striped background looks natural. The same scratch on a smooth surface may look artificial.

3. **Background Continuity**: ROIs should avoid background boundaries to prevent unnatural transitions in synthetic data.

4. **Position Optimization**: Small shifts (±32px) can significantly improve background uniformity without losing the defect.

5. **Prompt Engineering**: Text prompts combining defect subtype + background type guide ControlNet to generate realistic synthetic defects.

## Research Questions Addressed

- ✅ How to characterize defects beyond simple bounding boxes?
- ✅ How to quantify background suitability for synthetic defect placement?
- ✅ How to match defect types with appropriate backgrounds?
- ✅ How to optimize ROI positioning for maximum data quality?
- ✅ How to package ROI data for downstream ControlNet training?

## Next Steps

1. **Visualization Module**: Create visualization tools to inspect ROI selections
2. **ControlNet Training**: Use extracted ROIs with prompts for conditional generation
3. **Augmentation Pipeline**: Generate synthetic defects using the suitability scores
4. **Validation**: Compare model performance with/without suitability filtering

## References

- PROJECT(roi).md: Complete research methodology (Korean)
- Severstal Steel Defect Detection: Kaggle competition dataset
- skimage.measure.regionprops: Geometric property computation
- OpenCV: Image processing and FFT analysis

---

**Author**: OpenCode AI Assistant  
**Date**: February 2026  
**Research Framework**: PROJECT(roi).md
