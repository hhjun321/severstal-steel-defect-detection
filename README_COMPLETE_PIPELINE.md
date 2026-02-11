# Severstal Steel Defect Detection - ControlNet Data Preparation Pipeline

Complete implementation of the research pipeline described in `PROJECT(roi).md` and `PROJECT(prepare_control).md` for preparing high-quality ControlNet training data from the Severstal Steel Defect Detection dataset.

## Project Overview

This project implements a novel approach to defect data augmentation by carefully matching defect characteristics with appropriate background contexts. The key insight is: **"Which defect subclass" + "Which background context" = High-quality synthetic data**.

## Research Papers Implemented

### 1. PROJECT(roi).md - Statistical Indicator-Based ROI Extraction
Extracts Regions of Interest (ROIs) by analyzing both defect geometry and background suitability.

### 2. PROJECT(prepare_control).md - ControlNet Training Data Preparation
Packages extracted ROIs into ControlNet training format with multi-channel hints and hybrid prompts.

## Complete Pipeline

```
Raw Data (Images + RLE Masks)
  ↓
┌─────────────────────────────────────────────────┐
│ Stage 1: ROI Extraction (PROJECT(roi).md)      │
├─────────────────────────────────────────────────┤
│ 1. Background Analysis (Grid-based)            │
│    - Classify: smooth, textured, stripe, etc.  │
│    - Compute stability scores                  │
│ 2. Defect Analysis (4 Indicators)              │
│    - Linearity, Solidity, Extent, Aspect Ratio │
│    - Classify subtypes                         │
│ 3. ROI Suitability Evaluation                  │
│    - Match defect-background combinations      │
│    - Optimize ROI positioning                  │
│ 4. Extract & Package ROIs                      │
│    - 512×512 patches with metadata             │
└─────────────────────────────────────────────────┘
  ↓ ROI Metadata CSV
┌─────────────────────────────────────────────────┐
│ Stage 2: ControlNet Prep (prepare_control.md)  │
├─────────────────────────────────────────────────┤
│ 1. Multi-Channel Hint Generation               │
│    - Red: Defect mask (4-indicator enhanced)   │
│    - Green: Background structure lines         │
│    - Blue: Background texture                  │
│ 2. Hybrid Prompt Generation                    │
│    - Combine defect + background descriptions  │
│ 3. Dataset Validation                          │
│    - Distribution check                        │
│    - Visual inspection                         │
│ 4. Package for Training                        │
│    - train.jsonl + hints/ + metadata           │
└─────────────────────────────────────────────────┘
  ↓
ControlNet Training Ready Dataset
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Stage 1: Extract ROIs

```bash
# Test with 10 images
python scripts/extract_rois.py --max_images 10

# Process all images
python scripts/extract_rois.py
```

**Output**: `data/processed/roi_patches/roi_metadata.csv` + ROI image/mask patches

### Stage 2: Prepare ControlNet Data

```bash
# Full pipeline with validation
python scripts/prepare_controlnet_data.py \
    --roi_metadata data/processed/roi_patches/roi_metadata.csv

# Quick test (skip hints for speed)
python scripts/prepare_controlnet_data.py \
    --roi_metadata data/processed/roi_patches/roi_metadata.csv \
    --skip_hints \
    --max_samples 50
```

**Output**: `data/processed/controlnet_dataset/train.jsonl` + multi-channel hints

## Project Structure

```
severstal-steel-defect-detection/
├── src/
│   ├── utils/
│   │   ├── rle_utils.py              # RLE encoding/decoding
│   │   └── dataset_validator.py      # Quality validation
│   ├── analysis/
│   │   ├── defect_characterization.py    # 4 indicators
│   │   ├── background_characterization.py # Grid-based analysis
│   │   └── roi_suitability.py           # Matching evaluation
│   └── preprocessing/
│       ├── roi_extraction.py           # Stage 1 pipeline
│       ├── hint_generator.py           # Multi-channel hints
│       ├── prompt_generator.py         # Hybrid prompts
│       └── controlnet_packager.py      # Stage 2 pipeline
├── scripts/
│   ├── extract_rois.py                # Stage 1 CLI
│   └── prepare_controlnet_data.py     # Stage 2 CLI
├── data/
│   └── processed/
│       ├── roi_patches/               # Stage 1 output
│       └── controlnet_dataset/        # Stage 2 output
├── train_images/                      # Raw training images
├── train.csv                          # RLE annotations
├── PROJECT(roi).md                    # Research doc 1
├── PROJECT(prepare_control).md        # Research doc 2
├── IMPLEMENTATION_SUMMARY_KR.md       # Stage 1 summary
└── IMPLEMENTATION_CONTROLNET_PREP_KR.md # Stage 2 summary
```

## Key Features

### Stage 1: ROI Extraction

**1. Defect Characterization (4 Indicators)**
- Linearity: How linear/elongated (eigenvalue analysis)
- Solidity: Compactness (area/convex_hull)
- Extent: Bounding box filling (area/bbox)
- Aspect Ratio: Elongation (major/minor axis)

**Subtype Classification**:
- `linear_scratch`: High linearity + high aspect ratio
- `compact_blob`: Low aspect ratio + high solidity
- `elongated`: High aspect ratio + medium linearity
- `irregular`: Low solidity
- `general`: Default

**2. Background Characterization (Grid-Based)**

Analyzes 64×64 grid patches:
- **Variance** → smooth vs textured
- **Sobel edges** → vertical/horizontal stripe
- **FFT** → complex patterns
- **Stability score** → uniformity

**3. ROI Suitability Scoring**

```
Suitability = 0.5×matching + 0.3×continuity + 0.2×stability

Matching rules:
- linear_scratch + (vertical|horizontal)_stripe → 1.0
- compact_blob + smooth → 1.0
- irregular + complex_pattern → 1.0
```

**4. Position Optimization**

Shifts ROI window (±32px) to maximize background continuity while keeping defect centered.

### Stage 2: ControlNet Preparation

**1. Multi-Channel Hint Images**

- **Red**: Defect mask enhanced by indicators
  - High linearity → Skeleton extraction
  - High solidity → Filled mask
  - Otherwise → Edge-emphasized
- **Green**: Background structure (Sobel edges)
  - Vertical stripe → Vertical edges
  - Horizontal stripe → Horizontal edges
  - Complex → All edges
- **Blue**: Background texture (local variance)
  - Smooth → Low values
  - Textured → High values

**2. Hybrid Prompts**

Structure: `[Defect characteristics] + [Background type] + [Surface condition]`

Example (detailed style):
```
"a high-linearity elongated scratch on vertical striped metal surface 
with directional texture (pristine condition), steel defect class 1"
```

**3. Dataset Validation**

- **Distribution Check**: Detect class/subtype/background imbalance (>50-60%)
- **Visual Check**: Sample inspection for edge positioning and quality issues

**4. ControlNet Format**

`train.jsonl`:
```json
{
  "source": "path/to/roi_image.png",
  "target": "path/to/roi_image.png",
  "prompt": "detailed description...",
  "hint": "path/to/hint_image.png",
  "negative_prompt": "blurry, low quality..."
}
```

## Output Examples

### Stage 1 Output

`roi_metadata.csv` columns:
- `image_id`, `class_id`, `region_id`
- `roi_bbox`, `defect_bbox`, `centroid`, `area`
- `linearity`, `solidity`, `extent`, `aspect_ratio`
- `defect_subtype`, `background_type`
- `suitability_score`, `matching_score`, `continuity_score`, `stability_score`
- `recommendation` (suitable/acceptable/unsuitable)

### Stage 2 Output

```
controlnet_dataset/
├── hints/                    # RGB hint images
│   └── {image_id}_class{id}_region{id}_hint.png
├── validation/              # Quality reports
│   ├── distribution_analysis.png
│   └── visual_inspection.png
├── train.jsonl              # Training index
├── metadata.json            # Complete metadata
└── packaged_roi_metadata.csv
```

## Usage Examples

### Custom ROI Extraction

```bash
python scripts/extract_rois.py \
    --image_dir train_images \
    --train_csv train.csv \
    --output_dir data/processed/roi_patches \
    --roi_size 512 \
    --grid_size 64 \
    --min_suitability 0.6 \
    --max_images 1000
```

### Custom ControlNet Preparation

```bash
python scripts/prepare_controlnet_data.py \
    --roi_metadata data/processed/roi_patches/roi_metadata.csv \
    --output_dir data/processed/controlnet_dataset \
    --prompt_style technical \
    --validation_samples 32 \
    --skip_hints  # For quick prompt-only generation
```

## Research Contributions

### Methodological Innovations

1. **Context-Aware ROI Selection**
   - Not just "where defects are" but "where defects fit naturally"
   - Defect-background matching rules based on physical plausibility

2. **Multi-Channel Conditioning**
   - Beyond binary masks: 3-channel hints encode defect geometry + background context
   - Richer conditioning signal for ControlNet

3. **Automated Quality Control**
   - Statistical distribution checks
   - Visual inspection with automatic issue detection

4. **End-to-End Automation**
   - From raw RLE masks to ControlNet-ready dataset
   - Reproducible and scalable

### Practical Benefits

- **Higher quality synthetic data**: Context-appropriate defect placement
- **Better model generalization**: Diverse defect-background combinations
- **Faster iteration**: Automated validation catches issues early
- **Standardized format**: Compatible with existing ControlNet frameworks

## Performance Metrics

After running the pipeline, check:

1. **ROI Extraction** (`roi_metadata.csv`):
   - Total ROIs extracted
   - Distribution across classes and subtypes
   - Average suitability score (target: >0.7)

2. **Validation** (`validation/`):
   - Class balance (no class >60%)
   - Subtype coverage (all ideal combinations present)
   - Visual inspection pass rate (target: >80%)

3. **Final Dataset** (`train.jsonl`):
   - Total training samples
   - Prompt diversity
   - Hint image quality

## Next Steps

After completing this pipeline:

1. **Review Outputs**
   - Check validation reports
   - Inspect sample hint images
   - Verify prompts are descriptive

2. **ControlNet Training**
   - See `PROJECT(control_net).md` (to be implemented)
   - Use `train.jsonl` as training index
   - Configure base model and hyperparameters

3. **Evaluation**
   - Generate synthetic defects
   - Compare with baseline augmentation
   - Measure downstream segmentation performance

## Troubleshooting

### Issue: "ROI metadata not found"
Run Stage 1 first: `python scripts/extract_rois.py`

### Issue: "Distribution warnings"
- Check validation reports in `validation/`
- Consider adjusting `min_suitability` threshold
- May need more data or rebalancing

### Issue: Low suitability scores
- Check `roi_metadata.csv` for unsuitable samples
- Increase `min_suitability` to filter more strictly
- Review matching rules in `roi_suitability.py`

## References

- **Dataset**: [Severstal Steel Defect Detection (Kaggle)](https://www.kaggle.com/c/severstal-steel-defect-detection)
- **ControlNet**: [Adding Conditional Control to Text-to-Image Diffusion Models (Zhang et al., 2023)](https://arxiv.org/abs/2302.05543)
- **Research Docs**: 
  - `PROJECT(roi).md` - ROI extraction methodology
  - `PROJECT(prepare_control).md` - ControlNet preparation methodology

## Citation

If you use this code or methodology, please cite:

```
@misc{severstal-controlnet-prep-2026,
  title={Context-Aware ROI Extraction and Multi-Channel Conditioning for Steel Defect Synthesis},
  author={OpenCode AI Assistant},
  year={2026},
  note={Implementation of PROJECT(roi).md and PROJECT(prepare_control).md}
}
```

---

**Authors**: OpenCode AI Assistant  
**Date**: February 2026  
**License**: MIT  
**Status**: Research Implementation ✅ Complete

For questions or issues, please refer to the detailed implementation summaries:
- `IMPLEMENTATION_SUMMARY_KR.md` (Stage 1)
- `IMPLEMENTATION_CONTROLNET_PREP_KR.md` (Stage 2)
