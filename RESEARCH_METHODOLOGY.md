# CASDA Research Methodology
# CASDA 연구 방법론

**Date**: February 10, 2026  
**Project**: Context-Aware Steel Defect Augmentation (CASDA)

---

## 🎯 Core Research Protocol / 핵심 연구 원칙

### Data Separation Principle / 데이터 분리 원칙

```
┌──────────────────────────────────────────────────────────────────┐
│                    DATA SPLIT STRATEGY                            │
└──────────────────────────────────────────────────────────────────┘

ALL IMAGES (12,568 total)
├── train.csv images (6,666 images)
│   ├── Purpose: Images WITH defects
│   ├── Used for: 
│   │   - Stage 1: ROI Extraction (defect templates)
│   │   - Stage 2: ControlNet Training Data Preparation
│   │   - Final model training
│   └── Contains: RLE-encoded defect masks (4 classes)
│
├── Clean images (5,902 images) ← NOT in train.csv
│   ├── Purpose: Images WITHOUT defects (defect-free)
│   ├── Used for:
│   │   - Stage 3: Background extraction
│   │   - Synthetic defect placement
│   │   - Augmented dataset generation
│   └── Characteristics: No entry in train.csv
│
└── test.csv images
    ├── Purpose: Validation set
    ├── Used for: Final model evaluation
    └── Never used during training

CRITICAL RULE:
───────────────────────────────────────────────────────────────────
train.csv ONLY contains images WITH defects
Clean images are NOT listed in train.csv
To find clean images: all_images - train.csv_images
```

---

## 📊 Dataset Statistics / 데이터셋 통계

```
Total Training Images:      12,568
├── With Defects (train.csv):  6,666 (53.0%)
└── Clean (not in train.csv):  5,902 (47.0%)

Defect Classes in train.csv:
├── Class 1:  ~1,800 samples
├── Class 2:  ~1,900 samples
├── Class 3:  ~2,500 samples
└── Class 4:  ~1,000 samples (rare)

Average defects per image: ~1.1
```

---

## 🔬 CASDA Pipeline with Correct Methodology / 올바른 방법론의 CASDA 파이프라인

### Stage 1: ROI Extraction from Defect Images
**Input**: train.csv images (6,666 images WITH defects)

```python
# Extract defect ROI patches from images in train.csv
for image_id in train_csv['ImageId'].unique():
    # Load image with defects
    image = load_image(image_id)
    defect_masks = get_masks(train_csv, image_id)
    
    # Extract ROI around each defect
    for defect_mask in defect_masks:
        roi_patch = extract_roi(image, defect_mask)
        analyze_defect(roi_patch)  # 4 indicators
        classify_background(roi_patch)  # 5 types
        compute_suitability(roi_patch)
        save_roi_template(roi_patch)
```

**Output**:
- `data/processed/roi_patches/roi_metadata.csv` (3,247 defect ROIs)
- Defect templates with masks

---

### Stage 2: ControlNet Data Preparation
**Input**: ROI metadata from Stage 1

```python
# Prepare ControlNet training data
for roi in roi_metadata:
    hint = generate_multi_channel_hint(roi)  # R/G/B channels
    prompt = generate_hybrid_prompt(roi)
    package_for_controlnet(hint, prompt)
```

**Output**:
- `data/controlnet_data/train.jsonl`
- ControlNet training ready

---

### Stage 3: Background Extraction from Clean Images ⭐ KEY STAGE
**Input**: Clean images (NOT in train.csv) - 5,902 images

```python
# Find clean images
all_images = set(glob("train_images/*.jpg"))
defect_images = set(train_csv['ImageId'].unique())
clean_images = all_images - defect_images  # 5,902 images

# Extract backgrounds from clean images
for clean_image in clean_images:
    # This image has NO defects (not in train.csv)
    image = load_image(clean_image)
    
    # Analyze background types (no defect masks needed)
    background_grid = classify_background_grid(image)  # 64×64 grid
    
    # Extract diverse background ROIs
    for bg_type in ['smooth', 'vertical_stripe', 'horizontal_stripe', 
                    'textured', 'complex_pattern']:
        roi = select_best_roi(image, background_grid, bg_type)
        compute_stability(roi)
        save_background_template(roi)
```

**Output**:
- `data/processed/background_patches/background_metadata.csv`
- ~25,000+ background templates (5 per image × 5,000 images)

**Why Clean Images?**
- ✅ Provides defect-free backgrounds
- ✅ No need to avoid defect regions
- ✅ Maximum usable area per image
- ✅ Higher quality backgrounds

---

### Stage 3B: Augmented Dataset Generation
**Input**: 
- Background templates (from clean images)
- Defect templates (from train.csv images)

```python
# Match backgrounds with defects using compatibility matrix
for n in range(n_augmented_samples):
    # Select defect template (from train.csv ROIs)
    defect = select_defect_template(class_id, defect_type)
    
    # Find compatible background (from clean images)
    backgrounds = find_compatible_backgrounds(defect.type)
    background = select_best_match(backgrounds, defect)
    
    # Generate augmented sample
    hint = generate_controlnet_hint(background, defect.mask)
    # R: defect mask, G: edges, B: texture
    
    save_augmentation(background, defect.mask, hint)
```

**Output**:
- `data/augmented/images/*.png` - Background patches
- `data/augmented/masks/*.png` - Defect masks
- `data/augmented/hints/*.png` - ControlNet hints
- `data/augmented/augmentation_metadata.csv`

---

### Stage 4: Quality Validation
**Input**: Augmented dataset from Stage 3B

```python
# Validate quality of augmented samples
for augmented_sample in augmented_dataset:
    quality_scores = compute_quality_metrics(augmented_sample)
    if quality_scores.overall >= threshold:
        accept_sample(augmented_sample)
```

---

### Stage 5: Dataset Merging & Training
**Input**: 
- Original train.csv images (6,666)
- Validated augmented samples (~1,000)

```python
# Merge datasets
merged_dataset = original_dataset + augmented_dataset

# Train model
model = train_segmentation_model(merged_dataset)

# Evaluate on test.csv
results = evaluate_model(model, test_csv)
```

---

## 🔑 Key Differences from Initial Implementation / 초기 구현과의 주요 차이점

### ❌ WRONG (Initial Implementation)
```python
# INCORRECT: Looking for clean images in train.csv
clean_images = train_df[train_df['EncodedPixels'].isna()]['ImageId']
# Result: Empty list (train.csv has no NaN pixels!)
```

### ✅ CORRECT (Fixed Implementation)
```python
# CORRECT: Finding images NOT in train.csv
all_images = set([f.name for f in Path("train_images").glob("*.jpg")])
defect_images = set(pd.read_csv("train.csv")['ImageId'].unique())
clean_images = list(all_images - defect_images)
# Result: 5,902 clean images
```

---

## 📈 Expected Results / 예상 결과

### Background Extraction (Stage 3A)
```
Input:  5,902 clean images
Output: ~25,000 background ROI templates
        - smooth: ~5,000 templates
        - vertical_stripe: ~7,000 templates
        - horizontal_stripe: ~5,000 templates
        - textured: ~5,000 templates
        - complex_pattern: ~3,000 templates

Quality Distribution:
  High (≥0.8):     ~40%
  Medium (0.6-0.8): ~50%
  Low (<0.6):      ~10%
```

### Augmentation Generation (Stage 3B)
```
Input:  25,000 backgrounds × 3,247 defect templates
Matching: Compatibility-based selection
Output: 1,000-10,000 augmented samples (configurable)

Quality Metrics:
  Mean compatibility:  0.75
  Mean suitability:    0.68
  Mean stability:      0.72
```

---

## 🎓 Research Contribution / 연구 기여

### Novel Aspects / 새로운 측면

1. **Clean-Defect Separation**
   - Uses clean images for backgrounds (not in train.csv)
   - Uses defect images for templates (in train.csv)
   - Ensures maximum background quality

2. **Compatibility-Based Matching**
   - Smart pairing of backgrounds and defect types
   - Optimizes visual contrast and detectability

3. **Multi-Stage Quality Control**
   - Background stability scoring
   - Defect ROI suitability scoring
   - Compatibility scoring
   - Final quality validation

---

## 🔬 Experimental Validation / 실험 검증

### Hypothesis / 가설
Augmenting training data with synthetic defects on clean backgrounds improves model performance, especially for rare defect classes.

### Validation Protocol / 검증 프로토콜

```
1. Baseline Model (No Augmentation)
   - Train on original train.csv only (6,666 images)
   - Evaluate on test.csv
   - Record: Dice score, IoU, Precision, Recall per class

2. Augmented Model (With CASDA)
   - Train on: train.csv + augmented samples (6,666 + 1,000)
   - Evaluate on test.csv
   - Record: Same metrics

3. Comparison
   - Compute improvement: Δ_metric = metric_augmented - metric_baseline
   - Statistical significance testing (t-test)
   - Analyze per-class improvements (especially Class 4)
```

### Expected Improvements / 예상 개선

```
Overall Dice Score:     +3-5%
Class 1:                +2-3%
Class 2:                +2-3%
Class 3:                +3-4%
Class 4 (rare):         +8-12% (most improvement)

Training Time:          +15-20% (acceptable tradeoff)
Inference Time:         No change (same model architecture)
```

---

## 📝 Code Implementation Status / 코드 구현 상태

### ✅ Completed / 완료

- [x] Stage 1: ROI Extraction (roi_extraction.py)
- [x] Stage 2: ControlNet Data Prep (controlnet_packager.py)
- [x] Stage 3A: Background Extraction (**FIXED** - background_extraction.py)
- [x] Stage 3B: Augmentation Generation (augmentation_generator.py)
- [x] Background Library & Search (background_library.py)
- [x] Execution Scripts (run_background_extraction.py, stage3_generate_augmentations.py)

### 🔄 Pending / 대기 중

- [ ] Stage 4: Quality Validation Implementation
- [ ] Stage 5: Dataset Merging Implementation
- [ ] Model Training Pipeline
- [ ] test.csv Evaluation Pipeline

---

## 🚀 Quick Start Commands / 빠른 시작 명령어

### Step 1: Extract Defect ROIs (from train.csv images)
```bash
# Already completed in previous work
# Output: data/processed/roi_patches/roi_metadata.csv (3,247 ROIs)
```

### Step 2: Extract Clean Backgrounds (from images NOT in train.csv)
```bash
python scripts/run_background_extraction.py \
    --max-images 1000 \
    --min-stability 0.6

# Expected: ~5,000 background templates from 1,000 clean images
```

### Step 3: Generate Augmented Dataset
```bash
python scripts/stage3_generate_augmentations.py \
    --n-samples 1000 \
    --class-distribution 0.25,0.25,0.35,0.15

# Expected: 1,000 augmented samples (images + masks + hints)
```

### Step 4: Verify Results
```bash
# Check outputs
ls data/processed/background_patches/*.png | wc -l  # ~5,000
ls data/augmented/images/*.png | wc -l              # 1,000
ls data/augmented/masks/*.png | wc -l               # 1,000
ls data/augmented/hints/*.png | wc -l               # 1,000
```

---

## 📚 References / 참고 문헌

1. **CASDA Technical Whitepaper**: `TECHNICAL_WHITEPAPER_EN.md`
2. **CASDA Research Paper**: `RESEARCH_REPORT_EN.md`
3. **Stage 3 Implementation**: `STAGE3_IMPLEMENTATION_GUIDE.md`
4. **Original Pipeline Design**: `PROJECT(roi).md`

---

## ✅ Verification Checklist / 검증 체크리스트

- [x] Clean images correctly identified (NOT in train.csv)
- [x] Clean image count verified (5,902 images)
- [x] Background extraction uses ONLY clean images
- [x] Defect ROI extraction uses ONLY train.csv images
- [x] No defect mask loading in background extraction
- [x] Compatibility matrix implemented
- [x] Multi-channel hint generation implemented
- [x] All scripts updated with correct methodology
- [ ] End-to-end pipeline tested
- [ ] Results validated on test.csv

---

**Document Status**: ✅ COMPLETE  
**Last Updated**: February 10, 2026  
**Version**: 2.0 (Corrected Methodology)

---

**END OF RESEARCH METHODOLOGY DOCUMENT**
