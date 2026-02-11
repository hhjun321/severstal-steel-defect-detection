# Quick Verification Script: Background Extraction from Clean Images

## 📋 Purpose

This script verifies that background extraction correctly identifies **clean images** (those NOT in train.csv) and visualizes ROI extraction results.

## ✅ What Was Fixed

### Bug #1: Clean Image Detection (FIXED)
- **OLD (WRONG)**: `train_df[train_df['EncodedPixels'].isna()]` → Returns 0 images
- **NEW (CORRECT)**: `all_images - train_csv_images` → Returns 5,902 clean images

### Bug #2: BackgroundCharacterizer (AVOIDED)
- **Issue**: Code referenced `BackgroundCharacterizer` which doesn't exist
- **Solution**: This script uses `BackgroundAnalyzer` directly (the actual class)

## 🚀 How to Run

### Step 1: Activate Conda Environment
```cmd
cd D:\project\severstal-steel-defect-detection
conda activate PythonProject
```

### Step 2: Run the Script
```cmd
python scripts\quick_verify_clean_backgrounds.py
```

### Expected Runtime
- **~30-60 seconds** for 3 images
- Progress shown in console

## 📊 Expected Output

### Console Output
```
================================================================================
QUICK VERIFICATION: Background Extraction from Clean Images
깨끗한 이미지에서 배경 추출 빠른 검증
================================================================================

RESEARCH PROTOCOL:
- train.csv contains ONLY images WITH defects
- Clean images are NOT in train.csv
- We extract backgrounds from clean images for augmentation

1. Finding clean images...
   Total images in train_images/: 12,568
   Images WITH defects (in train.csv): 6,666
   Clean images (NOT in train.csv): 5,902
   ✓ Found 5,902 clean images

2. Selecting 3 random clean images for visualization...
   1. [image_id_1].jpg
   2. [image_id_2].jpg
   3. [image_id_3].jpg

3. Initializing background analyzer...
   ✓ BackgroundAnalyzer initialized

4. Processing images...
================================================================================

[1/3] Processing [image_id_1].jpg...
   Image size: (256, 1600, 3)
   Analyzing background grid (64×64)...
   Background distribution:
     - smooth: 45 cells (67.2%)
     - vertical_stripe: 15 cells (22.4%)
     - textured: 7 cells (10.4%)
   Selected 3 ROI regions (512×512)
   ✓ Saved: image_1_[image_id_1].jpg.png

[2/3] Processing [image_id_2].jpg...
   ...

[3/3] Processing [image_id_3].jpg...
   ...

5. Creating comparison grid...
   ✓ Saved: comparison_grid.png

================================================================================
✅ VERIFICATION COMPLETE!
================================================================================

📁 Output directory: D:\project\severstal-steel-defect-detection\outputs\clean_bg_verification

Generated files:
  - image_1_[id].png (detailed view with ROIs)
  - image_2_[id].png (detailed view with ROIs)
  - image_3_[id].png (detailed view with ROIs)
  - comparison_grid.png (summary view)
```

### Generated Files
```
outputs/
└── clean_bg_verification/
    ├── image_1_*.png        (Clean image with colored ROI boxes)
    ├── image_2_*.png        (Clean image with colored ROI boxes)
    ├── image_3_*.png        (Clean image with colored ROI boxes)
    └── comparison_grid.png  (All 3 images side-by-side)
```

### Visualization Details

Each `image_X_*.png` shows:
- **Left panel**: Full image with colored ROI boxes (512×512)
- **Right panels**: Extracted ROI patches with metadata
- **Colors**:
  - 🟢 Green = smooth (uniform surface)
  - 🔵 Blue = vertical_stripe
  - 🔴 Red = horizontal_stripe
  - 🟠 Orange = textured
  - 🟡 Yellow = complex_pattern

## ✅ Verification Checklist

After running the script, verify:

- [ ] **Console shows correct statistics**:
  - Total: 12,568 images
  - Defects: 6,666 images
  - Clean: 5,902 images

- [ ] **4 files created in `outputs/clean_bg_verification/`**:
  - 3 individual image files
  - 1 comparison grid

- [ ] **Visual inspection**:
  - Images appear clean (no visible defects)
  - ROI boxes are well-positioned
  - Different colors for different background types
  - Boxes don't overlap or go out of bounds

- [ ] **No errors in console**:
  - Script completes successfully
  - All imports work correctly
  - Files saved without errors

## 🔍 What to Look For

### 1. Confirm Clean Images
- Open the visualization files
- **CRITICAL**: Images should have NO visible defects
- If you see defects, something went wrong!

### 2. Verify ROI Extraction
- ROI boxes should cover representative background regions
- Different colors = different background types
- Boxes should be 512×512 pixels
- Stability scores should be between 0.0 and 1.0

### 3. Check Background Types
- **Smooth** (green): Flat, uniform areas
- **Vertical stripe** (blue): Vertical patterns/lines
- **Horizontal stripe** (red): Horizontal patterns/lines
- **Textured** (orange): Regular textures
- **Complex pattern** (yellow): Multi-directional patterns

## 📝 Next Steps

### If Verification Succeeds ✅

**Step 1**: Extract backgrounds from 100 clean images
```cmd
python scripts\run_background_extraction.py --max-images 100
```
Expected: ~500 background patches in `data/processed/background_patches/`

**Step 2**: Verify extracted patches
```cmd
dir /B data\processed\background_patches\*.png | find /C ".png"
```
Should show ~500 files

**Step 3**: Generate augmented dataset
```cmd
python scripts\stage3_generate_augmentations.py --n-samples 1000
```
Expected: 1000 augmented samples in `data/augmented/`

### If Verification Fails ❌

**Problem 1**: Import errors (cv2, pandas, numpy, etc.)
```cmd
# Check if packages are installed
conda list | findstr opencv
conda list | findstr pandas
conda list | findstr numpy
conda list | findstr matplotlib

# If missing, install:
conda install opencv pandas numpy matplotlib
```

**Problem 2**: Script shows wrong image counts
- Check if `train.csv` and `train_images/` are in correct location
- Verify file structure matches expected format

**Problem 3**: Images have defects visible
- This means clean image detection is still wrong
- Report the issue for further debugging

**Problem 4**: BackgroundAnalyzer not found
```cmd
# Verify src/analysis/background_characterization.py exists
dir src\analysis\background_characterization.py
```

## 🔧 Troubleshooting

### Error: "No module named 'analysis'"
**Solution**: Make sure you're running from the project root:
```cmd
cd D:\project\severstal-steel-defect-detection
python scripts\quick_verify_clean_backgrounds.py
```

### Error: "train.csv not found"
**Solution**: Verify train.csv exists:
```cmd
dir train.csv
```
If missing, you need to download the dataset.

### Error: "No clean images found"
**Solution**: This is a serious issue. Verify:
```cmd
# Count total images
dir /B train_images\*.jpg | find /C ".jpg"
# Should show: 12568

# Count unique images in train.csv
python -c "import pandas as pd; print(len(pd.read_csv('train.csv')['ImageId'].unique()))"
# Should show: 6666
```

### Error: Script runs but no files generated
**Solution**: Check if output directory was created:
```cmd
dir outputs\clean_bg_verification
```

## 📚 Technical Details

### Clean Image Detection Algorithm
```python
# Get all image files from directory
all_images = set([f.name for f in train_images_dir.glob("*.jpg")])

# Get images listed in train.csv (images WITH defects)
train_df = pd.read_csv(train_csv_path)
defect_images = set(train_df['ImageId'].unique())

# Clean images = images NOT in train.csv
clean_images = list(all_images - defect_images)
```

### Background Analysis Algorithm
```python
# 1. Analyze entire image using 64×64 grid
analysis = analyzer.analyze_image(img)

# 2. Classify each grid cell into background types
bg_map = analysis['background_map']        # (grid_h, grid_w)
stability_map = analysis['stability_map']  # (grid_h, grid_w)

# 3. For each background type, find best ROI location
for target_type in ['smooth', 'vertical_stripe', ...]:
    # Find grid cells matching this type
    matches = np.argwhere(bg_map == target_type)
    
    # Select cell with highest stability
    best_cell = matches[np.argmax(stability_map[matches])]
    
    # Extract 512×512 ROI centered on this cell
    roi = extract_roi(image, best_cell, size=512)
```

### Stability Score Computation
```python
# Stability = quality of background for defect placement
# Higher stability = better background
# Range: 0.0 (unstable) to 1.0 (very stable)

stability = 0.4 × variance_score + 0.3 × consistency_score + 0.3 × edge_score

Where:
  variance_score = 1 / (1 + variance/1000)
  consistency_score = uniformity across region
  edge_score = 1 - edge_density
```

## 🎯 Success Criteria

This verification succeeds if:
1. ✅ Script finds exactly 5,902 clean images
2. ✅ All 3 selected images are visually clean (no defects)
3. ✅ ROI boxes are properly positioned and colored
4. ✅ Background types make sense visually
5. ✅ All files are generated successfully

Once verified, you can proceed with full background extraction from all 5,902 clean images!

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all paths and file structures
3. Ensure Conda environment is activated
4. Check that all dependencies are installed

---

**Created**: 2026-02-10  
**Script**: `scripts/quick_verify_clean_backgrounds.py`  
**Purpose**: Verify clean image detection for background extraction  
**Status**: Ready to run
