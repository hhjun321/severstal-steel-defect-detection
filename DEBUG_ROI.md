# ROI Detection Debugging Guide

## Problem
ROI detection is returning 0 ROIs even after fixing variance_threshold to 100.0.

## Debug Steps

### Step 1: Run the updated script with debug output
```bash
python show_clean_roi_boxes.py
```

The script now prints background type distribution for each image:
```
Processing: image.jpg
  Image size: 1600x256
  Background types found: ['smooth' 'textured' 'vertical_stripe']
    - smooth: 5 cells
    - textured: 10 cells
    - vertical_stripe: 8 cells
  Selected ROIs: 3
```

### Step 2: Run basic diagnostics
```bash
python test_basic.py
```

This will check:
- If paths are correct
- If images can be loaded
- How many clean images exist

### Step 3: Run detailed debug
```bash
python debug_roi_detection.py
```

This will show:
- Background type distribution
- Stability scores for each type
- ROI extraction attempts
- Why ROIs are rejected (out of bounds, etc.)

## Expected Behavior

With variance_threshold=100.0, you should see:
- Multiple background types detected (smooth, textured, vertical_stripe, etc.)
- ROIs extracted from different background types
- At least 2-5 ROIs per image

## If Still Getting 0 ROIs

Possible causes:
1. **Image dimensions too small**: If images are smaller than ROI size (512x512), no ROIs can fit
   - Check: Image size should be at least 512x512
   
2. **Grid size too large**: If grid_size=64 and image is small, there might be too few grid cells
   - Check: grid_h and grid_w should be reasonable (at least 8x8)

3. **All cells classified as same type with low stability**: Unlikely but possible
   - Check: Background type distribution should show variety

4. **ROI bounds check failing**: ROI position calculation might be placing ROIs out of bounds
   - Check: Debug output shows why ROIs are rejected

## Quick Fix Options

If images are too small:
```python
# In show_clean_roi_boxes.py, line 53
def analyze_and_visualize(image_id, roi_size=256, grid_size=32, min_rois=2):
    # Reduced roi_size from 512 to 256
    # Reduced grid_size from 64 to 32
```

## Contact
If issue persists after these steps, share the debug output for further analysis.
