ROI Visualization Summary
================================================================================
Generated: 2026-02-10 09:03:41

DATASET OVERVIEW
================================================================================
Total ROIs: 272

ROIs per Class:
  - Class 1: 50 samples
  - Class 2: 2 samples ⚠️ (Very Low!)
  - Class 3: 189 samples
  - Class 4: 31 samples

ROIs per Defect Subtype:
  - general: 121 samples
  - compact_blob: 83 samples
  - linear_scratch: 62 samples
  - irregular: 6 samples

ROIs per Background Type:
  - complex_pattern: 171 samples
  - vertical_stripe: 47 samples
  - smooth: 29 samples
  - horizontal_stripe: 24 samples
  - textured: 1 samples

QUALITY STATISTICS
================================================================================
Suitability Score:
  - Mean: 0.6689
  - Median: 0.6661
  - Min: 0.5001
  - Max: 0.9282

Quality Distribution:
  - High (>0.8): 36 samples
  - Medium (0.6-0.8): 173 samples
  - Low (<0.6): 63 samples

Recommendation Distribution:
  - acceptable: 182 samples
  - suitable: 90 samples

GENERATED FILES
================================================================================
1. 01_class_grids.png
   - Class-wise ROI sample grid
   - Shows representative samples from each class

2. 02_defect_type_grids.png
   - Defect type-wise ROI sample grid
   - Groups by linear_scratch, compact_blob, irregular, general

3. 03_top_quality.png
   - Top 20 high-quality ROIs
   - Sorted by suitability_score

4. 04_class2_detailed.png
   - Detailed analysis of Class 2 samples
   - Shows original image, ROI patch, mask, and full metadata

5. 05_statistics.png
   - Statistical dashboard with 6 charts
   - Distribution analysis and cross-tabulations

6. 06_roi_overlays/
   - Original images with ROI bounding boxes
   - Color-coded by class

WARNINGS & RECOMMENDATIONS
================================================================================

⚠️  CRITICAL: Class 2 has only 2 samples!
    This severe class imbalance may cause training issues.
    Recommendations:
    - Review extraction parameters (min_suitability threshold)
    - Check if Class 2 defects exist in the original dataset
    - Consider data augmentation strategies
    - May need to adjust ROI extraction criteria

NEXT STEPS
================================================================================
1. Review visualizations to verify ROI quality
2. Check Class 2 samples manually (if count is low)
3. Adjust extraction parameters if needed:
   - min_suitability threshold
   - ROI size
   - Background/defect matching rules
4. Proceed to ControlNet data preparation if satisfied

For questions or issues, refer to:
- README_ROI_KR.md
- IMPLEMENTATION_SUMMARY_KR.md
- PROJECT(roi).md
