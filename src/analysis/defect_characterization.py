"""
Defect Characterization Module

This module analyzes defect masks and computes geometric statistical indicators:
- Linearity: How linear/elongated the defect is
- Solidity: Ratio of defect area to its convex hull area
- Extent: Ratio of defect area to its bounding box area
- Aspect Ratio: Width to height ratio of the bounding box

These indicators are used for defect sub-classification (PROJECT(roi).md step 1).
"""
import numpy as np
from skimage import measure
from typing import Dict, List, Tuple, Optional
import cv2


class DefectCharacterizer:
    """
    Analyzes defect masks and computes geometric properties.
    """
    
    def __init__(self):
        pass
    
    def compute_linearity(self, region) -> float:
        """
        Compute linearity score using eigenvalues of the covariance matrix.
        
        Linearity measures how elongated/linear a defect is:
        - Value close to 1.0: highly linear (e.g., scratches)
        - Value close to 0.0: compact/circular (e.g., spots)
        
        Args:
            region: Region properties from skimage.measure.regionprops
            
        Returns:
            Linearity score in range [0, 1]
        """
        # Get coordinates of defect pixels
        coords = region.coords
        
        if len(coords) < 3:
            return 0.0
        
        # Compute covariance matrix
        coords_centered = coords - coords.mean(axis=0)
        cov_matrix = np.cov(coords_centered.T)
        
        # Get eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        
        # Linearity = (λ1 - λ2) / λ1
        # where λ1 is the largest eigenvalue, λ2 is the second largest
        if eigenvalues[0] < 1e-6:
            return 0.0
        
        linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
        return float(linearity)
    
    def compute_solidity(self, region) -> float:
        """
        Compute solidity: ratio of defect area to convex hull area.
        
        Solidity measures how "solid" a defect is:
        - Value close to 1.0: solid, no holes or concavities
        - Lower values: defect has holes or irregular boundaries
        
        Args:
            region: Region properties from skimage.measure.regionprops
            
        Returns:
            Solidity score in range [0, 1]
        """
        return float(region.solidity)
    
    def compute_extent(self, region) -> float:
        """
        Compute extent: ratio of defect area to bounding box area.
        
        Extent measures how much the defect fills its bounding box:
        - Value close to 1.0: defect fills most of the bounding box
        - Lower values: defect is sparse within bounding box
        
        Args:
            region: Region properties from skimage.measure.regionprops
            
        Returns:
            Extent score in range [0, 1]
        """
        return float(region.extent)
    
    def compute_aspect_ratio(self, region) -> float:
        """
        Compute aspect ratio: major axis length / minor axis length.
        
        Aspect ratio measures the elongation of the defect:
        - Value close to 1.0: roughly circular/square
        - High values: very elongated (e.g., long scratches)
        
        Args:
            region: Region properties from skimage.measure.regionprops
            
        Returns:
            Aspect ratio (>= 1.0)
        """
        minor_axis = region.minor_axis_length
        major_axis = region.major_axis_length
        
        if minor_axis < 1e-6:
            return float(major_axis)
        
        return float(major_axis / minor_axis)
    
    def analyze_defect_region(self, mask: np.ndarray, region_id: Optional[int] = None) -> Optional[Dict]:
        """
        Analyze a single defect region and compute all 4 indicators.
        
        Args:
            mask: Binary mask (H, W) containing a single defect
            region_id: Optional region identifier
            
        Returns:
            Dictionary containing all computed metrics, or None if no defect found
        """
        # Label connected components
        labeled_mask = measure.label(mask, connectivity=2)
        regions = measure.regionprops(labeled_mask)
        
        if len(regions) == 0:
            return None
        
        # Use the largest region if multiple exist
        region = max(regions, key=lambda r: r.area)
        
        # Compute bounding box
        minr, minc, maxr, maxc = region.bbox
        
        result = {
            'region_id': region_id,
            'area': int(region.area),
            'bbox': (minc, minr, maxc, maxr),  # (x1, y1, x2, y2)
            'centroid': (float(region.centroid[1]), float(region.centroid[0])),  # (x, y)
            'linearity': self.compute_linearity(region),
            'solidity': self.compute_solidity(region),
            'extent': self.compute_extent(region),
            'aspect_ratio': self.compute_aspect_ratio(region),
        }
        
        return result
    
    def analyze_all_defects_in_mask(self, mask: np.ndarray, class_id: int) -> List[Dict]:
        """
        Analyze all separate defect regions in a mask (handles multiple disconnected defects).
        
        Args:
            mask: Binary mask (H, W) that may contain multiple disconnected defects
            class_id: Defect class identifier (1-4)
            
        Returns:
            List of dictionaries, one for each defect region
        """
        # Label connected components
        labeled_mask = measure.label(mask, connectivity=2)
        regions = measure.regionprops(labeled_mask)
        
        results = []
        for idx, region in enumerate(regions):
            # Create a binary mask for this specific region
            region_mask = (labeled_mask == region.label).astype(np.uint8)
            
            # Compute bounding box
            minr, minc, maxr, maxc = region.bbox
            
            result = {
                'region_id': idx,
                'class_id': class_id,
                'area': int(region.area),
                'bbox': (minc, minr, maxc, maxr),  # (x1, y1, x2, y2)
                'centroid': (float(region.centroid[1]), float(region.centroid[0])),  # (x, y)
                'linearity': self.compute_linearity(region),
                'solidity': self.compute_solidity(region),
                'extent': self.compute_extent(region),
                'aspect_ratio': self.compute_aspect_ratio(region),
            }
            
            results.append(result)
        
        return results
    
    def classify_defect_subtype(self, metrics: Dict) -> str:
        """
        Classify defect into sub-types based on the 4 indicators.
        
        Sub-types:
        - 'linear_scratch': High linearity + high aspect ratio
        - 'blob': Low linearity + low aspect ratio + high solidity
        - 'irregular': Low solidity
        - 'elongated': High aspect ratio + medium linearity
        - 'compact': Low aspect ratio + high solidity
        
        Args:
            metrics: Dictionary with linearity, solidity, extent, aspect_ratio
            
        Returns:
            Sub-type classification string
        """
        linearity = metrics['linearity']
        solidity = metrics['solidity']
        aspect_ratio = metrics['aspect_ratio']
        
        # Define thresholds
        HIGH_LINEARITY = 0.85
        HIGH_ASPECT_RATIO = 5.0
        LOW_ASPECT_RATIO = 2.0
        HIGH_SOLIDITY = 0.9
        LOW_SOLIDITY = 0.7
        
        # Classification rules
        if linearity > HIGH_LINEARITY and aspect_ratio > HIGH_ASPECT_RATIO:
            return 'linear_scratch'
        elif solidity < LOW_SOLIDITY:
            return 'irregular'
        elif aspect_ratio > HIGH_ASPECT_RATIO and linearity > 0.6:
            return 'elongated'
        elif aspect_ratio < LOW_ASPECT_RATIO and solidity > HIGH_SOLIDITY:
            return 'compact_blob'
        else:
            return 'general'
