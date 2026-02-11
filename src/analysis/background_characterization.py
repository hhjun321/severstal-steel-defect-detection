"""
Background Characterization Module

This module analyzes background textures using grid-based classification.
According to PROJECT(roi).md, it identifies background types:
- 'smooth': Low variance, flat surface
- 'textured': High variance, patterned surface
- 'vertical_stripe': Strong vertical edge patterns
- 'horizontal_stripe': Strong horizontal edge patterns
- 'complex_pattern': Multi-directional edge patterns

This helps determine suitable backgrounds for synthetic defect placement.
"""
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from enum import Enum


class BackgroundType(Enum):
    """Background texture types"""
    SMOOTH = 'smooth'
    TEXTURED = 'textured'
    VERTICAL_STRIPE = 'vertical_stripe'
    HORIZONTAL_STRIPE = 'horizontal_stripe'
    COMPLEX_PATTERN = 'complex_pattern'


class BackgroundAnalyzer:
    """
    Analyzes background textures using grid-based approach.
    """
    
    def __init__(self, grid_size: int = 64, variance_threshold: float = 100.0, 
                 edge_threshold: float = 0.3):
        """
        Initialize background analyzer.
        
        Args:
            grid_size: Size of grid patches (64x64 or 128x128)
            variance_threshold: Threshold to distinguish smooth vs textured
            edge_threshold: Threshold for edge direction analysis
        """
        self.grid_size = grid_size
        self.variance_threshold = variance_threshold
        self.edge_threshold = edge_threshold
    
    def compute_variance(self, patch: np.ndarray) -> float:
        """
        Compute variance of a grayscale patch.
        
        Args:
            patch: Grayscale image patch
            
        Returns:
            Variance value
        """
        return float(np.var(patch))
    
    def compute_edge_directions(self, patch: np.ndarray) -> Dict[str, float]:
        """
        Compute edge strengths in different directions using Sobel filters.
        
        Args:
            patch: Grayscale image patch
            
        Returns:
            Dictionary with directional edge strengths
        """
        # Compute Sobel gradients
        sobel_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude and direction
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Compute directional strengths
        vertical_strength = np.mean(np.abs(sobel_x))  # Vertical edges (horizontal gradients)
        horizontal_strength = np.mean(np.abs(sobel_y))  # Horizontal edges (vertical gradients)
        total_strength = np.mean(magnitude)
        
        return {
            'vertical': float(vertical_strength),
            'horizontal': float(horizontal_strength),
            'total': float(total_strength),
            'magnitude_std': float(np.std(magnitude))
        }
    
    def compute_frequency_spectrum(self, patch: np.ndarray) -> Dict[str, float]:
        """
        Compute FFT-based frequency characteristics.
        
        Args:
            patch: Grayscale image patch
            
        Returns:
            Dictionary with frequency domain features
        """
        # Compute 2D FFT
        f_transform = np.fft.fft2(patch)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Get center coordinates
        h, w = patch.shape
        center_y, center_x = h // 2, w // 2
        
        # Analyze frequency distribution
        # High frequency (far from center) indicates texture
        radius = min(h, w) // 4
        center_mask = np.zeros((h, w), dtype=bool)
        y, x = np.ogrid[:h, :w]
        center_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        center_energy = np.sum(magnitude_spectrum[center_mask])
        total_energy = np.sum(magnitude_spectrum)
        high_freq_ratio = 1.0 - (center_energy / (total_energy + 1e-6))
        
        return {
            'high_freq_ratio': float(high_freq_ratio),
            'total_energy': float(total_energy),
            'center_energy': float(center_energy)
        }
    
    def classify_patch(self, patch: np.ndarray) -> Tuple[BackgroundType, float]:
        """
        Classify a single patch into a background type.
        
        Args:
            patch: Grayscale image patch
            
        Returns:
            Tuple of (background_type, stability_score)
            stability_score indicates how stable/uniform the background is (0-1)
        """
        # Step 1: Variance analysis
        variance = self.compute_variance(patch)
        
        # Step 2: Edge direction analysis
        edge_info = self.compute_edge_directions(patch)
        
        # Step 3: Frequency analysis
        freq_info = self.compute_frequency_spectrum(patch)
        
        # Classification logic
        if variance < self.variance_threshold:
            # Low variance -> Smooth
            bg_type = BackgroundType.SMOOTH
            stability = 1.0 - (variance / self.variance_threshold)
        else:
            # High variance -> Check edge patterns
            v_strength = edge_info['vertical']
            h_strength = edge_info['horizontal']
            total_strength = edge_info['total']
            
            if total_strength < 1.0:  # Very weak edges
                bg_type = BackgroundType.TEXTURED
                stability = 0.5
            else:
                # Normalize strengths
                v_ratio = v_strength / (total_strength + 1e-6)
                h_ratio = h_strength / (total_strength + 1e-6)
                
                # Check for dominant direction
                if v_ratio > self.edge_threshold and v_ratio > h_ratio * 1.5:
                    bg_type = BackgroundType.VERTICAL_STRIPE
                    stability = v_ratio
                elif h_ratio > self.edge_threshold and h_ratio > v_ratio * 1.5:
                    bg_type = BackgroundType.HORIZONTAL_STRIPE
                    stability = h_ratio
                elif freq_info['high_freq_ratio'] > 0.3:
                    bg_type = BackgroundType.COMPLEX_PATTERN
                    stability = 1.0 - freq_info['high_freq_ratio']
                else:
                    bg_type = BackgroundType.TEXTURED
                    stability = 0.5
        
        return bg_type, float(np.clip(stability, 0.0, 1.0))
    
    def analyze_image(self, image: np.ndarray) -> Dict:
        """
        Analyze entire image using grid-based approach.
        
        Args:
            image: Input image (H, W, C) or (H, W)
            
        Returns:
            Dictionary containing:
            - 'background_map': 2D array of background types
            - 'stability_map': 2D array of stability scores
            - 'grid_info': List of grid cell information
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        grid_h = h // self.grid_size
        grid_w = w // self.grid_size
        
        # Initialize maps
        background_map = np.empty((grid_h, grid_w), dtype=object)
        stability_map = np.zeros((grid_h, grid_w), dtype=np.float32)
        grid_info = []
        
        # Analyze each grid cell
        for i in range(grid_h):
            for j in range(grid_w):
                # Extract patch
                y1 = i * self.grid_size
                x1 = j * self.grid_size
                y2 = min(y1 + self.grid_size, h)
                x2 = min(x1 + self.grid_size, w)
                patch = gray[y1:y2, x1:x2]
                
                # Classify patch
                bg_type, stability = self.classify_patch(patch)
                
                # Store results
                background_map[i, j] = bg_type.value
                stability_map[i, j] = stability
                
                grid_info.append({
                    'grid_id': (i, j),
                    'bbox': (x1, y1, x2, y2),
                    'background_type': bg_type.value,
                    'stability_score': float(stability)
                })
        
        return {
            'background_map': background_map,
            'stability_map': stability_map,
            'grid_info': grid_info,
            'grid_size': self.grid_size,
            'grid_shape': (grid_h, grid_w)
        }
    
    def get_background_at_location(self, analysis_result: Dict, x: int, y: int) -> Optional[Dict]:
        """
        Get background information at a specific pixel location.
        
        Args:
            analysis_result: Result from analyze_image()
            x, y: Pixel coordinates
            
        Returns:
            Dictionary with background type and stability at that location
        """
        grid_size = analysis_result['grid_size']
        grid_j = x // grid_size
        grid_i = y // grid_size
        
        grid_h, grid_w = analysis_result['grid_shape']
        
        if grid_i >= grid_h or grid_j >= grid_w or grid_i < 0 or grid_j < 0:
            return None
        
        bg_type = analysis_result['background_map'][grid_i, grid_j]
        stability = analysis_result['stability_map'][grid_i, grid_j]
        
        return {
            'background_type': bg_type,
            'stability_score': float(stability),
            'grid_id': (grid_i, grid_j)
        }
    
    def check_continuity(self, analysis_result: Dict, bbox: Tuple[int, int, int, int]) -> float:
        """
        Check background continuity within a bounding box (for ROI suitability).
        
        Args:
            analysis_result: Result from analyze_image()
            bbox: (x1, y1, x2, y2) bounding box
            
        Returns:
            Continuity score (0-1): 1.0 = highly continuous, 0.0 = discontinuous
        """
        x1, y1, x2, y2 = bbox
        grid_size = analysis_result['grid_size']
        
        # Get grid cells that overlap with bbox
        grid_i1 = y1 // grid_size
        grid_j1 = x1 // grid_size
        grid_i2 = y2 // grid_size
        grid_j2 = x2 // grid_size
        
        # Extract background types in this region
        bg_map = analysis_result['background_map']
        stability_map = analysis_result['stability_map']
        
        grid_h, grid_w = analysis_result['grid_shape']
        grid_i2 = min(grid_i2, grid_h - 1)
        grid_j2 = min(grid_j2, grid_w - 1)
        
        region_bg_types = bg_map[grid_i1:grid_i2+1, grid_j1:grid_j2+1]
        region_stability = stability_map[grid_i1:grid_i2+1, grid_j1:grid_j2+1]
        
        if region_bg_types.size == 0:
            return 0.0
        
        # Check uniformity: most common background type
        unique, counts = np.unique(region_bg_types, return_counts=True)
        max_count = np.max(counts)
        uniformity = max_count / region_bg_types.size
        
        # Average stability in region
        avg_stability = np.mean(region_stability)
        
        # Continuity = weighted combination
        continuity = 0.6 * uniformity + 0.4 * avg_stability
        
        return float(continuity)
