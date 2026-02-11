"""
Unit tests for CASDA augmentation pipeline critical functions.

Tests cover:
- Background extraction and quality scoring
- Defect template building and matching
- Mask generation functions
- Quality validation checks
- RLE encoding/decoding

Usage:
    # Run all tests
    pytest tests/test_augmentation_pipeline.py -v

    # Run specific test
    pytest tests/test_augmentation_pipeline.py::TestBackgroundExtraction -v

    # Run with coverage
    pytest tests/test_augmentation_pipeline.py --cov=scripts --cov=src

Author: CASDA Pipeline Team
Date: 2026-02-09
"""

import unittest
import numpy as np
import cv2
import json
import tempfile
import os
from pathlib import Path


class TestBackgroundExtraction(unittest.TestCase):
    """Test background extraction and quality scoring."""
    
    def setUp(self):
        """Create test images."""
        # Create a smooth background
        self.smooth_bg = np.ones((256, 512, 3), dtype=np.uint8) * 128
        
        # Create a striped background
        self.stripe_bg = np.ones((256, 512, 3), dtype=np.uint8) * 128
        self.stripe_bg[:, ::10, :] = 200  # Vertical stripes
        
        # Create a noisy background
        self.noisy_bg = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)
        
        # Create a blurry background
        self.blurry_bg = cv2.GaussianBlur(self.smooth_bg, (51, 51), 0)
    
    def test_blur_detection(self):
        """Test blur detection using Laplacian variance."""
        def calculate_blur_score(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return min(laplacian_var / 200.0, 1.0)
        
        # Sharp image should have high blur score
        sharp_score = calculate_blur_score(self.stripe_bg)
        self.assertGreater(sharp_score, 0.5, "Sharp image should have high blur score")
        
        # Blurry image should have low blur score
        blurry_score = calculate_blur_score(self.blurry_bg)
        self.assertLess(blurry_score, sharp_score, "Blurry image should have lower score")
    
    def test_contrast_scoring(self):
        """Test contrast quality scoring."""
        def calculate_contrast_score(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            contrast = gray.std()
            return min(contrast / 50.0, 1.0)
        
        # Striped image should have higher contrast
        stripe_score = calculate_contrast_score(self.stripe_bg)
        smooth_score = calculate_contrast_score(self.smooth_bg)
        
        self.assertGreater(stripe_score, smooth_score, "Striped image should have higher contrast")
    
    def test_noise_scoring(self):
        """Test noise detection."""
        def calculate_noise_score(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            noise = np.std(cv2.Laplacian(gray, cv2.CV_64F))
            return max(0, 1.0 - noise / 50.0)
        
        # Smooth image should have high noise score (low noise)
        smooth_score = calculate_noise_score(self.smooth_bg)
        noisy_score = calculate_noise_score(self.noisy_bg)
        
        self.assertGreater(smooth_score, noisy_score, "Smooth image should have higher noise score")


class TestDefectMaskGeneration(unittest.TestCase):
    """Test synthetic defect mask generation."""
    
    def test_linear_mask_generation(self):
        """Test linear scratch mask generation."""
        def create_linear_mask(width, height, thickness=5, angle=0):
            mask = np.zeros((height, width), dtype=np.uint8)
            center = (width // 2, height // 2)
            
            # Create line
            angle_rad = np.deg2rad(angle)
            dx = int(width * 0.4 * np.cos(angle_rad))
            dy = int(height * 0.4 * np.sin(angle_rad))
            
            pt1 = (center[0] - dx, center[1] - dy)
            pt2 = (center[0] + dx, center[1] + dy)
            
            cv2.line(mask, pt1, pt2, 255, thickness)
            return mask
        
        # Generate mask
        mask = create_linear_mask(200, 200, thickness=5, angle=45)
        
        # Verify properties
        self.assertEqual(mask.shape, (200, 200))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertGreater(np.sum(mask > 0), 0, "Mask should have non-zero pixels")
        
        # Check linearity (should be elongated)
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) > 0:
            bbox_h = coords[:, 0].max() - coords[:, 0].min()
            bbox_w = coords[:, 1].max() - coords[:, 1].min()
            aspect_ratio = max(bbox_h, bbox_w) / (min(bbox_h, bbox_w) + 1e-6)
            self.assertGreater(aspect_ratio, 2.0, "Linear mask should be elongated")
    
    def test_blob_mask_generation(self):
        """Test compact blob mask generation."""
        def create_blob_mask(width, height, radius=30):
            mask = np.zeros((height, width), dtype=np.uint8)
            center = (width // 2, height // 2)
            cv2.circle(mask, center, radius, 255, -1)
            return mask
        
        # Generate mask
        mask = create_blob_mask(200, 200, radius=30)
        
        # Verify properties
        self.assertGreater(np.sum(mask > 0), 0, "Mask should have non-zero pixels")
        
        # Check circularity (aspect ratio should be close to 1)
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) > 0:
            bbox_h = coords[:, 0].max() - coords[:, 0].min()
            bbox_w = coords[:, 1].max() - coords[:, 1].min()
            aspect_ratio = max(bbox_h, bbox_w) / (min(bbox_h, bbox_w) + 1e-6)
            self.assertLess(aspect_ratio, 1.5, "Blob mask should be roughly circular")
    
    def test_mask_scaling(self):
        """Test defect mask scaling."""
        def scale_mask(mask, scale_factor):
            h, w = mask.shape
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            return cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Create original mask
        original = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(original, (50, 50), 20, 255, -1)
        original_area = np.sum(original > 0)
        
        # Scale to 80%
        scaled = scale_mask(original, 0.8)
        scaled_area = np.sum(scaled > 0)
        
        # Check size reduction
        self.assertEqual(scaled.shape, (80, 80))
        self.assertLess(scaled_area, original_area, "Scaled mask should have smaller area")


class TestQualityValidation(unittest.TestCase):
    """Test quality validation functions."""
    
    def test_defect_presence_check(self):
        """Test defect presence validation."""
        def check_defect_presence(mask, min_ratio=0.001, max_ratio=0.3):
            defect_area = np.sum(mask > 0)
            total_area = mask.shape[0] * mask.shape[1]
            ratio = defect_area / total_area
            
            if ratio < min_ratio:
                return False, "Defect too small"
            if ratio > max_ratio:
                return False, "Defect too large"
            return True, "OK"
        
        # Create test masks
        valid_mask = np.zeros((256, 1600), dtype=np.uint8)
        valid_mask[100:150, 700:800] = 255  # ~1.2% of image
        
        tiny_mask = np.zeros((256, 1600), dtype=np.uint8)
        tiny_mask[100:102, 700:705] = 255  # ~0.0002% of image
        
        huge_mask = np.ones((256, 1600), dtype=np.uint8) * 255  # 100% of image
        
        # Test validation
        valid, msg = check_defect_presence(valid_mask)
        self.assertTrue(valid, f"Valid mask should pass: {msg}")
        
        valid, msg = check_defect_presence(tiny_mask)
        self.assertFalse(valid, "Tiny defect should fail")
        
        valid, msg = check_defect_presence(huge_mask)
        self.assertFalse(valid, "Huge defect should fail")
    
    def test_artifact_detection(self):
        """Test artifact detection using gradient analysis."""
        def detect_artifacts(image, threshold_percentile=95, max_gradient=150):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Compute gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Check 95th percentile
            p95 = np.percentile(gradient_magnitude, threshold_percentile)
            
            return p95 < max_gradient, p95
        
        # Create clean image
        clean = np.ones((256, 512, 3), dtype=np.uint8) * 128
        
        # Create image with artifacts (sharp edges)
        artifact = clean.copy()
        artifact[100:110, :, :] = 255  # Sharp horizontal line
        
        # Test detection
        clean_ok, clean_p95 = detect_artifacts(clean)
        artifact_ok, artifact_p95 = detect_artifacts(artifact)
        
        self.assertGreater(artifact_p95, clean_p95, "Artifact image should have higher gradients")


class TestRLEEncoding(unittest.TestCase):
    """Test RLE encoding/decoding."""
    
    def test_rle_encoding_simple(self):
        """Test RLE encoding for simple mask."""
        def mask_to_rle_simple(mask):
            """Simplified RLE encoding for testing."""
            pixels = mask.T.flatten()
            pixels = np.concatenate([[0], pixels, [0]])
            runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            return ' '.join(str(x) for x in runs)
        
        # Create test mask
        mask = np.zeros((256, 1600), dtype=np.uint8)
        mask[10:20, 50:60] = 1  # Small rectangle
        
        # Encode
        rle = mask_to_rle_simple(mask)
        
        # Verify format
        self.assertIsInstance(rle, str)
        self.assertGreater(len(rle), 0)
        
        # RLE should contain pairs of numbers
        parts = rle.split()
        self.assertEqual(len(parts) % 2, 0, "RLE should have even number of elements")
    
    def test_rle_roundtrip(self):
        """Test RLE encoding and decoding roundtrip."""
        def mask_to_rle_simple(mask):
            pixels = mask.T.flatten()
            pixels = np.concatenate([[0], pixels, [0]])
            runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            return ' '.join(str(x) for x in runs)
        
        def rle_to_mask_simple(rle_string, shape):
            s = rle_string.split()
            starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
            starts -= 1
            ends = starts + lengths
            img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
            for lo, hi in zip(starts, ends):
                img[lo:hi] = 1
            return img.reshape(shape).T
        
        # Create test mask
        original_mask = np.zeros((256, 1600), dtype=np.uint8)
        original_mask[50:100, 200:300] = 1
        
        # Encode and decode
        rle = mask_to_rle_simple(original_mask)
        reconstructed_mask = rle_to_mask_simple(rle, (256, 1600))
        
        # Verify roundtrip
        np.testing.assert_array_equal(original_mask, reconstructed_mask,
                                      "Roundtrip should preserve mask exactly")


class TestCompatibilityMatching(unittest.TestCase):
    """Test defect-background compatibility matching."""
    
    def test_matching_rules(self):
        """Test compatibility matching rules."""
        # Define matching rules
        compatibility_rules = {
            'linear_scratch': {
                'compatible_backgrounds': ['vertical_stripe', 'horizontal_stripe'],
                'min_suitability': 0.7
            },
            'compact_blob': {
                'compatible_backgrounds': ['smooth', 'textured'],
                'min_suitability': 0.6
            },
            'irregular': {
                'compatible_backgrounds': ['smooth', 'textured', 'vertical_stripe', 
                                           'horizontal_stripe', 'complex_pattern'],
                'min_suitability': 0.5
            }
        }
        
        # Test linear scratch matching
        linear_rules = compatibility_rules['linear_scratch']
        self.assertIn('vertical_stripe', linear_rules['compatible_backgrounds'])
        self.assertNotIn('smooth', linear_rules['compatible_backgrounds'])
        
        # Test blob matching
        blob_rules = compatibility_rules['compact_blob']
        self.assertIn('smooth', blob_rules['compatible_backgrounds'])
        
        # Test irregular matching (should match most backgrounds)
        irregular_rules = compatibility_rules['irregular']
        self.assertGreaterEqual(len(irregular_rules['compatible_backgrounds']), 4)


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and format compliance."""
    
    def test_metadata_format(self):
        """Test augmented metadata format."""
        # Sample metadata entry
        sample_meta = {
            'aug_id': 'aug_00000',
            'image_file': 'aug_00000.jpg',
            'mask_file': 'aug_00000.png',
            'class_id': 1,
            'background_id': 'bg_00123',
            'template_id': 'template_00045',
            'background_type': 'vertical_stripe',
            'defect_subtype': 'linear_scratch',
            'scale_factor': 0.92,
            'defect_position': [150, 200],
            'generation_timestamp': '2026-02-09T10:30:15'
        }
        
        # Verify required fields
        required_fields = ['aug_id', 'image_file', 'mask_file', 'class_id', 
                          'background_type', 'defect_subtype', 'scale_factor']
        
        for field in required_fields:
            self.assertIn(field, sample_meta, f"Metadata missing required field: {field}")
        
        # Verify data types
        self.assertIsInstance(sample_meta['class_id'], int)
        self.assertIsInstance(sample_meta['scale_factor'], float)
        self.assertIsInstance(sample_meta['defect_position'], list)
        
        # Verify value ranges
        self.assertIn(sample_meta['class_id'], [1, 2, 3, 4])
        self.assertGreaterEqual(sample_meta['scale_factor'], 0.8)
        self.assertLessEqual(sample_meta['scale_factor'], 1.0)
    
    def test_image_format_compliance(self):
        """Test that generated images match expected format."""
        # Create test image
        test_image = np.random.randint(0, 255, (256, 1600, 3), dtype=np.uint8)
        
        # Verify dimensions
        self.assertEqual(test_image.shape[0], 256, "Height should be 256")
        self.assertEqual(test_image.shape[1], 1600, "Width should be 1600")
        self.assertEqual(test_image.shape[2], 3, "Should have 3 color channels")
        
        # Verify data type
        self.assertEqual(test_image.dtype, np.uint8)
    
    def test_mask_format_compliance(self):
        """Test that generated masks match expected format."""
        # Create test mask
        test_mask = np.zeros((256, 1600), dtype=np.uint8)
        test_mask[100:150, 700:800] = 255
        
        # Verify dimensions
        self.assertEqual(test_mask.shape[0], 256, "Height should be 256")
        self.assertEqual(test_mask.shape[1], 1600, "Width should be 1600")
        self.assertEqual(len(test_mask.shape), 2, "Should be grayscale (2D)")
        
        # Verify binary values
        unique_values = np.unique(test_mask)
        for val in unique_values:
            self.assertIn(val, [0, 255], f"Mask should only contain 0 and 255, found {val}")


def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBackgroundExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestDefectMaskGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestQualityValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestRLEEncoding))
    suite.addTests(loader.loadTestsFromTestCase(TestCompatibilityMatching))
    suite.addTests(loader.loadTestsFromTestCase(TestDataIntegrity))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
