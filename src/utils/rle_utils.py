"""
RLE (Run-Length Encoding) utility functions for decoding mask data.

Performance notes (v5):
- build_image_index(): pre-groups DataFrame by ImageId for O(1) lookup
  instead of repeated O(N) DataFrame filtering per image.
- rle_decode(): uses np.concatenate + np.arange vectorization to avoid
  Python for-loop over run pairs.
- get_all_masks_for_image(): accepts optional pre-built index to skip
  redundant DataFrame scans.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional


def rle_decode(mask_rle: str, shape: tuple) -> np.ndarray:
    """
    Decode RLE encoded string to binary mask.
    
    Args:
        mask_rle: RLE encoded string (space-separated pixel positions and run lengths)
        shape: (height, width) of the mask
        
    Returns:
        Binary mask as numpy array of shape (height, width)
    """
    if pd.isna(mask_rle) or mask_rle == '' or mask_rle is None:
        return np.zeros(shape, dtype=np.uint8)
    
    s = mask_rle.split()
    starts = np.asarray(s[0::2], dtype=int)
    lengths = np.asarray(s[1::2], dtype=int)
    starts -= 1  # RLE is 1-indexed
    
    # Vectorized: build all pixel indices at once instead of Python for-loop
    if len(starts) > 0:
        pixel_indices = np.concatenate([
            np.arange(s, s + l) for s, l in zip(starts, lengths)
        ])
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        img[pixel_indices] = 1
    else:
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    
    return img.reshape(shape, order='F')  # Fortran order (column-major)


def build_image_index(train_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Pre-group DataFrame by ImageId for O(1) lookup.
    
    Without this, every call to get_all_masks_for_image() does
    4 separate DataFrame boolean filters (one per class), each scanning
    the full DataFrame — O(4*N) per image, O(4*N*M) total for M images.
    
    With this, groupby runs once — O(N) total — and lookups are O(1).
    
    Args:
        train_df: Training dataframe with columns [ImageId, ClassId, EncodedPixels]
        
    Returns:
        Dictionary mapping image_id -> sub-DataFrame of that image's rows
    """
    grouped = {name: group for name, group in train_df.groupby('ImageId')}
    return grouped


def rle_encode(mask: np.ndarray) -> str:
    """
    Encode binary mask to RLE format.
    
    Args:
        mask: Binary mask as numpy array of shape (height, width)
        
    Returns:
        RLE encoded string
    """
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def decode_mask_from_csv(image_id: str, class_id: int, train_df, 
                         shape: tuple = (256, 1600),
                         image_index: Optional[Dict[str, pd.DataFrame]] = None) -> np.ndarray:
    """
    Decode mask for a specific image and class from the training CSV.
    
    Args:
        image_id: Image identifier
        class_id: Defect class (1-4)
        train_df: Training dataframe with columns [ImageId, ClassId, EncodedPixels]
        shape: (height, width) of the mask
        image_index: Optional pre-built index from build_image_index() for O(1) lookup
        
    Returns:
        Binary mask as numpy array
    """
    if image_index is not None:
        img_df = image_index.get(image_id)
        if img_df is None:
            return np.zeros(shape, dtype=np.uint8)
        mask_data = img_df[img_df['ClassId'] == class_id]
    else:
        mask_data = train_df[(train_df['ImageId'] == image_id) & (train_df['ClassId'] == class_id)]
    
    if len(mask_data) == 0:
        return np.zeros(shape, dtype=np.uint8)
    
    rle_string = mask_data.iloc[0]['EncodedPixels']
    return rle_decode(rle_string, shape)


def get_all_masks_for_image(image_id: str, train_df, shape: tuple = (256, 1600),
                            image_index: Optional[Dict[str, pd.DataFrame]] = None) -> dict:
    """
    Get all defect masks for a given image.
    
    Args:
        image_id: Image identifier
        train_df: Training dataframe
        shape: (height, width) of the mask
        image_index: Optional pre-built index from build_image_index() for O(1) lookup
        
    Returns:
        Dictionary with class_id as key and binary mask as value
    """
    # Fast path: use pre-built index to get only this image's rows
    if image_index is not None:
        img_df = image_index.get(image_id)
        if img_df is None:
            return {}
        masks = {}
        for _, row in img_df.iterrows():
            class_id = int(row['ClassId'])
            rle_string = row['EncodedPixels']
            if pd.notna(rle_string) and rle_string != '':
                mask = rle_decode(rle_string, shape)
                if mask.sum() > 0:
                    masks[class_id] = mask
        return masks
    
    # Slow fallback: scan full DataFrame (4 filters per image)
    masks = {}
    for class_id in [1, 2, 3, 4]:
        mask = decode_mask_from_csv(image_id, class_id, train_df, shape)
        if mask.sum() > 0:  # Only include non-empty masks
            masks[class_id] = mask
    return masks

