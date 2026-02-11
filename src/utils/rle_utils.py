"""
RLE (Run-Length Encoding) utility functions for decoding mask data.
"""
import numpy as np
import pandas as pd


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
    ends = starts + lengths
    
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(shape, order='F')  # Fortran order (column-major)


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


def decode_mask_from_csv(image_id: str, class_id: int, train_df, shape: tuple = (256, 1600)) -> np.ndarray:
    """
    Decode mask for a specific image and class from the training CSV.
    
    Args:
        image_id: Image identifier
        class_id: Defect class (1-4)
        train_df: Training dataframe with columns [ImageId, ClassId, EncodedPixels]
        shape: (height, width) of the mask
        
    Returns:
        Binary mask as numpy array
    """
    mask_data = train_df[(train_df['ImageId'] == image_id) & (train_df['ClassId'] == class_id)]
    
    if len(mask_data) == 0:
        return np.zeros(shape, dtype=np.uint8)
    
    rle_string = mask_data.iloc[0]['EncodedPixels']
    return rle_decode(rle_string, shape)


def get_all_masks_for_image(image_id: str, train_df, shape: tuple = (256, 1600)) -> dict:
    """
    Get all defect masks for a given image.
    
    Args:
        image_id: Image identifier
        train_df: Training dataframe
        shape: (height, width) of the mask
        
    Returns:
        Dictionary with class_id as key and binary mask as value
    """
    masks = {}
    for class_id in [1, 2, 3, 4]:
        mask = decode_mask_from_csv(image_id, class_id, train_df, shape)
        if mask.sum() > 0:  # Only include non-empty masks
            masks[class_id] = mask
    return masks

