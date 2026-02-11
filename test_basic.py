#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple test to check if images exist and can be loaded"""

from pathlib import Path
import sys

# Setup paths
project_root = Path(__file__).parent
train_csv_path = project_root / "train.csv"
train_images_dir = project_root / "train_images"

print("Checking paths...")
print(f"Project root: {project_root}")
print(f"Train CSV: {train_csv_path} - Exists: {train_csv_path.exists()}")
print(f"Train images dir: {train_images_dir} - Exists: {train_images_dir.exists()}")

if train_images_dir.exists():
    image_files = list(train_images_dir.glob("*.jpg"))
    print(f"\nFound {len(image_files)} jpg files")
    
    if len(image_files) > 0:
        print(f"\nFirst 5 images:")
        for img in image_files[:5]:
            print(f"  - {img.name}")

import pandas as pd
df = pd.read_csv(train_csv_path)
print(f"\nTrain CSV loaded: {len(df)} rows")
print(f"Unique images in CSV: {df['ImageId'].nunique()}")

all_images = set([f.name for f in train_images_dir.glob("*.jpg")])
defect_images = set(df['ImageId'].unique())
clean_images = list(all_images - defect_images)

print(f"\nTotal images: {len(all_images)}")
print(f"Images with defects: {len(defect_images)}")
print(f"Clean images: {len(clean_images)}")

if len(clean_images) > 0:
    print(f"\nFirst 3 clean images:")
    for img in clean_images[:3]:
        print(f"  - {img}")
        
    # Try to load one image
    try:
        import cv2
        test_img_path = train_images_dir / clean_images[0]
        img = cv2.imread(str(test_img_path))
        if img is not None:
            print(f"\nSuccessfully loaded {clean_images[0]}")
            print(f"Image shape: {img.shape}")
        else:
            print(f"\nFailed to load {clean_images[0]}")
    except Exception as e:
        print(f"\nError loading image: {e}")
