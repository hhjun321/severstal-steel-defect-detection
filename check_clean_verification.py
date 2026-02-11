#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verify that clean_bg_verification images are truly defect-free
"""
import pandas as pd
from pathlib import Path

# Read train.csv
train_csv = Path('train.csv')
df = pd.read_csv(train_csv)

# Get all images with defects
defect_images = set(df['ImageId'].unique())
print(f"총 결함 있는 이미지 수 (train.csv): {len(defect_images)}")

# Images used in verification
test_images = ['830d3ac76.jpg', '32b55fcf1.jpg', 'fa6a00996.jpg']

print("\n검증 파일에 사용된 이미지:")
print("-" * 60)

all_clean = True
for img in test_images:
    is_in_traincsv = img in defect_images
    if is_in_traincsv:
        print(f"❌ {img}: train.csv에 있음 (결함 있음) - 문제!")
        all_clean = False
    else:
        print(f"✅ {img}: train.csv에 없음 (결함 없음) - 정확!")

print("-" * 60)
if all_clean:
    print("\n✅ 모든 검증 이미지가 결함 없는 이미지입니다!")
    print("   clean_bg_verification이 올바르게 생성되었습니다.")
else:
    print("\n❌ 일부 이미지가 결함이 있습니다!")
    print("   스크립트 재확인이 필요합니다.")
