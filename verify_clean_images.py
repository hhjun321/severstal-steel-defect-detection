import pandas as pd

df = pd.read_csv('train.csv')

# 결함 없는 이미지 찾기
defect_free_images = df[df['EncodedPixels'].isna()]['ImageId'].unique()
print(f'총 결함 없는 이미지 수: {len(defect_free_images)}')
print(f'처음 10개: {list(defect_free_images[:10])}')

# 검증 파일에 사용된 이미지 확인
test_images = ['830d3ac76.jpg', '32b55fcf1.jpg', 'fa6a00996.jpg']
print(f'\n검증에 사용된 이미지:')
for img in test_images:
    is_defect_free = img in defect_free_images
    status = "결함 없음 ✓" if is_defect_free else "결함 있음 ❌"
    print(f'  {img}: {status}')

# 결함 있는 이미지도 확인
defect_images = df[df['EncodedPixels'].notna()]['ImageId'].unique()
print(f'\n총 결함 있는 이미지 수: {len(defect_images)}')
