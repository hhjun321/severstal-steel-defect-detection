import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
print("데이터를 로드하는 중...")
train_df = pd.read_csv('train.csv')
roi_df = pd.read_csv('data/processed/roi_patches/roi_metadata.csv')

# 출력 디렉토리 생성
output_dir = Path('visualizations')
output_dir.mkdir(exist_ok=True)

# 색상 팔레트
colors = sns.color_palette("husl", 8)

# ============================================
# 1. 클래스별 데이터 분포 분석
# ============================================
print("1. 클래스별 데이터 분포 분석 중...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1-1: train.csv의 클래스별 결함 개수
class_counts = train_df['ClassId'].value_counts().sort_index()
axes[0, 0].bar(class_counts.index, class_counts.values, color=colors[:4])
axes[0, 0].set_xlabel('결함 클래스', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('결함 개수', fontsize=12, fontweight='bold')
axes[0, 0].set_title('원본 데이터: 클래스별 결함 개수', fontsize=14, fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(class_counts.values):
    axes[0, 0].text(class_counts.index[i], v + 100, str(v), ha='center', fontweight='bold')

# 1-2: train.csv의 클래스별 비율 (파이 차트)
axes[0, 1].pie(class_counts.values, labels=[f'Class {i}' for i in class_counts.index], 
               autopct='%1.1f%%', colors=colors[:4], startangle=90)
axes[0, 1].set_title('원본 데이터: 클래스별 비율', fontsize=14, fontweight='bold')

# 1-3: roi_metadata.csv의 클래스별 ROI 패치 개수
roi_class_counts = roi_df['class_id'].value_counts().sort_index()
axes[1, 0].bar(roi_class_counts.index, roi_class_counts.values, color=colors[:4])
axes[1, 0].set_xlabel('결함 클래스', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('ROI 패치 개수', fontsize=12, fontweight='bold')
axes[1, 0].set_title('ROI 데이터: 클래스별 패치 개수', fontsize=14, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(roi_class_counts.values):
    axes[1, 0].text(roi_class_counts.index[i], v + 5, str(v), ha='center', fontweight='bold')

# 1-4: 이미지당 평균 결함 개수
images_per_class = train_df.groupby('ClassId')['ImageId'].nunique()
defects_per_image = class_counts / images_per_class
axes[1, 1].bar(defects_per_image.index, defects_per_image.values, color=colors[:4])
axes[1, 1].set_xlabel('결함 클래스', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('이미지당 평균 결함 개수', fontsize=12, fontweight='bold')
axes[1, 1].set_title('이미지당 평균 결함 개수', fontsize=14, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(defects_per_image.values):
    axes[1, 1].text(defects_per_image.index[i], v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '01_class_distribution.png', dpi=300, bbox_inches='tight')
print(f"  → 저장: {output_dir / '01_class_distribution.png'}")
plt.close()

# ============================================
# 2. ROI 메타데이터 분석
# ============================================
print("2. ROI 메타데이터 분석 중...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 2-1: Area 분포
axes[0, 0].hist(roi_df['area'], bins=50, color=colors[0], alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('면적', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('빈도', fontsize=12, fontweight='bold')
axes[0, 0].set_title('결함 면적 분포', fontsize=14, fontweight='bold')
axes[0, 0].axvline(roi_df['area'].median(), color='red', linestyle='--', label=f'중앙값: {roi_df["area"].median():.1f}')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2-2: Linearity 분포
axes[0, 1].hist(roi_df['linearity'], bins=50, color=colors[1], alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('선형성', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('빈도', fontsize=12, fontweight='bold')
axes[0, 1].set_title('결함 선형성 분포', fontsize=14, fontweight='bold')
axes[0, 1].axvline(roi_df['linearity'].median(), color='red', linestyle='--', label=f'중앙값: {roi_df["linearity"].median():.2f}')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 2-3: Solidity 분포
axes[0, 2].hist(roi_df['solidity'], bins=50, color=colors[2], alpha=0.7, edgecolor='black')
axes[0, 2].set_xlabel('조밀도', fontsize=12, fontweight='bold')
axes[0, 2].set_ylabel('빈도', fontsize=12, fontweight='bold')
axes[0, 2].set_title('결함 조밀도 분포', fontsize=14, fontweight='bold')
axes[0, 2].axvline(roi_df['solidity'].median(), color='red', linestyle='--', label=f'중앙값: {roi_df["solidity"].median():.2f}')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# 2-4: Aspect Ratio 분포
axes[1, 0].hist(roi_df['aspect_ratio'], bins=50, color=colors[3], alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('종횡비', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('빈도', fontsize=12, fontweight='bold')
axes[1, 0].set_title('결함 종횡비 분포', fontsize=14, fontweight='bold')
axes[1, 0].axvline(roi_df['aspect_ratio'].median(), color='red', linestyle='--', label=f'중앙값: {roi_df["aspect_ratio"].median():.2f}')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 2-5: Extent 분포
axes[1, 1].hist(roi_df['extent'], bins=50, color=colors[4], alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('범위', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('빈도', fontsize=12, fontweight='bold')
axes[1, 1].set_title('결함 범위 분포', fontsize=14, fontweight='bold')
axes[1, 1].axvline(roi_df['extent'].median(), color='red', linestyle='--', label=f'중앙값: {roi_df["extent"].median():.2f}')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# 2-6: 클래스별 평균 면적 비교
class_area_mean = roi_df.groupby('class_id')['area'].mean()
axes[1, 2].bar(class_area_mean.index, class_area_mean.values, color=colors[:4])
axes[1, 2].set_xlabel('결함 클래스', fontsize=12, fontweight='bold')
axes[1, 2].set_ylabel('평균 면적', fontsize=12, fontweight='bold')
axes[1, 2].set_title('클래스별 평균 결함 면적', fontsize=14, fontweight='bold')
axes[1, 2].grid(axis='y', alpha=0.3)
for i, v in enumerate(class_area_mean.values):
    axes[1, 2].text(class_area_mean.index[i], v + 100, f'{v:.0f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '02_roi_metadata.png', dpi=300, bbox_inches='tight')
print(f"  → 저장: {output_dir / '02_roi_metadata.png'}")
plt.close()

# ============================================
# 3. 배경 타입별 분석
# ============================================
print("3. 배경 타입별 분석 중...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 3-1: 배경 타입별 개수
bg_counts = roi_df['background_type'].value_counts()
axes[0, 0].bar(range(len(bg_counts)), bg_counts.values, color=colors[:len(bg_counts)])
axes[0, 0].set_xticks(range(len(bg_counts)))
axes[0, 0].set_xticklabels(bg_counts.index, rotation=45, ha='right')
axes[0, 0].set_xlabel('배경 타입', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('개수', fontsize=12, fontweight='bold')
axes[0, 0].set_title('배경 타입별 ROI 개수', fontsize=14, fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(bg_counts.values):
    axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# 3-2: 배경 타입별 비율 (파이 차트)
axes[0, 1].pie(bg_counts.values, labels=bg_counts.index, autopct='%1.1f%%', 
               colors=colors[:len(bg_counts)], startangle=90)
axes[0, 1].set_title('배경 타입별 비율', fontsize=14, fontweight='bold')

# 3-3: 클래스별 배경 타입 분포 (히트맵)
bg_class_pivot = pd.crosstab(roi_df['background_type'], roi_df['class_id'])
sns.heatmap(bg_class_pivot, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 0], cbar_kws={'label': '개수'})
axes[1, 0].set_xlabel('결함 클래스', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('배경 타입', fontsize=12, fontweight='bold')
axes[1, 0].set_title('클래스별 배경 타입 분포', fontsize=14, fontweight='bold')

# 3-4: 배경 타입별 평균 적합성 점수
bg_suitability = roi_df.groupby('background_type')['suitability_score'].mean().sort_values(ascending=False)
axes[1, 1].barh(range(len(bg_suitability)), bg_suitability.values, color=colors[:len(bg_suitability)])
axes[1, 1].set_yticks(range(len(bg_suitability)))
axes[1, 1].set_yticklabels(bg_suitability.index)
axes[1, 1].set_xlabel('평균 적합성 점수', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('배경 타입', fontsize=12, fontweight='bold')
axes[1, 1].set_title('배경 타입별 평균 적합성 점수', fontsize=14, fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)
for i, v in enumerate(bg_suitability.values):
    axes[1, 1].text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '03_background_type.png', dpi=300, bbox_inches='tight')
print(f"  → 저장: {output_dir / '03_background_type.png'}")
plt.close()

# ============================================
# 4. 결함 서브타입 분석
# ============================================
print("4. 결함 서브타입 분석 중...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 4-1: 결함 서브타입별 개수
subtype_counts = roi_df['defect_subtype'].value_counts()
axes[0, 0].bar(range(len(subtype_counts)), subtype_counts.values, color=colors[:len(subtype_counts)])
axes[0, 0].set_xticks(range(len(subtype_counts)))
axes[0, 0].set_xticklabels(subtype_counts.index, rotation=45, ha='right')
axes[0, 0].set_xlabel('결함 서브타입', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('개수', fontsize=12, fontweight='bold')
axes[0, 0].set_title('결함 서브타입별 ROI 개수', fontsize=14, fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(subtype_counts.values):
    axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# 4-2: 결함 서브타입별 비율 (파이 차트)
axes[0, 1].pie(subtype_counts.values, labels=subtype_counts.index, autopct='%1.1f%%', 
               colors=colors[:len(subtype_counts)], startangle=90)
axes[0, 1].set_title('결함 서브타입별 비율', fontsize=14, fontweight='bold')

# 4-3: 클래스별 결함 서브타입 분포 (스택 바 차트)
subtype_class_pivot = pd.crosstab(roi_df['class_id'], roi_df['defect_subtype'])
subtype_class_pivot.plot(kind='bar', stacked=True, ax=axes[1, 0], color=colors[:len(subtype_counts)])
axes[1, 0].set_xlabel('결함 클래스', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('개수', fontsize=12, fontweight='bold')
axes[1, 0].set_title('클래스별 결함 서브타입 분포', fontsize=14, fontweight='bold')
axes[1, 0].legend(title='결함 서브타입', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1, 0].grid(axis='y', alpha=0.3)
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)

# 4-4: 결함 서브타입별 평균 선형성
subtype_linearity = roi_df.groupby('defect_subtype')['linearity'].mean().sort_values(ascending=False)
axes[1, 1].barh(range(len(subtype_linearity)), subtype_linearity.values, color=colors[:len(subtype_linearity)])
axes[1, 1].set_yticks(range(len(subtype_linearity)))
axes[1, 1].set_yticklabels(subtype_linearity.index)
axes[1, 1].set_xlabel('평균 선형성', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('결함 서브타입', fontsize=12, fontweight='bold')
axes[1, 1].set_title('결함 서브타입별 평균 선형성', fontsize=14, fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)
for i, v in enumerate(subtype_linearity.values):
    axes[1, 1].text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '04_defect_subtype.png', dpi=300, bbox_inches='tight')
print(f"  → 저장: {output_dir / '04_defect_subtype.png'}")
plt.close()

# ============================================
# 5. 적합성 점수 분석
# ============================================
print("5. 적합성 점수 분석 중...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 5-1: Suitability Score 분포
axes[0, 0].hist(roi_df['suitability_score'], bins=50, color=colors[0], alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('적합성 점수', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('빈도', fontsize=12, fontweight='bold')
axes[0, 0].set_title('적합성 점수 분포', fontsize=14, fontweight='bold')
axes[0, 0].axvline(roi_df['suitability_score'].median(), color='red', linestyle='--', 
                   label=f'중앙값: {roi_df["suitability_score"].median():.3f}')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 5-2: Matching Score 분포
axes[0, 1].hist(roi_df['matching_score'], bins=50, color=colors[1], alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('매칭 점수', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('빈도', fontsize=12, fontweight='bold')
axes[0, 1].set_title('매칭 점수 분포', fontsize=14, fontweight='bold')
axes[0, 1].axvline(roi_df['matching_score'].median(), color='red', linestyle='--', 
                   label=f'중앙값: {roi_df["matching_score"].median():.3f}')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 5-3: Continuity Score 분포
axes[0, 2].hist(roi_df['continuity_score'], bins=50, color=colors[2], alpha=0.7, edgecolor='black')
axes[0, 2].set_xlabel('연속성 점수', fontsize=12, fontweight='bold')
axes[0, 2].set_ylabel('빈도', fontsize=12, fontweight='bold')
axes[0, 2].set_title('연속성 점수 분포', fontsize=14, fontweight='bold')
axes[0, 2].axvline(roi_df['continuity_score'].median(), color='red', linestyle='--', 
                   label=f'중앙값: {roi_df["continuity_score"].median():.3f}')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# 5-4: Stability Score 분포
axes[1, 0].hist(roi_df['stability_score'], bins=50, color=colors[3], alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('안정성 점수', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('빈도', fontsize=12, fontweight='bold')
axes[1, 0].set_title('안정성 점수 분포', fontsize=14, fontweight='bold')
axes[1, 0].axvline(roi_df['stability_score'].median(), color='red', linestyle='--', 
                   label=f'중앙값: {roi_df["stability_score"].median():.3f}')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 5-5: 클래스별 평균 적합성 점수
class_suitability = roi_df.groupby('class_id')['suitability_score'].mean()
axes[1, 1].bar(class_suitability.index, class_suitability.values, color=colors[:4])
axes[1, 1].set_xlabel('결함 클래스', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('평균 적합성 점수', fontsize=12, fontweight='bold')
axes[1, 1].set_title('클래스별 평균 적합성 점수', fontsize=14, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(class_suitability.values):
    axes[1, 1].text(class_suitability.index[i], v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

# 5-6: Recommendation 분포
recommendation_counts = roi_df['recommendation'].value_counts()
axes[1, 2].bar(range(len(recommendation_counts)), recommendation_counts.values, color=colors[:len(recommendation_counts)])
axes[1, 2].set_xticks(range(len(recommendation_counts)))
axes[1, 2].set_xticklabels(recommendation_counts.index, rotation=45, ha='right')
axes[1, 2].set_xlabel('추천 등급', fontsize=12, fontweight='bold')
axes[1, 2].set_ylabel('개수', fontsize=12, fontweight='bold')
axes[1, 2].set_title('추천 등급별 ROI 개수', fontsize=14, fontweight='bold')
axes[1, 2].grid(axis='y', alpha=0.3)
for i, v in enumerate(recommendation_counts.values):
    axes[1, 2].text(i, v + 5, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '05_suitability_scores.png', dpi=300, bbox_inches='tight')
print(f"  → 저장: {output_dir / '05_suitability_scores.png'}")
plt.close()

# ============================================
# 통계 요약 저장
# ============================================
print("\n통계 요약 저장 중...")

with open(output_dir / 'statistics_summary.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("강철 결함 데이터 통계 요약\n")
    f.write("=" * 80 + "\n\n")
    
    # 1. 기본 통계
    f.write("1. 기본 데이터 통계\n")
    f.write("-" * 80 + "\n")
    f.write(f"총 이미지 수: {train_df['ImageId'].nunique()}\n")
    f.write(f"총 결함 수 (train.csv): {len(train_df)}\n")
    f.write(f"총 ROI 패치 수: {len(roi_df)}\n")
    f.write(f"이미지당 평균 결함 수: {len(train_df) / train_df['ImageId'].nunique():.2f}\n\n")
    
    # 2. 클래스별 통계
    f.write("2. 클래스별 통계\n")
    f.write("-" * 80 + "\n")
    for cls in sorted(train_df['ClassId'].unique()):
        cls_data = train_df[train_df['ClassId'] == cls]
        roi_cls_data = roi_df[roi_df['class_id'] == cls]
        f.write(f"클래스 {cls}:\n")
        f.write(f"  - 결함 수: {len(cls_data)}\n")
        f.write(f"  - ROI 패치 수: {len(roi_cls_data)}\n")
        f.write(f"  - 평균 면적: {roi_cls_data['area'].mean():.2f}\n")
        f.write(f"  - 평균 적합성 점수: {roi_cls_data['suitability_score'].mean():.3f}\n\n")
    
    # 3. 배경 타입 통계
    f.write("3. 배경 타입별 통계\n")
    f.write("-" * 80 + "\n")
    for bg_type in roi_df['background_type'].unique():
        bg_data = roi_df[roi_df['background_type'] == bg_type]
        f.write(f"{bg_type}:\n")
        f.write(f"  - 개수: {len(bg_data)}\n")
        f.write(f"  - 비율: {len(bg_data) / len(roi_df) * 100:.2f}%\n")
        f.write(f"  - 평균 적합성 점수: {bg_data['suitability_score'].mean():.3f}\n\n")
    
    # 4. 결함 서브타입 통계
    f.write("4. 결함 서브타입별 통계\n")
    f.write("-" * 80 + "\n")
    for subtype in roi_df['defect_subtype'].unique():
        subtype_data = roi_df[roi_df['defect_subtype'] == subtype]
        f.write(f"{subtype}:\n")
        f.write(f"  - 개수: {len(subtype_data)}\n")
        f.write(f"  - 비율: {len(subtype_data) / len(roi_df) * 100:.2f}%\n")
        f.write(f"  - 평균 선형성: {subtype_data['linearity'].mean():.3f}\n\n")
    
    # 5. 적합성 점수 통계
    f.write("5. 적합성 점수 통계\n")
    f.write("-" * 80 + "\n")
    f.write(f"적합성 점수:\n")
    f.write(f"  - 평균: {roi_df['suitability_score'].mean():.3f}\n")
    f.write(f"  - 중앙값: {roi_df['suitability_score'].median():.3f}\n")
    f.write(f"  - 최소값: {roi_df['suitability_score'].min():.3f}\n")
    f.write(f"  - 최대값: {roi_df['suitability_score'].max():.3f}\n\n")
    
    f.write(f"매칭 점수:\n")
    f.write(f"  - 평균: {roi_df['matching_score'].mean():.3f}\n")
    f.write(f"  - 중앙값: {roi_df['matching_score'].median():.3f}\n\n")
    
    f.write("추천 등급별 분포:\n")
    for rec in roi_df['recommendation'].unique():
        rec_count = len(roi_df[roi_df['recommendation'] == rec])
        f.write(f"  - {rec}: {rec_count} ({rec_count / len(roi_df) * 100:.2f}%)\n")

print(f"  → 저장: {output_dir / 'statistics_summary.txt'}")

print("\n" + "=" * 80)
print("모든 시각화 완료!")
print("=" * 80)
print(f"결과 저장 위치: {output_dir.absolute()}")
print("생성된 파일:")
print("  - 01_class_distribution.png")
print("  - 02_roi_metadata.png")
print("  - 03_background_type.png")
print("  - 04_defect_subtype.png")
print("  - 05_suitability_scores.png")
print("  - statistics_summary.txt")
print("=" * 80)
