#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean Background ROI Visualization
결함 없는 이미지에서 추출된 ROI 영역 시각화

이 스크립트는 결함이 없는(clean) 이미지에서 추출한 ROI 영역들을 시각화합니다.
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys
import random

# Setup paths
project_root = Path(__file__).parent
train_csv_path = project_root / "train.csv"
train_images_dir = project_root / "train_images"
output_dir = project_root / "outputs" / "clean_roi_visualization"

# Add src to path
sys.path.insert(0, str(project_root / "src"))

from analysis.background_characterization import BackgroundAnalyzer

# Background type colors for visualization
BG_COLORS = {
    'smooth': (0, 255, 0),           # Green
    'vertical_stripe': (0, 0, 255),   # Blue
    'horizontal_stripe': (255, 0, 0), # Red
    'textured': (255, 165, 0),        # Orange
    'complex_pattern': (255, 255, 0)  # Yellow
}

BG_TYPE_ORDER = ['smooth', 'vertical_stripe', 'horizontal_stripe', 'textured', 'complex_pattern']


def find_clean_images(train_csv_path, train_images_dir):
    """결함 없는 이미지 찾기 (train.csv에 없는 이미지)"""
    print("=" * 80)
    print("결함 없는 이미지 찾기...")
    print("=" * 80)
    
    # Get all images from directory
    all_images = set([f.name for f in train_images_dir.glob("*.jpg")])
    print(f"전체 이미지 수: {len(all_images)}")
    
    # Get images with defects (in train.csv)
    train_df = pd.read_csv(train_csv_path)
    defect_images = set(train_df['ImageId'].unique())
    print(f"결함 있는 이미지 (train.csv): {len(defect_images)}")
    
    # Clean images = all images - images with defects
    clean_images = list(all_images - defect_images)
    print(f"결함 없는 이미지 (NOT in train.csv): {len(clean_images)}")
    print()
    
    return clean_images


def analyze_background_rois(image_path, analyzer, roi_size=512, grid_size=64):
    """
    배경 분석 및 ROI 영역 추출
    
    Args:
        image_path: 이미지 경로
        analyzer: BackgroundAnalyzer 인스턴스
        roi_size: ROI 크기 (default: 512x512)
        grid_size: 그리드 크기 (default: 64x64)
        
    Returns:
        Tuple of (img_rgb, rois, bg_stats)
    """
    # Load image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    
    print(f"이미지 크기: {W}×{H}")
    
    # Analyze background
    print(f"배경 분석 중 (그리드: {grid_size}×{grid_size})...")
    analysis = analyzer.analyze_image(img)
    
    # Extract results
    bg_map = analysis['background_map']
    stability_map = analysis['stability_map']
    grid_h, grid_w = analysis['grid_shape']
    
    # Count background types
    unique_types, counts = np.unique(bg_map, return_counts=True)
    bg_stats = dict(zip(unique_types, counts))
    
    print(f"배경 타입 분포:")
    for bg_type in BG_TYPE_ORDER:
        if bg_type in bg_stats:
            count = bg_stats[bg_type]
            pct = (count / (grid_h * grid_w)) * 100
            print(f"  - {bg_type}: {count} cells ({pct:.1f}%)")
    
    # Select diverse ROI regions
    print(f"ROI 영역 선택 중 ({roi_size}×{roi_size})...")
    rois = []
    
    for target_type in BG_TYPE_ORDER:
        # Find grid cells with this background type
        matches = np.argwhere(bg_map == target_type)
        if len(matches) == 0:
            continue
        
        # Select cell with highest stability score
        best_idx = np.argmax([stability_map[m[0], m[1]] for m in matches])
        gi, gj = matches[best_idx]
        
        # Convert grid position to pixel position
        y_center = gi * grid_size + grid_size // 2
        x_center = gj * grid_size + grid_size // 2
        
        # Calculate ROI top-left corner
        y_roi = max(0, min(H - roi_size, y_center - roi_size // 2))
        x_roi = max(0, min(W - roi_size, x_center - roi_size // 2))
        
        # Verify ROI is within bounds
        if y_roi + roi_size > H or x_roi + roi_size > W:
            continue
        
        rois.append({
            'x': x_roi,
            'y': y_roi,
            'type': target_type,
            'score': float(stability_map[gi, gj]),
            'grid_pos': (gi, gj)
        })
        
        if len(rois) >= 5:
            break
    
    print(f"선택된 ROI: {len(rois)}개")
    print()
    
    return img_rgb, rois, bg_stats


def visualize_clean_image_with_rois(img_rgb, rois, image_id, output_path, roi_size=512):
    """
    결함 없는 이미지에 ROI 영역을 표시한 시각화 생성
    
    Args:
        img_rgb: RGB 이미지 배열
        rois: ROI 딕셔너리 리스트
        image_id: 이미지 파일명
        output_path: 저장 경로
        roi_size: ROI 크기
    """
    fig = plt.figure(figsize=(20, 5))
    
    # Plot 1: 전체 이미지에 ROI 박스 표시
    ax1 = plt.subplot(1, len(rois) + 1, 1)
    ax1.imshow(img_rgb)
    ax1.set_title(f"{image_id}\n결함 없는 이미지 (NOT in train.csv)\n{len(rois)}개 ROI 영역", 
                 fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Draw ROI boxes
    for idx, roi in enumerate(rois, 1):
        # Get color for this background type
        color = np.array(BG_COLORS.get(roi['type'], (128, 128, 128))) / 255.0
        
        # Draw rectangle
        rect = patches.Rectangle(
            (roi['x'], roi['y']), roi_size, roi_size,
            linewidth=3, edgecolor=color, facecolor='none'
        )
        ax1.add_patch(rect)
        
        # Add label
        label = f"ROI {idx}\n{roi['type']}"
        ax1.text(roi['x'], roi['y'] - 10, label, 
                color=color, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.5))
    
    # Plot 2+: 추출된 ROI 패치들
    for plot_idx, roi in enumerate(rois, start=2):
        ax = plt.subplot(1, len(rois) + 1, plot_idx)
        
        # Extract patch
        patch = img_rgb[roi['y']:roi['y'] + roi_size, 
                       roi['x']:roi['x'] + roi_size]
        
        ax.imshow(patch)
        
        # Title with metadata
        title = (f"ROI {plot_idx - 1}: {roi['type']}\n"
                f"위치: ({roi['x']}, {roi['y']})\n"
                f"안정성: {roi['score']:.3f}")
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        
        # Add colored border
        color = np.array(BG_COLORS.get(roi['type'], (128, 128, 128))) / 255.0
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
            spine.set_visible(True)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 저장 완료: {output_path.name}")


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 80)
    print("결함 없는 이미지 ROI 시각화")
    print("Clean Background ROI Visualization")
    print("=" * 80)
    print()
    
    # Check paths
    if not train_csv_path.exists():
        print(f"❌ ERROR: train.csv not found at {train_csv_path}")
        sys.exit(1)
    
    if not train_images_dir.exists():
        print(f"❌ ERROR: train_images/ not found at {train_images_dir}")
        sys.exit(1)
    
    # Find clean images
    clean_images = find_clean_images(train_csv_path, train_images_dir)
    
    # Select 3 random clean images
    print("=" * 80)
    print("3개 무작위 결함 없는 이미지 선택...")
    print("=" * 80)
    random.seed(42)  # For reproducibility
    selected_images = random.sample(clean_images, min(3, len(clean_images)))
    
    for i, img_id in enumerate(selected_images, 1):
        print(f"  {i}. {img_id}")
    print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize background analyzer
    print("=" * 80)
    print("BackgroundAnalyzer 초기화...")
    print("=" * 80)
    analyzer = BackgroundAnalyzer(grid_size=64, variance_threshold=100.0, edge_threshold=0.3)
    print("✅ 초기화 완료")
    print()
    
    # Process each image
    print("=" * 80)
    print("이미지 처리 중...")
    print("=" * 80)
    print()
    
    for idx, image_id in enumerate(selected_images, 1):
        print(f"[{idx}/3] {image_id} 처리 중...")
        print("-" * 80)
        
        # Full path to image
        img_path = train_images_dir / image_id
        
        # Analyze background and extract ROIs
        img_rgb, rois, bg_stats = analyze_background_rois(img_path, analyzer)
        
        # Create visualization
        output_path = output_dir / f"clean_image_{idx}_{image_id}.png"
        visualize_clean_image_with_rois(img_rgb, rois, image_id, output_path)
        
        print()
    
    # Print summary
    print("=" * 80)
    print("✅ 완료!")
    print("=" * 80)
    print()
    print(f"📁 출력 디렉토리: {output_dir}")
    print()
    print("생성된 파일:")
    for i, img_id in enumerate(selected_images, 1):
        print(f"  - clean_image_{i}_{img_id}.png")
    print()
    print("📊 배경 타입 색상 범례:")
    print("  🟢 Green  = smooth (균일한 표면)")
    print("  🔵 Blue   = vertical_stripe (수직 줄무늬)")
    print("  🔴 Red    = horizontal_stripe (수평 줄무늬)")
    print("  🟠 Orange = textured (텍스처)")
    print("  🟡 Yellow = complex_pattern (복잡한 패턴)")
    print()
    print("💡 각 이미지는:")
    print("  - 왼쪽: 전체 이미지에 ROI 박스 표시")
    print("  - 오른쪽: 추출된 ROI 패치들")
    print()
    print("✅ 이 영역들이 증강 데이터 생성 시 '배경'으로 사용됩니다!")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
