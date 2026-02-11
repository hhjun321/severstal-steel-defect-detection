"""
ROI 샘플 시각화 스크립트

추출된 ROI를 다양한 관점에서 시각화하여 품질을 확인합니다.

생성되는 시각화:
1. 클래스별 ROI 그리드 (각 클래스별 샘플)
2. 결함 유형별 ROI 그리드 (linear, blob, irregular, general)
3. 고품질 ROI Top 20 (suitability_score 상위)
4. Class 2 상세 분석 (특별 뷰)
5. 통계 대시보드 (분포 및 관계 분석)
6. 원본 이미지에 ROI 박스 오버레이

사용법:
    # 기본 실행 (모든 시각화)
    python scripts/visualize_roi_samples.py
    
    # 샘플 수 조정
    python scripts/visualize_roi_samples.py --samples_per_class 20
    
    # 출력 디렉토리 지정
    python scripts/visualize_roi_samples.py --output_dir my_viz

Author: CASDA Pipeline Team
Date: 2026-02-09
"""

import argparse
import ast
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm


class ROIVisualizer:
    """ROI 샘플 시각화 클래스"""
    
    def __init__(self, roi_dir: str, metadata_path: str, 
                 train_images_dir: str, output_dir: str, seed: int = 42):
        """
        초기화
        
        Args:
            roi_dir: ROI 패치가 저장된 디렉토리
            metadata_path: ROI 메타데이터 CSV 경로
            train_images_dir: 원본 훈련 이미지 디렉토리
            output_dir: 시각화 출력 디렉토리
            seed: 무작위 시드
        """
        self.roi_dir = Path(roi_dir)
        self.train_images_dir = Path(train_images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 무작위 시드 설정
        random.seed(seed)
        np.random.seed(seed)
        
        # 메타데이터 로드
        print(f"✓ ROI 메타데이터 로드 중: {metadata_path}")
        self.df = pd.read_csv(metadata_path)
        print(f"  총 {len(self.df)}개 ROI 로드됨")
        
        # 클래스별 색상
        self.class_colors = {
            1: '#e74c3c',  # 빨강
            2: '#2ecc71',  # 초록
            3: '#3498db',  # 파랑
            4: '#f39c12',  # 주황
        }
        
        # 배경 타입별 색상
        self.bg_colors = {
            'smooth': '#3498db',
            'vertical_stripe': '#2ecc71',
            'horizontal_stripe': '#f39c12',
            'textured': '#e74c3c',
            'complex_pattern': '#9b59b6'
        }
        
        # 한글 폰트 설정
        self._setup_font()
        
        print("✓ ROIVisualizer 초기화 완료\n")
    
    def _setup_font(self):
        """matplotlib 폰트 설정"""
        import platform
        import matplotlib.font_manager as fm
        
        system = platform.system()
        
        try:
            if system == 'Windows':
                plt.rcParams['font.family'] = 'Malgun Gothic'
            elif system == 'Darwin':  # Mac
                plt.rcParams['font.family'] = 'AppleGothic'
            else:  # Linux
                plt.rcParams['font.family'] = 'NanumGothic'
            
            plt.rcParams['axes.unicode_minus'] = False
            print("✓ 한글 폰트 설정 완료")
        except:
            print("⚠️ 한글 폰트를 찾을 수 없습니다. 기본 폰트 사용")
    
    def _parse_bbox(self, bbox_str: str) -> Tuple[int, int, int, int]:
        """
        bbox 문자열을 튜플로 파싱
        
        Args:
            bbox_str: "(x1, y1, x2, y2)" 형식의 문자열
            
        Returns:
            (x1, y1, x2, y2) 튜플
        """
        try:
            return ast.literal_eval(bbox_str)
        except:
            return (0, 0, 0, 0)
    
    def _load_roi_image(self, row: pd.Series) -> Optional[np.ndarray]:
        """
        ROI 패치 이미지 로드
        
        Args:
            row: 메타데이터 행
            
        Returns:
            RGB 이미지 배열 또는 None
        """
        filename = f"{row['image_id']}_class{row['class_id']}_region{row['region_id']}.png"
        filepath = self.roi_dir / 'images' / filename
        
        if not filepath.exists():
            return None
        
        img = cv2.imread(str(filepath))
        if img is None:
            return None
        
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def _load_roi_mask(self, row: pd.Series) -> Optional[np.ndarray]:
        """
        ROI 마스크 로드
        
        Args:
            row: 메타데이터 행
            
        Returns:
            Binary 마스크 배열 또는 None
        """
        filename = f"{row['image_id']}_class{row['class_id']}_region{row['region_id']}.png"
        filepath = self.roi_dir / 'masks' / filename
        
        if not filepath.exists():
            return None
        
        mask = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        
        return (mask > 127).astype(np.uint8)
    
    def _load_original_image(self, image_id: str) -> Optional[np.ndarray]:
        """
        원본 이미지 로드
        
        Args:
            image_id: 이미지 ID
            
        Returns:
            RGB 이미지 배열 또는 None
        """
        filepath = self.train_images_dir / image_id
        
        if not filepath.exists():
            return None
        
        img = cv2.imread(str(filepath))
        if img is None:
            return None
        
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def _resize_for_display(self, img: np.ndarray, 
                           target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """
        종횡비를 유지하며 이미지 리사이즈
        
        Args:
            img: 입력 이미지
            target_size: 목표 크기 (height, width)
            
        Returns:
            리사이즈된 이미지
        """
        if img is None or img.size == 0:
            return np.zeros((*target_size, 3), dtype=np.uint8)
        
        h, w = img.shape[:2]
        target_h, target_w = target_size
        
        # 종횡비 계산
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 리사이즈
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 패딩
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        
        if len(resized.shape) == 2:  # Grayscale
            padded = np.zeros(target_size, dtype=np.uint8)
            padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        else:  # RGB
            padded = np.zeros((*target_size, 3), dtype=np.uint8)
            padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        return padded
    
    def _create_metadata_text(self, row: pd.Series, include_score: bool = True) -> str:
        """
        ROI 메타데이터 텍스트 생성
        
        Args:
            row: 메타데이터 행
            include_score: 점수 포함 여부
            
        Returns:
            메타데이터 텍스트
        """
        parts = [
            f"C{row['class_id']}",
            row['defect_subtype'][:8],  # 길이 제한
            row['background_type'][:8]
        ]
        
        if include_score:
            parts.append(f"{row['suitability_score']:.2f}")
        
        return " | ".join(parts)
    
    def plot_class_grids(self, samples_per_class: int = 10):
        """
        클래스별 ROI 그리드 시각화
        
        Args:
            samples_per_class: 각 클래스당 샘플 수
        """
        print("[1/6] 클래스별 그리드 생성 중...")
        
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('ROI Samples by Class', fontsize=20, fontweight='bold', y=0.995)
        
        n_cols = 5
        class_ids = [1, 2, 3, 4]
        
        for idx, class_id in enumerate(class_ids):
            # 해당 클래스 ROI 가져오기
            class_df = self.df[self.df['class_id'] == class_id]
            
            # 샘플링 (Class 2는 전체)
            if class_id == 2 or len(class_df) <= samples_per_class:
                samples = class_df
                n_samples = len(samples)
            else:
                samples = class_df.sample(n=samples_per_class, random_state=42)
                n_samples = samples_per_class
            
            # 행 수 계산
            n_rows = (n_samples + n_cols - 1) // n_cols
            
            # 서브플롯 시작 위치
            base_idx = idx * 100  # 각 클래스에 충분한 공간
            
            # 클래스 제목
            title_ax = plt.subplot2grid((40, n_cols), (idx * 10, 0), colspan=n_cols, fig=fig)
            title_text = f"Class {class_id} ROIs ({n_samples} samples)"
            if class_id == 2:
                title_text += " ⚠️"
            title_ax.text(0.5, 0.5, title_text, 
                         ha='center', va='center', 
                         fontsize=16, fontweight='bold',
                         color=self.class_colors[class_id])
            title_ax.axis('off')
            
            # ROI 이미지 표시
            for i, (_, row) in enumerate(samples.iterrows()):
                if i >= n_cols * 2:  # 최대 2행만 표시
                    break
                
                row_idx = i // n_cols
                col_idx = i % n_cols
                
                ax = plt.subplot2grid((40, n_cols), 
                                     (idx * 10 + 1 + row_idx * 3, col_idx),
                                     rowspan=3, fig=fig)
                
                # 이미지 로드 및 표시
                img = self._load_roi_image(row)
                if img is not None:
                    img_resized = self._resize_for_display(img, (150, 150))
                    ax.imshow(img_resized)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                
                # 메타데이터 표시
                meta_text = self._create_metadata_text(row)
                ax.set_title(meta_text, fontsize=8)
                ax.axis('off')
            
            print(f"  - Class {class_id}: {n_samples} samples")
        
        plt.tight_layout()
        output_path = self.output_dir / '01_class_grids.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 저장: {output_path}\n")
    
    def plot_defect_type_grids(self, samples_per_type: int = 10):
        """
        결함 유형별 ROI 그리드 시각화
        
        Args:
            samples_per_type: 각 유형당 샘플 수
        """
        print("[2/6] 결함 유형별 그리드 생성 중...")
        
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('ROI Samples by Defect Type', fontsize=20, fontweight='bold', y=0.995)
        
        n_cols = 5
        defect_types = ['linear_scratch', 'compact_blob', 'irregular', 'general']
        
        for idx, defect_type in enumerate(defect_types):
            # 해당 유형 ROI 가져오기
            type_df = self.df[self.df['defect_subtype'] == defect_type]
            
            # 샘플링
            if len(type_df) <= samples_per_type:
                samples = type_df
                n_samples = len(samples)
            else:
                samples = type_df.sample(n=samples_per_type, random_state=42)
                n_samples = samples_per_type
            
            # 서브플롯 시작 위치
            title_ax = plt.subplot2grid((40, n_cols), (idx * 10, 0), colspan=n_cols, fig=fig)
            title_text = f"{defect_type.replace('_', ' ').title()} ({n_samples} samples)"
            if n_samples < samples_per_type:
                title_text += f" (전체)"
            title_ax.text(0.5, 0.5, title_text, 
                         ha='center', va='center', 
                         fontsize=16, fontweight='bold')
            title_ax.axis('off')
            
            # ROI 이미지 표시
            for i, (_, row) in enumerate(samples.iterrows()):
                if i >= n_cols * 2:  # 최대 2행만 표시
                    break
                
                row_idx = i // n_cols
                col_idx = i % n_cols
                
                ax = plt.subplot2grid((40, n_cols), 
                                     (idx * 10 + 1 + row_idx * 3, col_idx),
                                     rowspan=3, fig=fig)
                
                # 이미지 로드 및 표시
                img = self._load_roi_image(row)
                if img is not None:
                    img_resized = self._resize_for_display(img, (150, 150))
                    ax.imshow(img_resized)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                
                # 메타데이터 표시
                meta_text = f"C{row['class_id']} | {row['background_type'][:8]} | {row['suitability_score']:.2f}"
                ax.set_title(meta_text, fontsize=8)
                ax.axis('off')
            
            print(f"  - {defect_type}: {n_samples} samples")
        
        plt.tight_layout()
        output_path = self.output_dir / '02_defect_type_grids.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 저장: {output_path}\n")
    
    def plot_top_quality_rois(self, top_n: int = 20):
        """
        고품질 ROI Top N 시각화
        
        Args:
            top_n: 상위 N개
        """
        print("[3/6] 고품질 ROI Top 20 생성 중...")
        
        # 품질 점수로 정렬
        top_rois = self.df.nlargest(top_n, 'suitability_score')
        
        # 그리드 설정
        n_cols = 4
        n_rows = (top_n + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
        fig.suptitle(f'Top {top_n} High Quality ROIs (Sorted by Suitability Score)', 
                     fontsize=18, fontweight='bold')
        
        axes = axes.flatten() if top_n > 1 else [axes]
        
        for i, (_, row) in enumerate(top_rois.iterrows()):
            ax = axes[i]
            
            # 이미지 로드 및 표시
            img = self._load_roi_image(row)
            if img is not None:
                img_resized = self._resize_for_display(img, (200, 200))
                ax.imshow(img_resized)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
            
            # 제목 (순위 + 점수)
            rank = i + 1
            title = f"Rank #{rank}: {row['suitability_score']:.4f}\n"
            title += f"C{row['class_id']} | {row['defect_subtype'][:8]} | {row['background_type'][:8]}"
            
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # 남은 서브플롯 숨기기
        for i in range(top_n, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        output_path = self.output_dir / '03_top_quality.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  최고 점수: {top_rois.iloc[0]['suitability_score']:.4f}")
        print(f"  최저 점수 (Top {top_n}): {top_rois.iloc[-1]['suitability_score']:.4f}")
        print(f"  ✓ 저장: {output_path}\n")
    
    def plot_class2_detailed(self):
        """
        Class 2 상세 분석 (특별 뷰)
        """
        print("[4/6] Class 2 상세 분석 생성 중...")
        
        # Class 2 ROI 가져오기
        class2_df = self.df[self.df['class_id'] == 2]
        n_samples = len(class2_df)
        
        print(f"  ⚠️ Class 2는 {n_samples}개 샘플만 있습니다!")
        
        if n_samples == 0:
            print("  ⚠️ Class 2 샘플이 없습니다. 건너뜁니다.\n")
            return
        
        # 그리드 설정 (각 ROI당 1행)
        fig = plt.figure(figsize=(20, n_samples * 6))
        
        # 경고 제목
        fig.text(0.5, 0.98, 
                f'⚠️ Class 2 Detailed Analysis - Only {n_samples} Samples Available ⚠️',
                ha='center', fontsize=18, fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        for idx, (_, row) in enumerate(class2_df.iterrows()):
            # 4열 레이아웃: 원본 이미지 | ROI 패치 | 마스크 | 메타데이터
            base_row = idx * 6 + 1
            
            # 1. 원본 이미지 (ROI 박스 표시)
            ax1 = plt.subplot2grid((n_samples * 6, 4), (base_row, 0), rowspan=5)
            original_img = self._load_original_image(row['image_id'])
            
            if original_img is not None:
                ax1.imshow(original_img)
                
                # ROI 박스 그리기
                bbox = self._parse_bbox(row['roi_bbox'])
                x1, y1, x2, y2 = bbox
                rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1,
                                         linewidth=3, edgecolor='red',
                                         facecolor='none')
                ax1.add_patch(rect)
                ax1.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 
                        'r-', linewidth=2)
            else:
                ax1.text(0.5, 0.5, 'Original Image\nNot Found', 
                        ha='center', va='center')
            
            ax1.set_title(f'Original Image\n{row["image_id"]}', fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            # 2. ROI 패치 (확대)
            ax2 = plt.subplot2grid((n_samples * 6, 4), (base_row, 1), rowspan=5)
            roi_img = self._load_roi_image(row)
            
            if roi_img is not None:
                ax2.imshow(roi_img)
            else:
                ax2.text(0.5, 0.5, 'ROI Patch\nNot Found', 
                        ha='center', va='center')
            
            ax2.set_title('ROI Patch (Enlarged)', fontsize=12, fontweight='bold')
            ax2.axis('off')
            
            # 3. 마스크
            ax3 = plt.subplot2grid((n_samples * 6, 4), (base_row, 2), rowspan=5)
            mask = self._load_roi_mask(row)
            
            if mask is not None:
                ax3.imshow(mask, cmap='gray')
            else:
                ax3.text(0.5, 0.5, 'Mask\nNot Found', 
                        ha='center', va='center')
            
            ax3.set_title('Defect Mask', fontsize=12, fontweight='bold')
            ax3.axis('off')
            
            # 4. 메타데이터 텍스트 박스
            ax4 = plt.subplot2grid((n_samples * 6, 4), (base_row, 3), rowspan=5)
            ax4.axis('off')
            
            # 메타데이터 텍스트 생성
            metadata_text = f"""
ROI #{idx + 1} Metadata
{'='*30}

Image ID: {row['image_id']}
Class: {row['class_id']}
Region ID: {row['region_id']}

Defect Type: {row['defect_subtype']}
Background: {row['background_type']}

Scores:
  Suitability: {row['suitability_score']:.4f}
  Matching: {row['matching_score']:.4f}
  Continuity: {row['continuity_score']:.4f}
  Stability: {row['stability_score']:.4f}

Defect Metrics:
  Area: {row['area']:.0f} px
  Linearity: {row['linearity']:.3f}
  Solidity: {row['solidity']:.3f}
  Extent: {row['extent']:.3f}
  Aspect Ratio: {row['aspect_ratio']:.3f}

Recommendation: {row['recommendation']}

ROI BBox: {row['roi_bbox']}
            """
            
            ax4.text(0.1, 0.95, metadata_text, 
                    fontsize=9, family='monospace',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        output_path = self.output_dir / '04_class2_detailed.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 저장: {output_path}\n")
    
    def plot_statistics_dashboard(self):
        """
        통계 대시보드 시각화
        """
        print("[5/6] 통계 대시보드 생성 중...")
        
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('ROI Statistics Dashboard', fontsize=20, fontweight='bold')
        
        # 2x3 그리드
        
        # 1. 클래스별 ROI 개수 (바 차트)
        ax1 = plt.subplot(2, 3, 1)
        class_counts = self.df['class_id'].value_counts().sort_index()
        bars = ax1.bar(class_counts.index, class_counts.values, 
                      color=[self.class_colors[c] for c in class_counts.index])
        ax1.set_xlabel('Class ID', fontsize=12)
        ax1.set_ylabel('Number of ROIs', fontsize=12)
        ax1.set_title('ROI Distribution by Class', fontsize=14, fontweight='bold')
        ax1.set_xticks(class_counts.index)
        
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Class 2 강조
        if 2 in class_counts.index:
            idx = list(class_counts.index).index(2)
            bars[idx].set_edgecolor('red')
            bars[idx].set_linewidth(3)
        
        # 2. 결함 유형 분포 (파이 차트)
        ax2 = plt.subplot(2, 3, 2)
        defect_counts = self.df['defect_subtype'].value_counts()
        colors_defect = plt.cm.Set3(range(len(defect_counts)))
        ax2.pie(defect_counts.values, labels=defect_counts.index, 
               autopct='%1.1f%%', startangle=90, colors=colors_defect)
        ax2.set_title('Defect Type Distribution', fontsize=14, fontweight='bold')
        
        # 3. 배경 유형 분포 (바 차트)
        ax3 = plt.subplot(2, 3, 3)
        bg_counts = self.df['background_type'].value_counts()
        colors_bg = [self.bg_colors.get(bg, '#cccccc') for bg in bg_counts.index]
        bars_bg = ax3.barh(bg_counts.index, bg_counts.values, color=colors_bg)
        ax3.set_xlabel('Number of ROIs', fontsize=12)
        ax3.set_ylabel('Background Type', fontsize=12)
        ax3.set_title('Background Type Distribution', fontsize=14, fontweight='bold')
        
        # 값 표시
        for bar in bars_bg:
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}',
                    ha='left', va='center', fontsize=10, fontweight='bold')
        
        # 4. 품질 점수 분포 (히스토그램)
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(self.df['suitability_score'], bins=20, color='skyblue', edgecolor='black')
        ax4.axvline(self.df['suitability_score'].mean(), 
                   color='red', linestyle='--', linewidth=2, label='Mean')
        ax4.axvline(self.df['suitability_score'].median(), 
                   color='green', linestyle='--', linewidth=2, label='Median')
        ax4.set_xlabel('Suitability Score', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Quality Score Distribution', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 클래스 x 결함 유형 크로스탭 (히트맵)
        ax5 = plt.subplot(2, 3, 5)
        crosstab_class_defect = pd.crosstab(self.df['class_id'], self.df['defect_subtype'])
        sns.heatmap(crosstab_class_defect, annot=True, fmt='d', cmap='YlOrRd', 
                   ax=ax5, cbar_kws={'label': 'Count'})
        ax5.set_xlabel('Defect Type', fontsize=12)
        ax5.set_ylabel('Class ID', fontsize=12)
        ax5.set_title('Class vs Defect Type', fontsize=14, fontweight='bold')
        
        # 6. 결함 유형 x 배경 유형 매칭 (히트맵)
        ax6 = plt.subplot(2, 3, 6)
        crosstab_defect_bg = pd.crosstab(self.df['defect_subtype'], self.df['background_type'])
        sns.heatmap(crosstab_defect_bg, annot=True, fmt='d', cmap='BuPu', 
                   ax=ax6, cbar_kws={'label': 'Count'})
        ax6.set_xlabel('Background Type', fontsize=12)
        ax6.set_ylabel('Defect Type', fontsize=12)
        ax6.set_title('Defect vs Background Matching', fontsize=14, fontweight='bold')
        plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        output_path = self.output_dir / '05_statistics.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  - 클래스 분포")
        print(f"  - 결함 유형 분포")
        print(f"  - 배경 유형 분포")
        print(f"  - 품질 점수 분포")
        print(f"  - 크로스 분석 히트맵")
        print(f"  ✓ 저장: {output_path}\n")
    
    def plot_sample_with_overlay(self, image_ids: List[str] = None, max_images: int = 5):
        """
        원본 이미지에 ROI 박스 오버레이
        
        Args:
            image_ids: 시각화할 이미지 ID 리스트 (None이면 자동 선택)
            max_images: 최대 이미지 수
        """
        print("[6/6] ROI 오버레이 생성 중...")
        
        # 오버레이 디렉토리 생성
        overlay_dir = self.output_dir / '06_roi_overlays'
        overlay_dir.mkdir(exist_ok=True)
        
        # 이미지 ID 선택
        if image_ids is None:
            # ROI가 많은 이미지 선택
            roi_per_image = self.df['image_id'].value_counts()
            image_ids = roi_per_image.head(max_images).index.tolist()
        else:
            image_ids = image_ids[:max_images]
        
        for image_id in image_ids:
            # 해당 이미지의 모든 ROI
            image_rois = self.df[self.df['image_id'] == image_id]
            n_rois = len(image_rois)
            
            # 원본 이미지 로드
            original_img = self._load_original_image(image_id)
            
            if original_img is None:
                print(f"  ⚠️ {image_id}: 원본 이미지 없음")
                continue
            
            # 플롯 생성
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.imshow(original_img)
            
            # 각 ROI 박스 그리기
            for idx, (_, row) in enumerate(image_rois.iterrows()):
                bbox = self._parse_bbox(row['roi_bbox'])
                x1, y1, x2, y2 = bbox
                
                # 클래스별 색상
                color = self.class_colors[row['class_id']]
                
                # 박스 그리기
                rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1,
                                         linewidth=3, edgecolor=color,
                                         facecolor='none')
                ax.add_patch(rect)
                
                # 라벨 (ROI 번호 + 클래스)
                label = f"#{idx+1}\nC{row['class_id']}"
                ax.text(x1, y1-5, label, 
                       color=color, fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'{image_id} - {n_rois} ROIs', 
                        fontsize=16, fontweight='bold')
            ax.axis('off')
            
            # 범례 추가
            legend_elements = [mpatches.Patch(facecolor=self.class_colors[c], 
                                             edgecolor='black',
                                             label=f'Class {c}')
                             for c in sorted(image_rois['class_id'].unique())]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
            
            plt.tight_layout()
            output_path = overlay_dir / f'{image_id.replace(".jpg", "")}_overlay.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  - {image_id} ({n_rois} ROIs)")
        
        print(f"  ✓ 저장: {overlay_dir}/\n")
    
    def create_readme(self):
        """
        README.txt 생성
        """
        readme_content = f"""ROI Visualization Summary
{'='*80}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW
{'='*80}
Total ROIs: {len(self.df)}

ROIs per Class:
"""
        
        class_counts = self.df['class_id'].value_counts().sort_index()
        for class_id, count in class_counts.items():
            warning = " ⚠️ (Very Low!)" if count < 10 else ""
            readme_content += f"  - Class {class_id}: {count} samples{warning}\n"
        
        readme_content += f"""
ROIs per Defect Subtype:
"""
        defect_counts = self.df['defect_subtype'].value_counts()
        for defect_type, count in defect_counts.items():
            readme_content += f"  - {defect_type}: {count} samples\n"
        
        readme_content += f"""
ROIs per Background Type:
"""
        bg_counts = self.df['background_type'].value_counts()
        for bg_type, count in bg_counts.items():
            readme_content += f"  - {bg_type}: {count} samples\n"
        
        readme_content += f"""
QUALITY STATISTICS
{'='*80}
Suitability Score:
  - Mean: {self.df['suitability_score'].mean():.4f}
  - Median: {self.df['suitability_score'].median():.4f}
  - Min: {self.df['suitability_score'].min():.4f}
  - Max: {self.df['suitability_score'].max():.4f}

Quality Distribution:
  - High (>0.8): {len(self.df[self.df['suitability_score'] > 0.8])} samples
  - Medium (0.6-0.8): {len(self.df[(self.df['suitability_score'] >= 0.6) & (self.df['suitability_score'] <= 0.8)])} samples
  - Low (<0.6): {len(self.df[self.df['suitability_score'] < 0.6])} samples

Recommendation Distribution:
"""
        rec_counts = self.df['recommendation'].value_counts()
        for rec, count in rec_counts.items():
            readme_content += f"  - {rec}: {count} samples\n"
        
        readme_content += f"""
GENERATED FILES
{'='*80}
1. 01_class_grids.png
   - Class-wise ROI sample grid
   - Shows representative samples from each class

2. 02_defect_type_grids.png
   - Defect type-wise ROI sample grid
   - Groups by linear_scratch, compact_blob, irregular, general

3. 03_top_quality.png
   - Top 20 high-quality ROIs
   - Sorted by suitability_score

4. 04_class2_detailed.png
   - Detailed analysis of Class 2 samples
   - Shows original image, ROI patch, mask, and full metadata

5. 05_statistics.png
   - Statistical dashboard with 6 charts
   - Distribution analysis and cross-tabulations

6. 06_roi_overlays/
   - Original images with ROI bounding boxes
   - Color-coded by class

WARNINGS & RECOMMENDATIONS
{'='*80}
"""
        
        # Class 2 경고
        class2_count = len(self.df[self.df['class_id'] == 2])
        if class2_count < 10:
            readme_content += f"""
⚠️  CRITICAL: Class 2 has only {class2_count} samples!
    This severe class imbalance may cause training issues.
    Recommendations:
    - Review extraction parameters (min_suitability threshold)
    - Check if Class 2 defects exist in the original dataset
    - Consider data augmentation strategies
    - May need to adjust ROI extraction criteria
"""
        
        # 저품질 ROI 경고
        low_quality_count = len(self.df[self.df['suitability_score'] < 0.6])
        if low_quality_count > len(self.df) * 0.3:
            readme_content += f"""
⚠️  WARNING: {low_quality_count} ROIs ({low_quality_count/len(self.df)*100:.1f}%) have low suitability (<0.6)
    Consider:
    - Increasing min_suitability threshold
    - Reviewing extraction criteria
    - Manual inspection of low-quality samples
"""
        
        readme_content += f"""
NEXT STEPS
{'='*80}
1. Review visualizations to verify ROI quality
2. Check Class 2 samples manually (if count is low)
3. Adjust extraction parameters if needed:
   - min_suitability threshold
   - ROI size
   - Background/defect matching rules
4. Proceed to ControlNet data preparation if satisfied

For questions or issues, refer to:
- README_ROI_KR.md
- IMPLEMENTATION_SUMMARY_KR.md
- PROJECT(roi).md
"""
        
        # README 저장
        readme_path = self.output_dir / 'README.txt'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"✓ README.txt 생성 완료: {readme_path}")
    
    def create_all_visualizations(self, 
                                   samples_per_class: int = 10,
                                   samples_per_type: int = 10,
                                   top_n: int = 20,
                                   overlay_images: int = 5):
        """
        모든 시각화 생성
        
        Args:
            samples_per_class: 각 클래스당 샘플 수
            samples_per_type: 각 결함 유형당 샘플 수
            top_n: 고품질 ROI 개수
            overlay_images: 오버레이할 이미지 수
        """
        print("\n" + "="*80)
        print("ROI 샘플 시각화 시작...")
        print("="*80 + "\n")
        
        # 1. 클래스별 그리드
        self.plot_class_grids(samples_per_class)
        
        # 2. 결함 유형별 그리드
        self.plot_defect_type_grids(samples_per_type)
        
        # 3. 고품질 ROI Top N
        self.plot_top_quality_rois(top_n)
        
        # 4. Class 2 상세 분석
        self.plot_class2_detailed()
        
        # 5. 통계 대시보드
        self.plot_statistics_dashboard()
        
        # 6. ROI 오버레이
        self.plot_sample_with_overlay(max_images=overlay_images)
        
        # README 생성
        self.create_readme()
        
        print("="*80)
        print("✓ 시각화 완료!")
        print(f"출력 디렉토리: {self.output_dir}")
        print("="*80)
        
        # 경고 메시지
        class2_count = len(self.df[self.df['class_id'] == 2])
        if class2_count < 10:
            print(f"\n⚠️  중요: Class 2가 {class2_count}개 샘플만 있습니다!")
            print("   데이터 불균형 문제를 검토해주세요.\n")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='ROI 샘플 시각화 스크립트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 실행 (모든 시각화)
  python scripts/visualize_roi_samples.py
  
  # 샘플 수 조정
  python scripts/visualize_roi_samples.py --samples_per_class 20 --top_n 30
  
  # 출력 디렉토리 지정
  python scripts/visualize_roi_samples.py --output_dir my_visualizations
        """
    )
    
    parser.add_argument(
        '--roi_dir',
        type=str,
        default='data/processed/roi_patches',
        help='ROI 패치 디렉토리 (기본: data/processed/roi_patches)'
    )
    
    parser.add_argument(
        '--metadata',
        type=str,
        default='data/processed/roi_patches/roi_metadata.csv',
        help='ROI 메타데이터 CSV 경로 (기본: data/processed/roi_patches/roi_metadata.csv)'
    )
    
    parser.add_argument(
        '--train_images_dir',
        type=str,
        default='train_images',
        help='원본 훈련 이미지 디렉토리 (기본: train_images)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='visualizations/roi_samples',
        help='출력 디렉토리 (기본: visualizations/roi_samples)'
    )
    
    parser.add_argument(
        '--samples_per_class',
        type=int,
        default=10,
        help='각 클래스당 표시할 샘플 수 (기본: 10)'
    )
    
    parser.add_argument(
        '--samples_per_type',
        type=int,
        default=10,
        help='각 결함 유형당 표시할 샘플 수 (기본: 10)'
    )
    
    parser.add_argument(
        '--top_n',
        type=int,
        default=20,
        help='고품질 ROI 개수 (기본: 20)'
    )
    
    parser.add_argument(
        '--overlay_images',
        type=int,
        default=5,
        help='오버레이할 이미지 수 (기본: 5)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='무작위 시드 (기본: 42)'
    )
    
    args = parser.parse_args()
    
    # 시각화 실행
    try:
        visualizer = ROIVisualizer(
            roi_dir=args.roi_dir,
            metadata_path=args.metadata,
            train_images_dir=args.train_images_dir,
            output_dir=args.output_dir,
            seed=args.seed
        )
        
        visualizer.create_all_visualizations(
            samples_per_class=args.samples_per_class,
            samples_per_type=args.samples_per_type,
            top_n=args.top_n,
            overlay_images=args.overlay_images
        )
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
