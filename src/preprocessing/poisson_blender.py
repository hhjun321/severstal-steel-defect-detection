"""
Poisson Blender - ControlNet 생성 결함을 결함 없는 원본 이미지에 합성
Poisson Blender - Composites ControlNet-generated defects onto defect-free images

This module provides Poisson Blending (cv2.seamlessClone) based composition
of 512x512 ControlNet-generated defect ROI images onto 1600x256 defect-free
original steel images, producing realistic full-size augmented training images.

핵심 기능:
- 512x512 생성 이미지 → 256x256 ROI 스케일로 다운스케일
- 힌트 Red 채널에서 결함 마스크 추출
- 마스크 확장(dilation)으로 Poisson 블렌딩 경계 맥락 제공
- cv2.seamlessClone()으로 결함을 깨끗한 배경에 자연스럽게 합성
- 1600x256 전체 크기 마스크 및 YOLO bbox 생성
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CompositionResult:
    """Poisson Blending 합성 결과"""
    composited_image: np.ndarray      # 1600x256x3 합성 이미지
    full_mask: np.ndarray             # 1600x256 전체 크기 이진 마스크
    bboxes: List[List[float]]         # 정규화 YOLO bbox [[cx, cy, w, h], ...]
    labels: List[int]                 # bbox별 class_id (0-indexed)
    success: bool                     # 합성 성공 여부
    blend_method: str                 # 사용된 블렌딩 방식
    message: str                      # 상태 메시지


class PoissonBlender:
    """
    ControlNet 생성 결함 이미지를 결함 없는 원본 이미지에 Poisson 블렌딩하는 모듈.
    
    파이프라인:
      1. 힌트 Red 채널에서 이진 마스크 추출
      2. 512x512 → 256x256 다운스케일 (원본 ROI 스케일 보존)
      3. 마스크 확장 (Poisson 블렌딩 경계 맥락 제공)
      4. roi_bbox 기반 붙여넣기 위치 계산
      5. cv2.seamlessClone()으로 Poisson 블렌딩
      6. 1600x256 전체 크기 마스크 + YOLO bbox 생성
    """
    
    # Poisson Blending 모드 상수
    NORMAL_CLONE = cv2.NORMAL_CLONE
    MIXED_CLONE = cv2.MIXED_CLONE
    
    def __init__(
        self,
        dilation_px: int = 15,
        blend_mode: int = cv2.NORMAL_CLONE,
        mask_threshold: int = 127,
        min_defect_area: int = 16,
    ):
        """
        Args:
            dilation_px: 마스크 확장 픽셀 수 (기본 15px, 타원 커널)
            blend_mode: cv2.NORMAL_CLONE 또는 cv2.MIXED_CLONE
            mask_threshold: 힌트 Red 채널 이진화 임계값 (기본 127)
            min_defect_area: YOLO bbox 추출 시 최소 면적 (기본 16px)
        """
        self.dilation_px = dilation_px
        self.blend_mode = blend_mode
        self.mask_threshold = mask_threshold
        self.min_defect_area = min_defect_area
        
        # dilation 커널 사전 생성
        if dilation_px > 0:
            kernel_size = 2 * dilation_px + 1
            self.dilation_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
        else:
            self.dilation_kernel = None
    
    # ──────────────────────────────────────────────
    # 마스크 추출
    # ──────────────────────────────────────────────
    
    def extract_mask_from_hint(
        self, hint_image: np.ndarray, threshold: Optional[int] = None
    ) -> np.ndarray:
        """
        힌트 이미지의 Red 채널에서 이진 결함 마스크를 추출한다.
        
        힌트 이미지는 3채널 PNG로, Red 채널에 결함 영역이 4단계 강도로 인코딩됨.
        임계값 처리하면 이진 마스크(0/255)를 얻음.
        
        Args:
            hint_image: BGR 3채널 힌트 이미지 (H, W, 3)
            threshold: 이진화 임계값 (None이면 self.mask_threshold 사용)
            
        Returns:
            이진 마스크 (H, W), 값은 0 또는 255
        """
        if threshold is None:
            threshold = self.mask_threshold
        
        if hint_image is None:
            raise ValueError("힌트 이미지가 None입니다")
        
        # BGR에서 Red 채널 = index 2
        if len(hint_image.shape) == 3 and hint_image.shape[2] == 3:
            red_channel = hint_image[:, :, 2]
        elif len(hint_image.shape) == 2:
            # 이미 그레이스케일인 경우
            red_channel = hint_image
        else:
            raise ValueError(f"예상치 못한 힌트 이미지 형태: {hint_image.shape}")
        
        _, binary_mask = cv2.threshold(
            red_channel, threshold, 255, cv2.THRESH_BINARY
        )
        return binary_mask
    
    # ──────────────────────────────────────────────
    # 다운스케일
    # ──────────────────────────────────────────────
    
    def downscale_to_roi(
        self,
        image_512: np.ndarray,
        mask_512: np.ndarray,
        target_h: int = 256,
        target_w: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        512x512 생성 이미지와 마스크를 원본 ROI 크기로 다운스케일한다.
        
        Args:
            image_512: 512x512x3 생성 이미지
            mask_512: 512x512 이진 마스크
            target_h: 목표 높이 (기본 256)
            target_w: 목표 너비 (기본 256)
            
        Returns:
            (image_roi, mask_roi) 튜플
            - image_roi: target_h x target_w x 3
            - mask_roi: target_h x target_w, 이진 (0/255)
        """
        # 이미지: INTER_AREA (축소 시 앨리어싱 방지에 최적)
        image_roi = cv2.resize(
            image_512, (target_w, target_h), interpolation=cv2.INTER_AREA
        )
        
        # 마스크: INTER_NEAREST (이진 특성 보존)
        mask_roi = cv2.resize(
            mask_512, (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )
        
        return image_roi, mask_roi
    
    # ──────────────────────────────────────────────
    # 마스크 확장
    # ──────────────────────────────────────────────
    
    def dilate_mask(
        self, mask: np.ndarray, dilation_px: Optional[int] = None
    ) -> np.ndarray:
        """
        이진 마스크를 확장(dilate)하여 Poisson 블렌딩 경계 맥락을 제공한다.
        
        Args:
            mask: 이진 마스크 (H, W), 값 0 또는 255
            dilation_px: 확장 픽셀 (None이면 self.dilation_px 사용)
            
        Returns:
            확장된 이진 마스크 (H, W)
        """
        if dilation_px is None:
            dilation_px = self.dilation_px
        
        if dilation_px <= 0:
            return mask.copy()
        
        # 요청된 크기가 사전 생성 커널과 다르면 새로 생성
        if dilation_px != self.dilation_px or self.dilation_kernel is None:
            kernel_size = 2 * dilation_px + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
        else:
            kernel = self.dilation_kernel
        
        dilated = cv2.dilate(mask, kernel, iterations=1)
        
        # seamlessClone 요구사항: 마스크 비제로 픽셀이 source 가장자리에
        # 닿으면 crash 또는 심한 아티팩트 발생 (Dirichlet 경계 조건 위반).
        # 가장자리 1px을 강제로 0으로 설정하여 안전한 경계를 보장한다.
        dilated[0, :] = 0
        dilated[-1, :] = 0
        dilated[:, 0] = 0
        dilated[:, -1] = 0
        
        return dilated
    
    # ──────────────────────────────────────────────
    # 붙여넣기 위치 계산
    # ──────────────────────────────────────────────
    
    def compute_paste_center(
        self,
        roi_bbox: Tuple[int, int, int, int],
        target_shape: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        roi_bbox를 기반으로 Poisson 블렌딩의 중심점(center)을 계산한다.
        
        cv2.seamlessClone의 center는 source 이미지의 중심이 target에 배치될 위치.
        roi_bbox의 중심점을 사용하되, target 경계를 벗어나지 않도록 클리핑한다.
        
        Args:
            roi_bbox: (x1, y1, x2, y2) 원본 이미지 좌표계
            target_shape: (height, width) target 이미지 크기
            
        Returns:
            (center_x, center_y) 정수 튜플
        """
        x1, y1, x2, y2 = roi_bbox
        target_h, target_w = target_shape
        
        # ROI 중심점
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # ROI 크기 (source 크기)
        roi_w = x2 - x1
        roi_h = y2 - y1
        half_w = roi_w // 2
        half_h = roi_h // 2
        
        # center에서 source의 반쪽 크기만큼 확장한 영역이
        # target 경계 안에 들어오도록 클리핑
        center_x = max(half_w, min(target_w - half_w, center_x))
        center_y = max(half_h, min(target_h - half_h, center_y))
        
        return (int(center_x), int(center_y))
    
    # ──────────────────────────────────────────────
    # 블렌딩 영역 검증
    # ──────────────────────────────────────────────
    
    def validate_blend_region(
        self,
        mask: np.ndarray,
        center: Tuple[int, int],
        target_shape: Tuple[int, int],
    ) -> bool:
        """
        블렌딩 영역이 target 이미지 경계 내에 있는지 검증한다.
        
        Args:
            mask: source 마스크 (H, W)
            center: (center_x, center_y) 붙여넣기 중심점
            target_shape: (height, width) target 이미지 크기
            
        Returns:
            True이면 유효한 블렌딩 영역
        """
        mask_h, mask_w = mask.shape[:2]
        target_h, target_w = target_shape
        center_x, center_y = center
        
        # source 영역의 경계 계산
        left = center_x - mask_w // 2
        top = center_y - mask_h // 2
        right = left + mask_w
        bottom = top + mask_h
        
        # target 경계 내에 있는지 확인
        if left < 0 or top < 0 or right > target_w or bottom > target_h:
            return False
        
        # 마스크에 실제 결함 픽셀이 있는지 확인
        if np.count_nonzero(mask) == 0:
            return False
        
        return True
    
    # ──────────────────────────────────────────────
    # Poisson 블렌딩 실행
    # ──────────────────────────────────────────────
    
    def poisson_blend(
        self,
        source: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        center: Tuple[int, int],
    ) -> Tuple[np.ndarray, str]:
        """
        cv2.seamlessClone()으로 Poisson 블렌딩을 실행한다.
        실패 시 단순 알파 블렌딩으로 폴백한다.
        
        Args:
            source: ROI 크기의 3채널 이미지 (H, W, 3)
            mask: ROI 크기의 이진 마스크 (H, W), 값 0/255
            target: 1600x256x3 배경 이미지 (수정됨)
            center: (center_x, center_y) 붙여넣기 중심점
            
        Returns:
            (composited_image, method) 튜플
            - composited_image: 합성된 이미지
            - method: "poisson_normal_clone", "poisson_mixed_clone", 또는 "alpha_fallback"
        """
        # 3채널 확인
        if len(source.shape) == 2:
            source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        if len(target.shape) == 2:
            target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
        
        method_name = (
            "poisson_normal_clone" if self.blend_mode == cv2.NORMAL_CLONE
            else "poisson_mixed_clone"
        )
        
        try:
            result = cv2.seamlessClone(
                source, target, mask, center, self.blend_mode
            )
            return result, method_name
        except cv2.error as e:
            logger.warning(f"seamlessClone 실패, 알파 블렌딩으로 폴백: {e}")
            return self._alpha_blend_fallback(source, mask, target, center), "alpha_fallback"
    
    def _alpha_blend_fallback(
        self,
        source: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        center: Tuple[int, int],
    ) -> np.ndarray:
        """
        Poisson 블렌딩 실패 시 단순 알파 블렌딩 폴백.
        
        마스크 경계를 가우시안 블러로 소프트하게 만들어
        경계 전환을 부드럽게 한다.
        """
        result = target.copy()
        src_h, src_w = source.shape[:2]
        center_x, center_y = center
        
        # source 이미지가 배치될 영역 계산
        x1 = center_x - src_w // 2
        y1 = center_y - src_h // 2
        x2 = x1 + src_w
        y2 = y1 + src_h
        
        # target 경계 클리핑
        tgt_h, tgt_w = target.shape[:2]
        src_x1 = max(0, -x1)
        src_y1 = max(0, -y1)
        src_x2 = src_w - max(0, x2 - tgt_w)
        src_y2 = src_h - max(0, y2 - tgt_h)
        
        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(tgt_w, x2)
        dst_y2 = min(tgt_h, y2)
        
        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            return result
        
        # 마스크를 소프트하게 (가우시안 블러)
        soft_mask = mask[src_y1:src_y2, src_x1:src_x2].astype(np.float32) / 255.0
        if soft_mask.size > 0:
            soft_mask = cv2.GaussianBlur(soft_mask, (21, 21), 5.0)
        
        # 알파 블렌딩
        alpha = soft_mask[:, :, np.newaxis]  # (H, W, 1)
        src_region = source[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
        dst_region = result[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
        
        blended = (alpha * src_region + (1.0 - alpha) * dst_region).astype(np.uint8)
        result[dst_y1:dst_y2, dst_x1:dst_x2] = blended
        
        return result
    
    # ──────────────────────────────────────────────
    # 전체 크기 마스크 생성
    # ──────────────────────────────────────────────
    
    def generate_full_mask(
        self,
        defect_mask_roi: np.ndarray,
        roi_bbox: Tuple[int, int, int, int],
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        ROI 크기의 결함 마스크를 1600x256 전체 크기 마스크로 변환한다.
        
        확장(dilated) 마스크가 아닌 원본 결함 마스크를 사용한다.
        (확장 영역은 Poisson 블렌딩 보조용이므로 라벨에는 포함하지 않음)
        
        Args:
            defect_mask_roi: ROI 크기 이진 마스크 (H, W), 비확장 원본
            roi_bbox: (x1, y1, x2, y2) 원본 이미지 좌표계
            target_shape: (height, width) 전체 이미지 크기
            
        Returns:
            전체 크기 이진 마스크 (target_h, target_w)
        """
        target_h, target_w = target_shape
        full_mask = np.zeros((target_h, target_w), dtype=np.uint8)
        
        x1, y1, x2, y2 = roi_bbox
        roi_h = y2 - y1
        roi_w = x2 - x1
        mask_h, mask_w = defect_mask_roi.shape[:2]
        
        # ROI 크기와 마스크 크기가 다르면 리사이즈
        if mask_h != roi_h or mask_w != roi_w:
            defect_mask_resized = cv2.resize(
                defect_mask_roi, (roi_w, roi_h),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            defect_mask_resized = defect_mask_roi
        
        # target 경계 내로 클리핑
        paste_x1 = max(0, x1)
        paste_y1 = max(0, y1)
        paste_x2 = min(target_w, x2)
        paste_y2 = min(target_h, y2)
        
        # 마스크 내 대응 영역
        src_x1 = paste_x1 - x1
        src_y1 = paste_y1 - y1
        src_x2 = src_x1 + (paste_x2 - paste_x1)
        src_y2 = src_y1 + (paste_y2 - paste_y1)
        
        full_mask[paste_y1:paste_y2, paste_x1:paste_x2] = \
            defect_mask_resized[src_y1:src_y2, src_x1:src_x2]
        
        return full_mask
    
    # ──────────────────────────────────────────────
    # YOLO bbox 계산
    # ──────────────────────────────────────────────
    
    def compute_yolo_bboxes(
        self,
        full_mask: np.ndarray,
        class_id: int,
    ) -> Tuple[List[List[float]], List[int]]:
        """
        전체 크기 마스크에서 정규화 YOLO bbox를 추출한다.
        
        Args:
            full_mask: 전체 크기 이진 마스크 (H, W)
            class_id: 결함 클래스 ID (0-indexed)
            
        Returns:
            (bboxes, labels) 튜플
            - bboxes: [[cx, cy, w, h], ...] 정규화 [0, 1]
            - labels: [class_id, ...] per-bbox
        """
        h, w = full_mask.shape[:2]
        
        contours, _ = cv2.findContours(
            full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        bboxes = []
        labels = []
        
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw * bh < self.min_defect_area:
                continue
            
            cx = (bx + bw / 2.0) / w
            cy = (by + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            
            bboxes.append([cx, cy, nw, nh])
            labels.append(class_id)
        
        return bboxes, labels
    
    # ──────────────────────────────────────────────
    # 단일 이미지 합성 (전체 파이프라인)
    # ──────────────────────────────────────────────
    
    def compose_single(
        self,
        generated_image: np.ndarray,
        hint_image: np.ndarray,
        clean_background: np.ndarray,
        roi_bbox: Tuple[int, int, int, int],
        class_id: int,
    ) -> CompositionResult:
        """
        단일 이미지의 전체 합성 파이프라인을 실행한다.
        
        처리 흐름:
          1. 힌트에서 마스크 추출
          2. 512→256 다운스케일
          3. 마스크 확장
          4. 붙여넣기 위치 계산
          5. Poisson 블렌딩
          6. 전체 크기 마스크 + YOLO bbox
        
        Args:
            generated_image: 512x512x3 ControlNet 생성 이미지 (BGR)
            hint_image: 힌트 이미지 (BGR, 512x512 또는 원본 크기)
            clean_background: 1600x256x3 결함 없는 배경 이미지 (BGR)
            roi_bbox: (x1, y1, x2, y2) 원본 이미지 좌표계
            class_id: 결함 클래스 ID (0-indexed)
            
        Returns:
            CompositionResult
        """
        target_h, target_w = clean_background.shape[:2]
        x1, y1, x2, y2 = roi_bbox
        roi_h = y2 - y1
        roi_w = x2 - x1
        
        # Step 1: 힌트에서 마스크 추출
        try:
            mask_512 = self.extract_mask_from_hint(hint_image)
        except ValueError as e:
            return CompositionResult(
                composited_image=clean_background.copy(),
                full_mask=np.zeros((target_h, target_w), dtype=np.uint8),
                bboxes=[], labels=[], success=False,
                blend_method="none", message=f"마스크 추출 실패: {e}"
            )
        
        # 마스크에 결함 픽셀이 없으면 스킵
        if np.count_nonzero(mask_512) == 0:
            return CompositionResult(
                composited_image=clean_background.copy(),
                full_mask=np.zeros((target_h, target_w), dtype=np.uint8),
                bboxes=[], labels=[], success=False,
                blend_method="none", message="마스크에 결함 픽셀 없음"
            )
        
        # Step 2: 512→ROI 크기 다운스케일
        image_roi, mask_roi = self.downscale_to_roi(
            generated_image, mask_512,
            target_h=roi_h, target_w=roi_w
        )
        
        # 다운스케일 후 마스크 검증: INTER_NEAREST로 축소 시
        # 아주 작은 결함이 사라질 수 있음. seamlessClone에 빈 마스크가
        # 들어가면 crash하므로 여기서 조기 리턴.
        if np.count_nonzero(mask_roi) == 0:
            return CompositionResult(
                composited_image=clean_background.copy(),
                full_mask=np.zeros((target_h, target_w), dtype=np.uint8),
                bboxes=[], labels=[], success=False,
                blend_method="none",
                message="다운스케일 후 마스크에 결함 픽셀 없음 (결함이 너무 작음)"
            )
        
        # Step 3: 마스크 확장 (블렌딩용)
        mask_dilated = self.dilate_mask(mask_roi)
        
        # Step 3.5: seamlessClone 크기 제약 처리
        # seamlessClone은 source 전체 영역(center ± source_size/2)이 target
        # 경계 안에 있어야 한다. Severstal에서 ROI 높이(256) == target 높이(256)
        # 이면 수직 방향 배치가 불가능하므로, source를 상하좌우 1px씩 crop한다.
        # 이 crop은 dilate_mask의 edge margin(가장자리 0 처리)과 결합되어
        # 결함 영역 자체에는 영향을 주지 않는다.
        blend_src = image_roi
        blend_mask = mask_dilated
        src_h, src_w = blend_mask.shape[:2]
        
        needs_crop = (src_h >= target_h) or (src_w >= target_w)
        if needs_crop:
            crop_y = 1 if src_h >= target_h else 0
            crop_x = 1 if src_w >= target_w else 0
            blend_src = blend_src[crop_y:src_h - crop_y, crop_x:src_w - crop_x]
            blend_mask = blend_mask[crop_y:src_h - crop_y, crop_x:src_w - crop_x]
            logger.debug(
                f"seamlessClone 크기 제약: source {src_h}x{src_w} → "
                f"{blend_src.shape[0]}x{blend_src.shape[1]} (crop_y={crop_y}, crop_x={crop_x})"
            )
        
        # Step 4: 붙여넣기 중심점 계산
        center = self.compute_paste_center(roi_bbox, (target_h, target_w))
        
        # Step 5: 블렌딩 영역 검증
        if not self.validate_blend_region(blend_mask, center, (target_h, target_w)):
            # 검증 실패 시 center 재조정 시도
            logger.warning(
                f"블렌딩 영역 검증 실패 (roi_bbox={roi_bbox}, center={center}), "
                f"center 재조정 시도"
            )
            center = self._adjust_center_for_boundary(
                blend_mask, center, (target_h, target_w)
            )
            if not self.validate_blend_region(blend_mask, center, (target_h, target_w)):
                return CompositionResult(
                    composited_image=clean_background.copy(),
                    full_mask=np.zeros((target_h, target_w), dtype=np.uint8),
                    bboxes=[], labels=[], success=False,
                    blend_method="none",
                    message=f"블렌딩 영역 검증 실패: roi_bbox={roi_bbox}"
                )
        
        # Step 6: Poisson 블렌딩
        composited, blend_method = self.poisson_blend(
            blend_src, blend_mask, clean_background.copy(), center
        )
        
        # Step 7: 전체 크기 마스크 생성 (비확장 원본 마스크 사용)
        full_mask = self.generate_full_mask(
            mask_roi, roi_bbox, (target_h, target_w)
        )
        
        # Step 8: YOLO bbox 계산
        bboxes, labels = self.compute_yolo_bboxes(full_mask, class_id)
        
        if len(bboxes) == 0:
            return CompositionResult(
                composited_image=composited, full_mask=full_mask,
                bboxes=[], labels=[], success=False,
                blend_method=blend_method,
                message="유효한 bbox를 추출할 수 없음"
            )
        
        return CompositionResult(
            composited_image=composited,
            full_mask=full_mask,
            bboxes=bboxes,
            labels=labels,
            success=True,
            blend_method=blend_method,
            message="합성 성공"
        )
    
    # ──────────────────────────────────────────────
    # 파일 경로 기반 합성 (편의 메서드)
    # ──────────────────────────────────────────────
    
    def compose_from_paths(
        self,
        generated_path: str,
        hint_path: str,
        clean_bg_path: str,
        roi_bbox: Tuple[int, int, int, int],
        class_id: int,
    ) -> CompositionResult:
        """
        파일 경로에서 이미지를 로드하여 합성을 실행한다.
        
        Args:
            generated_path: 생성 이미지 경로 (512x512)
            hint_path: 힌트 이미지 경로
            clean_bg_path: 결함 없는 배경 이미지 경로 (1600x256)
            roi_bbox: (x1, y1, x2, y2) 원본 이미지 좌표계
            class_id: 결함 클래스 ID (0-indexed)
            
        Returns:
            CompositionResult
        """
        # 이미지 로드
        generated = cv2.imread(str(generated_path), cv2.IMREAD_COLOR)
        if generated is None:
            return CompositionResult(
                composited_image=None, full_mask=None,
                bboxes=[], labels=[], success=False,
                blend_method="none",
                message=f"생성 이미지 로드 실패: {generated_path}"
            )
        
        hint = cv2.imread(str(hint_path), cv2.IMREAD_COLOR)
        if hint is None:
            return CompositionResult(
                composited_image=None, full_mask=None,
                bboxes=[], labels=[], success=False,
                blend_method="none",
                message=f"힌트 이미지 로드 실패: {hint_path}"
            )
        
        clean_bg = cv2.imread(str(clean_bg_path), cv2.IMREAD_COLOR)
        if clean_bg is None:
            return CompositionResult(
                composited_image=None, full_mask=None,
                bboxes=[], labels=[], success=False,
                blend_method="none",
                message=f"배경 이미지 로드 실패: {clean_bg_path}"
            )
        
        return self.compose_single(
            generated, hint, clean_bg, roi_bbox, class_id
        )
    
    # ──────────────────────────────────────────────
    # 내부 유틸리티
    # ──────────────────────────────────────────────
    
    def _adjust_center_for_boundary(
        self,
        mask: np.ndarray,
        center: Tuple[int, int],
        target_shape: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        블렌딩 영역이 target 경계를 벗어날 때 center를 재조정한다.
        """
        mask_h, mask_w = mask.shape[:2]
        target_h, target_w = target_shape
        
        half_w = mask_w // 2
        half_h = mask_h // 2
        
        cx = max(half_w + 1, min(target_w - half_w - 1, center[0]))
        cy = max(half_h + 1, min(target_h - half_h - 1, center[1]))
        
        return (int(cx), int(cy))
