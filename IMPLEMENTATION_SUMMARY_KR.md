# ROI 추출 연구 구현 요약

PROJECT(roi).md의 통계적 지표 기반 ROI 추출 파이프라인을 완전히 구현했습니다.

## 구현 완료 모듈

### 1. 유틸리티 모듈
- **src/utils/rle_utils.py**: RLE 인코딩/디코딩 함수

### 2. 분석 모듈
- **src/analysis/defect_characterization.py**: 결함 특성 분석 (4대 지표)
  - Linearity (선형성)
  - Solidity (견고성)
  - Extent (범위)
  - Aspect Ratio (종횡비)
  - 결함 서브타입 분류 (linear_scratch, compact_blob, elongated, irregular, general)

- **src/analysis/background_characterization.py**: 배경 특성 분석 (그리드 기반)
  - 분산 분석 (smooth vs textured)
  - 엣지 방향 분석 (Sobel) → vertical_stripe, horizontal_stripe
  - 주파수 분석 (FFT) → complex_pattern
  - 안정성 점수 계산

- **src/analysis/roi_suitability.py**: ROI 적합도 평가
  - 결함-배경 매칭 규칙
  - 적합도 점수 계산 (matching + continuity + stability)
  - ROI 위치 최적화 (배경 연속성 극대화)
  - ControlNet 프롬프트 생성

### 3. 전처리 모듈
- **src/preprocessing/roi_extraction.py**: 통합 ROI 추출 파이프라인
  - 전체 이미지 처리
  - ROI 패치 추출 및 저장
  - 메타데이터 생성

### 4. 실행 스크립트
- **scripts/extract_rois.py**: 메인 실행 스크립트
  - 명령줄 인터페이스
  - 진행 상황 표시
  - 통계 생성 및 저장

## 파이프라인 워크플로우

```
입력: 원본 이미지 + RLE 마스크 (train.csv)
  ↓
[1] 배경 분석
  - 이미지를 64×64 그리드로 분할
  - 각 패치의 배경 타입 분류 (smooth, textured, stripe, etc.)
  - 안정성 맵 생성
  ↓
[2] 결함 분석
  - RLE 디코딩하여 마스크 생성
  - Connected Component Analysis
  - 4대 지표 계산 (linearity, solidity, extent, aspect_ratio)
  - 결함 서브타입 분류
  ↓
[3] ROI 적합도 평가
  - 결함 서브타입 + 배경 타입 매칭 점수
  - 배경 연속성 점수
  - 전체 적합도 점수 (0-1)
  ↓
[4] ROI 위치 최적화
  - 결함 중심 기준 512×512 패치
  - 배경 경계면 감지 시 패치 이동 (±32px)
  - 배경 연속성 최대화
  ↓
[5] 데이터 패키징
  - ROI 이미지 패치 저장
  - ROI 마스크 패치 저장
  - 메타데이터 CSV 생성
  - ControlNet 프롬프트 생성
  ↓
출력: ROI 패치 + 메타데이터 CSV + 통계
```

## 사용 방법

### 기본 실행 (테스트 모드)
```bash
python scripts/extract_rois.py --max_images 10
```

### 전체 데이터셋 처리
```bash
python scripts/extract_rois.py
```

### 메타데이터만 생성 (패치 저장 안 함)
```bash
python scripts/extract_rois.py --no_save_patches --max_images 100
```

### 커스텀 파라미터
```bash
python scripts/extract_rois.py \
    --roi_size 512 \
    --grid_size 64 \
    --min_suitability 0.5 \
    --max_images 1000
```

## 출력 데이터

### 디렉토리 구조
```
data/processed/roi_patches/
├── images/               # ROI 이미지 패치 (PNG)
│   ├── 0002cc93b_class1_region0.png
│   ├── 0002cc93b_class2_region0.png
│   └── ...
├── masks/                # ROI 마스크 패치 (PNG)
│   ├── 0002cc93b_class1_region0.png
│   ├── 0002cc93b_class2_region0.png
│   └── ...
├── roi_metadata.csv      # 전체 메타데이터
└── statistics.txt        # 통계 요약
```

### 메타데이터 컬럼
- **식별자**: image_id, class_id, region_id
- **위치**: roi_bbox, defect_bbox, centroid
- **결함 지표**: linearity, solidity, extent, aspect_ratio
- **분류**: defect_subtype, background_type
- **점수**: suitability_score, matching_score, continuity_score, stability_score
- **추천**: recommendation (suitable/acceptable/unsuitable)
- **프롬프트**: prompt (ControlNet 학습용)
- **파일 경로**: roi_image_path, roi_mask_path

## 핵심 기여

### 1. 결함-배경 매칭 규칙
PROJECT(roi).md의 핵심 아이디어를 코드로 구현:
- Linear scratch → Vertical/Horizontal stripe (매칭 점수 1.0)
- Compact blob → Smooth surface (매칭 점수 1.0)
- Irregular → Complex pattern (매칭 점수 1.0)

### 2. 배경 연속성 확보
ROI 윈도우를 미세 조정하여 배경 경계면 회피

### 3. 데이터 품질 필터링
적합도 점수 기반으로 unsuitable ROI 제외 (min_suitability 임계값)

### 4. ControlNet 통합
각 ROI에 대해 자동으로 프롬프트 생성:
```
"a linear scratch defect on vertical striped metal surface, class 1, industrial steel defect detection"
```

## 연구 질문 해결

✅ **결함을 어떻게 특성화할 것인가?**
→ 4대 기하학적 지표 (Linearity, Solidity, Extent, Aspect Ratio)

✅ **배경의 적합성을 어떻게 정량화할 것인가?**
→ 그리드 기반 분류 + 안정성 점수

✅ **결함과 배경을 어떻게 매칭할 것인가?**
→ 결함 서브타입별 배경 타입 매칭 규칙 테이블

✅ **ROI 위치를 어떻게 최적화할 것인가?**
→ 배경 연속성 극대화를 목표로 ±32px 탐색

✅ **ControlNet 학습을 위해 데이터를 어떻게 패키징할 것인가?**
→ [이미지 패치] + [마스크 패치] + [메타데이터] + [프롬프트]

## 다음 단계

1. **시각화 도구**: ROI 선택 결과를 시각적으로 검증
2. **ControlNet 학습**: 추출된 ROI로 조건부 생성 모델 학습
3. **증강 파이프라인**: 적합도 점수 기반 합성 결함 생성
4. **성능 검증**: 필터링 유무에 따른 모델 성능 비교

## 참고 문서

- **PROJECT(roi).md**: 연구 방법론 전체 (한국어)
- **README_ROI.md**: 구현 상세 설명 (영어)
- **requirements.txt**: 필요 라이브러리

---
구현 완료: 2026년 2월 9일  
연구 프레임워크: PROJECT(roi).md
