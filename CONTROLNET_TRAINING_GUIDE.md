# ControlNet Training Guide for Steel Defect Augmentation

## 프로젝트 개요

본 프로젝트는 **Context-Aware Steel Defect Augmentation (CASDA)** 시스템을 구현합니다. 이 시스템은 철강 표면 결함 이미지의 물리적 특성을 고려한 고품질 증강 데이터를 생성합니다.

### 핵심 특징

1. **4대 지표 기반 결함 분석**
   - Linearity (직선성): 결함의 선형 정도
   - Solidity (치밀도): 결함 영역의 밀집도
   - Extent (분산도): 바운딩 박스 대비 결함 면적
   - Aspect Ratio (종횡비): 결함의 방향성

2. **배경 컨텍스트 분석**
   - Smooth: 균일한 평면
   - Textured: 거친 질감
   - Vertical Stripe: 수직 패턴 (압연 흔적)
   - Horizontal Stripe: 수평 패턴
   - Complex Pattern: 복합 패턴

3. **Multi-Channel Hint 이미지**
   - Red Channel: 결함 마스크 (4대 지표 반영)
   - Green Channel: 배경 구조선 (에지 정보)
   - Blue Channel: 배경 질감 (로컬 variance)

4. **Hybrid Prompt Engineering**
   - 결함 특성 + 배경 타입 + 표면 상태를 통합한 텍스트 프롬프트

## 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│                   1. ROI 추출 단계                           │
├─────────────────────────────────────────────────────────────┤
│ Input:  train_images/ + train.csv                           │
│ Script: scripts/extract_rois.py                             │
│ Output: data/processed/roi_patches/                         │
│         ├── images/                                         │
│         ├── masks/                                          │
│         └── roi_metadata.csv                                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              2. ControlNet 데이터 준비 단계                  │
├─────────────────────────────────────────────────────────────┤
│ Input:  data/processed/roi_patches/roi_metadata.csv         │
│ Script: scripts/prepare_controlnet_data.py                  │
│ Output: data/processed/controlnet_dataset/                  │
│         ├── hints/           (multi-channel hint images)    │
│         ├── train.jsonl      (training index)               │
│         ├── metadata.json    (dataset metadata)             │
│         └── validation/      (quality reports)              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   3. ControlNet 학습 단계                    │
├─────────────────────────────────────────────────────────────┤
│ Input:  data/processed/controlnet_dataset/train.jsonl       │
│ Script: scripts/train_controlnet.py                         │
│ Output: outputs/controlnet_training/                        │
│         ├── best.pth         (best model)                   │
│         ├── last.pth         (latest checkpoint)            │
│         └── checkpoint_epoch_*.pth                          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  4. 증강 데이터 생성 단계                     │
├─────────────────────────────────────────────────────────────┤
│ Input:  trained ControlNet model + new backgrounds          │
│ Output: synthetic defect images for training augmentation   │
└─────────────────────────────────────────────────────────────┘
```

## 실행 방법

### 사전 준비

1. **환경 설정**
```bash
# Python 환경 생성 (권장: Python 3.8+)
conda create -n steel-defect python=3.8
conda activate steel-defect

# 의존성 설치
pip install -r requirements.txt
```

2. **데이터 준비**
- `train_images/`: 원본 학습 이미지
- `train.csv`: RLE 형식의 마스크 어노테이션

### Step 1: ROI 추출

결함 영역과 배경을 분석하여 최적의 ROI를 추출합니다.

```bash
# 전체 데이터셋 처리
python scripts/extract_rois.py

# 테스트용 (처음 100개 이미지만)
python scripts/extract_rois.py --max_images 100

# 커스텀 설정
python scripts/extract_rois.py \
    --roi_size 512 \
    --grid_size 64 \
    --min_suitability 0.5 \
    --output_dir data/processed/roi_patches
```

**출력:**
- `data/processed/roi_patches/roi_metadata.csv`: ROI 메타데이터
- `data/processed/roi_patches/images/`: ROI 이미지 패치
- `data/processed/roi_patches/masks/`: ROI 마스크 패치
- `data/processed/roi_patches/statistics.txt`: 추출 통계

**주요 파라미터:**
- `--roi_size`: ROI 패치 크기 (기본값: 512)
- `--grid_size`: 배경 분석 그리드 크기 (기본값: 64)
- `--min_suitability`: 최소 적합도 점수 (기본값: 0.5)
- `--max_images`: 처리할 이미지 수 제한 (테스트용)

### Step 2: ControlNet 데이터 준비

Multi-channel hint 이미지와 프롬프트를 생성합니다.

```bash
# 기본 실행
python scripts/prepare_controlnet_data.py \
    --roi_metadata data/processed/roi_patches/roi_metadata.csv

# 프롬프트 스타일 선택
python scripts/prepare_controlnet_data.py \
    --roi_metadata data/processed/roi_patches/roi_metadata.csv \
    --prompt_style detailed \
    --output_dir data/processed/controlnet_dataset

# 검증 단계 스킵 (빠른 테스트)
python scripts/prepare_controlnet_data.py \
    --roi_metadata data/processed/roi_patches/roi_metadata.csv \
    --skip_validation

# Hint 생성 스킵 (프롬프트만 생성)
python scripts/prepare_controlnet_data.py \
    --roi_metadata data/processed/roi_patches/roi_metadata.csv \
    --skip_hints
```

**출력:**
- `data/processed/controlnet_dataset/train.jsonl`: 학습 데이터 인덱스
- `data/processed/controlnet_dataset/hints/`: Multi-channel hint 이미지
- `data/processed/controlnet_dataset/metadata.json`: 전체 데이터셋 메타데이터
- `data/processed/controlnet_dataset/validation/`: 품질 검증 리포트

**주요 파라미터:**
- `--prompt_style`: 프롬프트 스타일 (simple/detailed/technical)
- `--skip_validation`: 데이터셋 검증 스킵
- `--skip_hints`: Hint 이미지 생성 스킵
- `--max_samples`: 처리할 샘플 수 제한

**Prompt 스타일:**

1. **Simple:**
```
"a linear scratch defect on vertical striped metal surface, class 3"
```

2. **Detailed:**
```
"a high-linearity elongated scratch on vertical striped metal surface 
with directional texture (pristine condition), steel defect class 3"
```

3. **Technical:**
```
"Industrial steel defect: highly linear, very elongated defect (class 3) 
on vertical striped metal surface, vertical line pattern, 
background stability 0.85, match quality 0.92"
```

### Step 3: ControlNet 학습

Multi-channel hint를 조건으로 하는 ControlNet을 학습합니다.

```bash
# 기본 학습
python scripts/train_controlnet.py \
    --data_dir data/processed/controlnet_dataset

# 커스텀 설정
python scripts/train_controlnet.py \
    --data_dir data/processed/controlnet_dataset \
    --batch_size 4 \
    --num_epochs 100 \
    --lr 1e-4 \
    --image_size 512 \
    --output_dir outputs/controlnet_training

# 체크포인트에서 재개
python scripts/train_controlnet.py \
    --data_dir data/processed/controlnet_dataset \
    --resume outputs/controlnet_training/last.pth

# CPU 사용 (GPU 없는 경우)
python scripts/train_controlnet.py \
    --data_dir data/processed/controlnet_dataset \
    --device cpu
```

**출력:**
- `outputs/controlnet_training/best.pth`: 최고 성능 모델
- `outputs/controlnet_training/last.pth`: 최신 체크포인트
- `outputs/controlnet_training/checkpoint_epoch_*.pth`: 주기적 체크포인트

**주요 파라미터:**
- `--batch_size`: 배치 크기 (GPU 메모리에 따라 조정)
- `--num_epochs`: 학습 에포크 수
- `--lr`: 학습률
- `--image_size`: 학습 이미지 크기
- `--save_every`: 체크포인트 저장 주기 (에포크 단위)
- `--resume`: 재개할 체크포인트 경로

**학습 모니터링:**
- 콘솔에서 실시간 loss 확인
- 주기적으로 체크포인트 저장
- 최고 성능 모델 자동 저장

## 구현된 모듈 설명

### 1. 결함 분석 모듈 (`src/analysis/defect_characterization.py`)

**DefectCharacterizer 클래스:**
- `compute_linearity()`: 결함의 선형성 계산 (eigenvalue 기반)
- `compute_solidity()`: 결함의 치밀도 계산
- `compute_extent()`: 결함의 분산도 계산
- `compute_aspect_ratio()`: 결함의 종횡비 계산
- `analyze_defect_region()`: 단일 결함 영역 분석
- `classify_defect_subtype()`: 결함 서브타입 분류

**결함 서브타입:**
- `linear_scratch`: 높은 선형성 + 높은 종횡비
- `elongated`: 높은 종횡비 + 중간 선형성
- `compact_blob`: 낮은 종횡비 + 높은 치밀도
- `irregular`: 낮은 치밀도
- `general`: 일반 결함

### 2. 배경 분석 모듈 (`src/analysis/background_characterization.py`)

**BackgroundAnalyzer 클래스:**
- `compute_variance()`: 패치의 variance 계산
- `compute_edge_directions()`: Sobel 필터를 이용한 방향성 에지 계산
- `compute_frequency_spectrum()`: FFT 기반 주파수 특성 계산
- `classify_patch()`: 패치의 배경 타입 분류
- `analyze_image()`: 그리드 기반 전체 이미지 분석
- `check_continuity()`: 배경의 연속성 검사

**배경 타입:**
- `smooth`: 낮은 variance, 평평한 표면
- `textured`: 높은 variance, 거친 표면
- `vertical_stripe`: 강한 수직 에지 패턴
- `horizontal_stripe`: 강한 수평 에지 패턴
- `complex_pattern`: 다방향 에지 패턴

### 3. ROI 적합성 평가 모듈 (`src/analysis/roi_suitability.py`)

**ROISuitabilityEvaluator 클래스:**
- `compute_matching_score()`: 결함-배경 매칭 점수 계산
- `evaluate_roi_suitability()`: ROI 적합성 종합 평가
- `optimize_roi_position()`: ROI 위치 최적화 (배경 연속성 최대화)

**적합성 점수 구성:**
```python
suitability_score = (
    0.5 * matching_score +    # 결함-배경 매칭 (가장 중요)
    0.3 * continuity_score +  # 배경 연속성
    0.2 * stability_score     # 배경 안정성
)
```

**매칭 규칙 예시:**
- Linear scratch + Vertical stripe = 1.0 (완벽한 매칭)
- Compact blob + Smooth = 1.0 (완벽한 매칭)
- Linear scratch + Complex pattern = 0.3 (부적합)

### 4. Hint 생성 모듈 (`src/preprocessing/hint_generator.py`)

**HintImageGenerator 클래스:**
- `generate_red_channel()`: 결함 마스크 (4대 지표 반영)
  - 높은 선형성 → Skeleton 강조
  - 높은 치밀도 → 채워진 영역
  - 낮은 치밀도 → 에지 강조
  
- `generate_green_channel()`: 배경 구조선
  - Vertical stripe → 수직 에지 강조
  - Horizontal stripe → 수평 에지 강조
  - Complex pattern → 전체 에지
  
- `generate_blue_channel()`: 배경 질감
  - Smooth → 낮은 값
  - Textured/Complex → 로컬 variance 높음

### 5. 프롬프트 생성 모듈 (`src/preprocessing/prompt_generator.py`)

**PromptGenerator 클래스:**
- `generate_simple_prompt()`: 간단한 프롬프트
- `generate_detailed_prompt()`: 상세한 프롬프트
- `generate_technical_prompt()`: 기술적 프롬프트
- `generate_negative_prompt()`: 부정 프롬프트

## 데이터 포맷

### train.jsonl 포맷

```json
{
  "source": "roi_patches/images/0002cc93b_class1_region0.png",
  "target": "roi_patches/images/0002cc93b_class1_region0.png",
  "prompt": "a linear scratch on vertical striped metal surface...",
  "hint": "controlnet_dataset/hints/0002cc93b_class1_region0_hint.png",
  "negative_prompt": "blurry, low quality, artifacts..."
}
```

### roi_metadata.csv 주요 필드

```
image_id,class_id,region_id,
linearity,solidity,extent,aspect_ratio,
defect_subtype,background_type,
suitability_score,matching_score,continuity_score,
recommendation,prompt
```

## 품질 검증

### 자동 검증 항목

1. **분포 균형 검사**
   - 클래스별 샘플 수
   - 결함 서브타입별 분포
   - 배경 타입별 분포

2. **적합성 점수 검증**
   - 평균 suitability score
   - 평균 matching score
   - 평균 continuity score

3. **시각적 검증**
   - Hint 이미지 샘플 저장
   - 채널별 분리 시각화

### 수동 검증 권장사항

1. **Hint 이미지 확인**
   - Red channel: 결함 형태가 올바르게 표현되었는지
   - Green channel: 배경 구조선이 명확한지
   - Blue channel: 질감 정보가 적절한지

2. **프롬프트 확인**
   - 결함과 배경 설명이 정확한지
   - 프롬프트가 이미지와 일치하는지

## 트러블슈팅

### 1. ROI 추출 시 샘플 수가 적음
**원인:** min_suitability 임계값이 너무 높음
**해결:**
```bash
python scripts/extract_rois.py --min_suitability 0.3
```

### 2. GPU 메모리 부족
**원인:** 배치 크기가 너무 큼
**해결:**
```bash
python scripts/train_controlnet.py --batch_size 2 --image_size 256
```

### 3. 학습이 수렴하지 않음
**원인:** 학습률이 부적절하거나 데이터 품질 문제
**해결:**
- 학습률 조정: `--lr 5e-5` 또는 `--lr 5e-4`
- 데이터 검증 리포트 확인
- Hint 이미지 시각적 확인

### 4. Python 실행 오류
**원인:** 의존성 패키지 미설치
**해결:**
```bash
pip install -r requirements.txt
# 또는 특정 패키지만
pip install numpy pandas opencv-python scikit-image torch torchvision
```

## 고급 설정

### 커스텀 배경 분석 파라미터

```python
background_analyzer = BackgroundAnalyzer(
    grid_size=128,              # 더 세밀한 그리드
    variance_threshold=150.0,   # Smooth 판정 기준 조정
    edge_threshold=0.4          # Stripe 판정 기준 조정
)
```

### 커스텀 매칭 규칙

`src/analysis/roi_suitability.py`의 `MATCHING_RULES` 딕셔너리를 수정하여 결함-배경 매칭 점수를 조정할 수 있습니다.

### ControlNet 아키텍처 교체

현재 구현은 데모용 단순 ControlNet입니다. 프로덕션 환경에서는 공식 ControlNet 구현을 사용하는 것을 권장합니다:

```python
# 공식 ControlNet 사용 예시
from cldm.cldm import ControlLDM
from ldm.models.diffusion.ddim import DDIMSampler

# Stable Diffusion 기반 ControlNet 로드
model = ControlLDM(...)
```

## 성능 최적화

1. **데이터 로딩 최적화**
   - `num_workers` 조정 (CPU 코어 수에 맞춤)
   - `pin_memory=True` (GPU 사용 시)

2. **Mixed Precision Training**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

3. **Gradient Accumulation**
   - 메모리 부족 시 배치를 나눠서 처리

## 참고 자료

- [ControlNet 논문](https://arxiv.org/abs/2302.05543)
- [ControlNet GitHub](https://github.com/lllyasviel/ControlNet)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- Severstal Steel Defect Detection: [Kaggle Competition](https://www.kaggle.com/c/severstal-steel-defect-detection)

## 라이선스 및 인용

본 프로젝트는 연구 및 교육 목적으로 사용할 수 있습니다.

```bibtex
@article{controlnet2023,
  title={Adding Conditional Control to Text-to-Image Diffusion Models},
  author={Zhang, Lvmin and Rao, Anyi and Agrawala, Maneesh},
  journal={arXiv preprint arXiv:2302.05543},
  year={2023}
}
```
