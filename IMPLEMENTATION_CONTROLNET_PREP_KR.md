# ControlNet 학습 데이터 준비 구현 완료

PROJECT(prepare_control).md에 명시된 **데이터 패키징 및 학습 진입 단계**를 완전히 구현했습니다.

## 구현 개요

이 단계는 앞서 추출한 ROI 데이터를 실제 ControlNet 학습이 가능한 형태로 가공합니다.

## 구현된 모듈

### 1. Multi-Channel Hint 이미지 생성 (`src/preprocessing/hint_generator.py`)

3채널 힌트 이미지를 생성하여 단순 흑백 마스크보다 훨씬 풍부한 조건부 정보를 제공합니다.

**Red 채널**: 결함 정밀 마스크 (4대 지표 반영)
- High linearity (선형성 > 0.7) → Skeleton 추출 + 엣지 강조
- High solidity (견고성 > 0.8) → Filled mask
- 기타 → Edge-emphasized mask

**Green 채널**: 배경 구조선 (패턴의 엣지 정보)
- `vertical_stripe` → 수직 엣지 강조 (Sobel X)
- `horizontal_stripe` → 수평 엣지 강조 (Sobel Y)
- `complex_pattern` → 전방향 엣지
- `smooth`/`textured` → 최소 엣지 정보

**Blue 채널**: 배경 미세 질감 (노이즈 밀도)
- Local variance 계산 (7×7 윈도우)
- `smooth` → 낮은 값 (균일)
- `textured`/`complex_pattern` → 높은 값 (복잡한 질감)

### 2. 하이브리드 프롬프트 생성 (`src/preprocessing/prompt_generator.py`)

배경 타입과 결함 Sub-class를 결합한 자연어 설명을 자동 생성합니다.

**프롬프트 구조**: `[Sub-class 특성] + [배경 타입] + [표면 상태]`

**3가지 스타일**:
- **Simple**: `"a linear scratch defect on vertical striped metal surface, class 1"`
- **Detailed**: `"a high-linearity elongated scratch on vertical striped metal surface with directional texture (pristine condition), steel defect class 1"`
- **Technical**: `"Industrial steel defect: highly linear, solid defect (class 1) on vertical striped metal surface, vertical line pattern, background stability 0.85, match quality 0.92"`

**Negative Prompt**: `"blurry, low quality, artifacts, noise, distorted, warped, unrealistic, oversaturated, cartoon, painting, text, watermark, logo"`

### 3. 데이터셋 검수 도구 (`src/utils/dataset_validator.py`)

학습 전 데이터 품질을 검증합니다.

**Distribution Check (분포 확인)**:
- Class 분포 불균형 감지 (>60% in one class)
- Defect subtype 분포 불균형 감지 (>50%)
- Background type 분포 불균형 감지 (>50%)
- 이상적인 조합 누락 감지:
  - `(linear_scratch, vertical_stripe)`
  - `(compact_blob, smooth)`
  - `(irregular, complex_pattern)`

**Visual Check (시각적 검사)**:
- 배경 패턴 연속성 확인
- 결함 위치 확인 (ROI 가장자리에서 10% 이상 마진)
- 적합도 점수 확인 (<0.5 경고)
- 연속성 점수 확인 (<0.5 경고)

**출력**:
- `distribution_analysis.png`: 분포 시각화
- `visual_inspection.png`: 샘플 이미지 + 이슈 표시

### 4. ControlNet 데이터셋 패키저 (`src/preprocessing/controlnet_packager.py`)

최종 학습 데이터셋을 패키징합니다.

**생성 파일**:
- `train.jsonl`: 학습 인덱스 파일
  ```json
  {"source": "images/xxx.png", "target": "images/xxx.png", "prompt": "...", "hint": "hints/xxx_hint.png", "negative_prompt": "..."}
  ```
- `metadata.json`: 전체 데이터셋 메타데이터
- `packaged_roi_metadata.csv`: 프롬프트가 추가된 ROI 메타데이터
- `hints/`: Multi-channel 힌트 이미지 디렉토리

### 5. 메인 실행 스크립트 (`scripts/prepare_controlnet_data.py`)

전체 파이프라인을 실행하는 CLI 도구입니다.

## 사용 방법

### 기본 실행 (전체 파이프라인)
```bash
python scripts/prepare_controlnet_data.py \
    --roi_metadata data/processed/roi_patches/roi_metadata.csv
```

### 검증 스킵 (빠른 테스트)
```bash
python scripts/prepare_controlnet_data.py \
    --roi_metadata data/processed/roi_patches/roi_metadata.csv \
    --skip_validation
```

### 힌트 이미지 없이 프롬프트만 생성
```bash
python scripts/prepare_controlnet_data.py \
    --roi_metadata data/processed/roi_patches/roi_metadata.csv \
    --skip_hints
```

### 커스텀 설정
```bash
python scripts/prepare_controlnet_data.py \
    --roi_metadata data/processed/roi_patches/roi_metadata.csv \
    --output_dir data/processed/controlnet_dataset \
    --prompt_style technical \
    --validation_samples 32 \
    --max_samples 100
```

## 출력 디렉토리 구조

```
data/processed/controlnet_dataset/
├── hints/                           # Multi-channel 힌트 이미지
│   ├── 0002cc93b_class1_region0_hint.png
│   ├── 0002cc93b_class2_region0_hint.png
│   └── ...
├── validation/                      # 검증 리포트
│   ├── distribution_analysis.png   # 분포 시각화
│   └── visual_inspection.png       # 샘플 검사
├── train.jsonl                      # ControlNet 학습 인덱스
├── metadata.json                    # 전체 메타데이터
├── packaged_roi_metadata.csv        # 업데이트된 ROI 정보
└── packaging_summary.txt            # 패키징 요약
```

## train.jsonl 포맷

각 라인은 하나의 학습 샘플:
```json
{
  "source": "roi_patches/images/0002cc93b_class1_region0.png",
  "target": "roi_patches/images/0002cc93b_class1_region0.png",
  "prompt": "a high-linearity elongated scratch on vertical striped metal surface with directional texture (pristine condition), steel defect class 1",
  "hint": "controlnet_dataset/hints/0002cc93b_class1_region0_hint.png",
  "negative_prompt": "blurry, low quality, artifacts, noise, distorted, warped, unrealistic, oversaturated, cartoon, painting, text, watermark, logo"
}
```

## 파이프라인 흐름

```
ROI 메타데이터 (roi_metadata.csv)
  ↓
[검증 단계]
  - Distribution Check → 분포 불균형 감지
  - Visual Check → 품질 이슈 감지
  ↓
[힌트 생성]
  - Red: 결함 마스크 (4대 지표 반영)
  - Green: 배경 구조선
  - Blue: 배경 질감
  ↓
[프롬프트 생성]
  - 결함 subtype + 배경 type 조합
  - Negative prompt
  ↓
[패키징]
  - train.jsonl 생성
  - metadata.json 생성
  - 디렉토리 구조화
  ↓
ControlNet 학습 준비 완료!
```

## PROJECT(prepare_control).md 체크리스트

✅ **1. Multi-Channel Hint 이미지 제작**
- Red: 4대 지표 반영 결함 마스크
- Green: 배경 구조선 (Stripe 엣지)
- Blue: 배경 미세 질감

✅ **2. 하이브리드 프롬프트 생성**
- 구조: [Sub-class] + [Background] + [Surface]
- 예시: "A high-linearity scratch on a vertical striped metal surface with smooth texture."
- train.jsonl에 자동 저장

✅ **3. 학습 데이터셋 최종 검수**
- Distribution Check: 불균형 감지 및 경고
- Visual Check: 샘플링 검사 (배경 연속성, 결함 위치)

✅ **4. ControlNet 학습 설정 준비**
- train.jsonl 생성
- 표준 ControlNet 포맷 준수
- PROJECT(control_net).md로 진행 가능

## 다음 단계

PROJECT(control_net).md를 참고하여:
1. Base Model 선정 (Stable Diffusion v1.5 등)
2. Hyperparameter 설정
3. ControlNet 본격 학습 시작

## 핵심 기여

1. **단순 마스크를 넘어선 Multi-channel 조건부 입력**
   - Red: 결함 특성 강조
   - Green: 배경 패턴 보존
   - Blue: 질감 정보

2. **Context-aware 프롬프트**
   - 결함-배경 매칭 정보를 자연어로 표현
   - ControlNet이 물리적으로 타당한 생성을 하도록 유도

3. **자동화된 품질 검증**
   - 데이터 불균형 조기 감지
   - 시각적 이슈 자동 탐지

4. **표준화된 학습 포맷**
   - train.jsonl (ControlNet 표준)
   - Relative paths (이식성)

---

**구현 완료**: 2026년 2월 9일  
**연구 프레임워크**: PROJECT(prepare_control).md
