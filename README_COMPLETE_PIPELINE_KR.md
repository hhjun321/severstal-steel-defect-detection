# Severstal Steel Defect Detection - 전체 파이프라인

Severstal Steel Defect Detection 데이터셋에서 고품질 ControlNet 훈련 데이터를 준비하기 위한 `PROJECT(roi).md` 및 `PROJECT(prepare_control).md`에 설명된 연구 파이프라인의 완전한 구현입니다.

## 프로젝트 개요

본 프로젝트는 결함 특성을 적절한 배경 컨텍스트와 신중하게 매칭하여 결함 데이터 증강에 대한 새로운 접근 방식을 구현합니다. 핵심 통찰은: **"어떤 결함 하위 클래스" + "어떤 배경 컨텍스트" = 고품질 합성 데이터**입니다.

## 구현된 연구 논문

### 1. PROJECT(roi).md - 통계적 지표 기반 ROI 추출
결함 기하학 및 배경 적합성을 모두 분석하여 관심 영역(ROI)을 추출합니다.

### 2. PROJECT(prepare_control).md - ControlNet 훈련 데이터 준비
추출된 ROI를 다중 채널 힌트 및 하이브리드 프롬프트와 함께 ControlNet 훈련 형식으로 패키징합니다.

## 전체 파이프라인

```
원시 데이터 (이미지 + RLE 마스크)
  ↓
┌─────────────────────────────────────────────────┐
│ 1단계: ROI 추출 (PROJECT(roi).md)              │
├─────────────────────────────────────────────────┤
│ 1. 배경 분석 (그리드 기반)                      │
│    - 분류: smooth, textured, stripe 등         │
│    - 안정성 점수 계산                           │
│ 2. 결함 분석 (4가지 지표)                      │
│    - 직선성, 치밀도, 분산도, 종횡비             │
│    - 하위 유형 분류                             │
│ 3. ROI 적합성 평가                             │
│    - 결함-배경 조합 매칭                        │
│    - ROI 위치 최적화                           │
│ 4. ROI 추출 및 패키징                          │
│    - 메타데이터가 있는 512×512 패치             │
└─────────────────────────────────────────────────┘
  ↓ ROI 메타데이터 CSV
┌─────────────────────────────────────────────────┐
│ 2단계: ControlNet 준비 (prepare_control.md)    │
├─────────────────────────────────────────────────┤
│ 1. 다중 채널 힌트 생성                          │
│    - 빨강: 결함 마스크 (4-지표 향상)            │
│    - 초록: 배경 구조선                          │
│    - 파랑: 배경 텍스처                          │
│ 2. 하이브리드 프롬프트 생성                     │
│    - 결함 + 배경 설명 결합                      │
│ 3. 데이터셋 검증                                │
│    - 분포 확인                                  │
│    - 시각적 검사                                │
│ 4. 훈련용 패키징                                │
│    - train.jsonl + hints/ + metadata           │
└─────────────────────────────────────────────────┘
  ↓
ControlNet 훈련 준비 완료 데이터셋
```

## 빠른 시작

### 설치

```bash
# 의존성 설치
pip install -r requirements.txt
```

### 1단계: ROI 추출

```bash
# 10개 이미지로 테스트
python scripts/extract_rois.py --max_images 10

# 모든 이미지 처리
python scripts/extract_rois.py
```

**출력**: `data/processed/roi_patches/roi_metadata.csv` + ROI 이미지/마스크 패치

### 2단계: ControlNet 데이터 준비

```bash
# 검증을 포함한 전체 파이프라인
python scripts/prepare_controlnet_data.py \
    --roi_metadata data/processed/roi_patches/roi_metadata.csv

# 빠른 테스트 (속도를 위해 힌트 건너뛰기)
python scripts/prepare_controlnet_data.py \
    --roi_metadata data/processed/roi_patches/roi_metadata.csv \
    --skip_hints \
    --max_samples 50
```

**출력**: `data/processed/controlnet_dataset/train.jsonl` + 다중 채널 힌트

## 프로젝트 구조

```
severstal-steel-defect-detection/
├── src/
│   ├── utils/
│   │   ├── rle_utils.py              # RLE 인코딩/디코딩
│   │   └── dataset_validator.py      # 품질 검증
│   ├── analysis/
│   │   ├── defect_characterization.py    # 4가지 지표
│   │   ├── background_characterization.py # 그리드 기반 분석
│   │   └── roi_suitability.py           # 매칭 평가
│   └── preprocessing/
│       ├── roi_extraction.py           # 1단계 파이프라인
│       ├── hint_generator.py           # 다중 채널 힌트
│       ├── prompt_generator.py         # 하이브리드 프롬프트
│       └── controlnet_packager.py      # 2단계 파이프라인
├── scripts/
│   ├── extract_rois.py                # 1단계 CLI
│   └── prepare_controlnet_data.py     # 2단계 CLI
├── data/
│   └── processed/
│       ├── roi_patches/               # 1단계 출력
│       └── controlnet_dataset/        # 2단계 출력
├── train_images/                      # 원시 훈련 이미지
├── train.csv                          # RLE 어노테이션
├── PROJECT(roi).md                    # 연구 문서 1
├── PROJECT(prepare_control).md        # 연구 문서 2
├── IMPLEMENTATION_SUMMARY_KR.md       # 1단계 요약
└── IMPLEMENTATION_CONTROLNET_PREP_KR.md # 2단계 요약
```

## 주요 기능

### 1단계: ROI 추출

**1. 결함 특성화 (4가지 지표)**
- 직선성: 얼마나 선형/길쭉한지 (고유값 분석)
- 치밀도: 압축도 (면적/볼록_껍질)
- 분산도: 경계 상자 채우기 (면적/bbox)
- 종횡비: 길쭉함 (주축/부축)

**하위 유형 분류**:
- `linear_scratch`: 높은 직선성 + 높은 종횡비
- `compact_blob`: 낮은 종횡비 + 높은 치밀도
- `elongated`: 높은 종횡비 + 중간 직선성
- `irregular`: 낮은 치밀도
- `general`: 기본값

**2. 배경 특성화 (그리드 기반)**

64×64 그리드 패치 분석:
- **분산** → smooth vs textured
- **Sobel 경계선** → 수직/수평 줄무늬
- **FFT** → 복잡한 패턴
- **안정성 점수** → 균일성

**3. ROI 적합성 점수**

```
적합성 = 0.5×매칭 + 0.3×연속성 + 0.2×안정성

매칭 규칙:
- linear_scratch + (vertical|horizontal)_stripe → 1.0
- compact_blob + smooth → 1.0
- irregular + complex_pattern → 1.0
```

**4. 위치 최적화**

결함을 중앙에 유지하면서 배경 연속성을 최대화하기 위해 ROI 윈도우를 (±32px) 이동합니다.

### 2단계: ControlNet 준비

**1. 다중 채널 힌트 이미지**

- **빨강**: 지표로 향상된 결함 마스크
  - 높은 직선성 → 골격 추출
  - 높은 치밀도 → 채워진 마스크
  - 그 외 → 경계선 강조
- **초록**: 배경 구조 (Sobel 경계선)
  - 수직 줄무늬 → 수직 경계선
  - 수평 줄무늬 → 수평 경계선
  - 복잡함 → 모든 경계선
- **파랑**: 배경 텍스처 (로컬 분산)
  - Smooth → 낮은 값
  - Textured → 높은 값

**2. 하이브리드 프롬프트**

구조: `[결함 특성] + [배경 유형] + [표면 상태]`

예시 (상세 스타일):
```
"수직 줄무늬 금속 표면의 높은 직선성 길쭉한 스크래치, 
방향성 텍스처 (깨끗한 상태), 철강 결함 클래스 1"
```

**3. 데이터셋 검증**

- **분포 확인**: 클래스/하위유형/배경 불균형 감지 (>50-60%)
- **시각적 확인**: 경계선 위치 및 품질 문제에 대한 샘플 검사

**4. ControlNet 형식**

`train.jsonl`:
```json
{
  "image": "roi_patches/images/00a0f53fb_class1_region0.png",
  "mask": "roi_patches/masks/00a0f53fb_class1_region0.png",
  "hint": "controlnet_dataset/hints/00a0f53fb_class1_region0.png",
  "prompt": "수직 줄무늬 배경의 높은 직선성 선형 스크래치...",
  "class_id": 1,
  "defect_subtype": "linear_scratch",
  "background_type": "vertical_stripe"
}
```

## 실행 세부사항

### 1단계: ROI 추출

**매개변수:**
```bash
python scripts/extract_rois.py \
    --image_dir train_images \
    --train_csv train.csv \
    --output_dir data/processed/roi_patches \
    --roi_size 512 \
    --grid_size 64 \
    --min_suitability 0.5 \
    --max_images 100
```

**예상 출력:**
- ~3,000-5,000 ROI 패치 (데이터셋 크기에 따라)
- 각 ROI에 대한 상세 메타데이터
- 적합성 점수 ≥0.7인 경우 ~60-70%

**실행 시간:**
- 이미지당 ~2-3초
- 전체 데이터셋: ~7-10시간

### 2단계: ControlNet 데이터 준비

**매개변수:**
```bash
python scripts/prepare_controlnet_data.py \
    --roi_metadata data/processed/roi_patches/roi_metadata.csv \
    --output_dir data/processed/controlnet_dataset \
    --prompt_style detailed \
    --skip_validation \
    --max_samples 1000
```

**프롬프트 스타일:**
- `simple`: 기본 설명
- `detailed`: 포괄적인 특성 (권장)
- `technical`: 메트릭 포함

**예상 출력:**
- `train.jsonl`: 훈련 인덱스
- `hints/`: RGB 다중 채널 힌트 이미지
- `metadata.json`: 데이터셋 통계
- `validation/`: 분포 리포트

**실행 시간:**
- ROI당 ~0.5-1초
- 3,000 ROI: ~30-50분

## 데이터셋 통계

### 예상 분포

**클래스별:**
- 클래스 1: ~35-40%
- 클래스 2: ~20-25%
- 클래스 3: ~25-30%
- 클래스 4: ~10-15%

**결함 하위 유형별:**
- linear_scratch: ~40-45%
- compact_blob: ~25-30%
- elongated: ~15-20%
- irregular: ~10-15%

**배경 유형별:**
- smooth: ~30-35%
- vertical_stripe: ~25-30%
- horizontal_stripe: ~15-20%
- textured: ~10-15%
- complex_pattern: ~5-10%

## 품질 관리

### 자동 검증

파이프라인은 자동으로 확인합니다:
1. **클래스 불균형**: 단일 클래스가 >60% 지배하는지
2. **하위 유형 커버리지**: 모든 하위 유형이 표현되는지
3. **배경 다양성**: 충분한 배경 변형이 있는지
4. **적합성 분포**: 대부분의 ROI가 적합성 임계값을 통과하는지

### 수동 검사

다음을 위해 샘플 검토:
- 경계선 위치 정확도
- 힌트 이미지 품질
- 프롬프트 설명의 정확성
- 결함-배경 매칭 적절성

## 문제 해결

### 일반적인 문제

**1. 낮은 적합성 점수**

증상: 대부분의 ROI가 "unsuitable"로 표시됨

해결책:
```bash
# 임계값 낮추기
python scripts/extract_rois.py --min_suitability 0.3

# 또는 매칭 규칙 검토 및 조정
# src/analysis/roi_suitability.py 편집
```

**2. 클래스 불균형 경고**

증상: "Class distribution highly imbalanced" 경고

해결책:
- 예상됨 (원본 데이터셋 특성)
- 나중 단계에서 클래스별 샘플링으로 처리
- 또는 `--balance_classes` 플래그 사용 (구현된 경우)

**3. 메모리 부족 오류**

증상: 큰 이미지 처리 시 프로세스 종료

해결책:
```bash
# 배치로 처리
python scripts/extract_rois.py --max_images 1000
python scripts/extract_rois.py --max_images 1000 --offset 1000
```

**4. 느린 처리**

증상: 예상보다 오래 걸림

해결책:
```bash
# 힌트 생성 건너뛰기 (테스트용)
python scripts/prepare_controlnet_data.py --skip_hints

# 또는 검증 건너뛰기
python scripts/prepare_controlnet_data.py --skip_validation
```

## 다음 단계

데이터 준비 후:

### 3단계: ControlNet 훈련

```bash
python scripts/train_controlnet.py \
    --data_dir data/processed/controlnet_dataset \
    --batch_size 4 \
    --num_epochs 100 \
    --lr 1e-4
```

### 4단계: 증강 데이터 생성

학습된 모델로:
- 새로운 배경에서 합성 결함 생성
- 클래스별 샘플 균형
- 크기/위치/방향 변형

## 성능 벤치마크

**시스템 사양**: RTX 3060 (12GB VRAM), 16GB RAM, i7 CPU

| 단계 | 소요 시간 | GPU 사용 | 디스크 |
|------|----------|----------|--------|
| ROI 추출 | 7-10시간 | 0% | ~5-8 GB |
| ControlNet 준비 | 30-50분 | 0% | ~3-5 GB |
| ControlNet 훈련 | 10-20시간 | 90-95% | ~2 GB |
| 증강 생성 | 2-4시간 | 80-90% | ~10-15 GB |

## 고급 사용법

### 사용자 정의 매칭 규칙

`src/analysis/roi_suitability.py` 편집:

```python
MATCHING_RULES = {
    ("linear_scratch", "vertical_stripe"): 1.0,
    ("linear_scratch", "horizontal_stripe"): 1.0,
    ("compact_blob", "smooth"): 1.0,
    # 사용자 정의 규칙 추가
    ("elongated", "textured"): 0.8,
}
```

### 사용자 정의 프롬프트 템플릿

`src/preprocessing/prompt_generator.py` 편집:

```python
def generate_detailed_prompt(self, defect_info, bg_info):
    template = (
        "a {linearity_desc} {subtype} "
        "on {bg_type} metal surface "
        "with {texture_desc} texture"
    )
    return template.format(**info)
```

### 배치 처리

여러 이미지 세트 처리:

```bash
# 스크립트 생성: batch_process.sh
for i in {0..10}; do
    start=$((i * 1000))
    python scripts/extract_rois.py \
        --max_images 1000 \
        --offset $start \
        --output_dir data/batch_$i
done
```

## 참조 문서

- **연구 문서**:
  - `PROJECT(roi).md`: ROI 추출 이론
  - `PROJECT(prepare_control).md`: ControlNet 준비 상세

- **구현 요약**:
  - `IMPLEMENTATION_SUMMARY_KR.md`: 1단계 구현
  - `IMPLEMENTATION_CONTROLNET_PREP_KR.md`: 2단계 구현

- **가이드**:
  - `README_ROI_KR.md`: ROI 추출 가이드
  - `CONTROLNET_TRAINING_GUIDE.md`: 훈련 가이드

## 기여

이슈 및 개선 제안을 환영합니다. 기여 시:
1. 기존 코드 스타일 따르기
2. docstring으로 새 함수 문서화
3. 주요 변경 사항에 대한 테스트 추가

## 라이선스

본 프로젝트는 Severstal Steel Defect Detection 챌린지의 일부입니다.

## 인용

본 연구를 사용하는 경우 인용하십시오:

```bibtex
@misc{severstal-steel-defect-casda,
  title={Context-Aware Steel Defect Augmentation with ControlNet},
  author={Your Name},
  year={2026},
  note={Severstal Steel Defect Detection Challenge}
}
```
