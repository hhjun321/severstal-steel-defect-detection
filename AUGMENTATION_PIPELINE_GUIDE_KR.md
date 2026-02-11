# 컨텍스트 인식 철강 결함 증강(CASDA) 파이프라인 가이드

## 목차
1. [개요](#개요)
2. [아키텍처](#아키텍처)
3. [사전 요구사항](#사전-요구사항)
4. [설치](#설치)
5. [파이프라인 실행](#파이프라인-실행)
6. [매개변수 참조](#매개변수-참조)
7. [출력 구조](#출력-구조)
8. [품질 관리](#품질-관리)
9. [문제 해결](#문제-해결)
10. [성능 벤치마크](#성능-벤치마크)
11. [고급 구성](#고급-구성)

---

## 개요

컨텍스트 인식 철강 결함 증강(CASDA) 파이프라인은 학습된 ControlNet 모델을 사용하여 사실적인 합성 철강 결함 이미지를 생성합니다. 시스템은 결함 유형을 호환 가능한 배경 텍스처와 매칭하여 물리적 타당성을 보장합니다.

### 주요 기능
- **물리 인식 증강**: 호환 가능한 배경에만 결함 배치
- **제어된 크기 변형**: 원본 결함 크기의 80-100% (확대 없음)
- **클래스 균형 생성**: 4개 결함 클래스에 균등한 샘플
- **다단계 품질 검증**: 블러, 아티팩트, 일관성 검사
- **GPU 가속**: CUDA 지원으로 빠른 생성
- **원활한 통합**: 원본 훈련 데이터와 직접 CSV 병합

### 목표 지표
- **증강 규모**: 원본 데이터셋의 20% (~2,514개 샘플)
- **클래스별 샘플**: 클래스당 ~628-629개 샘플 (총 4개 클래스)
- **품질 임계값**: 0.7 (조정 가능)
- **예상 통과율**: 검증 후 70-85%

---

## 아키텍처

### 5단계 파이프라인

```
┌──────────────────────────────────────────────────────────────┐
│  1단계: 배경 추출                                              │
│  ├─ 결함 없는 영역에서 512x512 패치 추출                       │
│  ├─ 텍스처 유형별 분류 (매끄러움, 줄무늬, 질감)                 │
│  └─ 출력: data/backgrounds/                                   │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  2단계: 결함 템플릿 라이브러리                                  │
│  ├─ 클래스 & 하위 유형별 기존 ROI 메타데이터 인덱싱             │
│  ├─ 배경 호환성 규칙 계산                                      │
│  └─ 출력: data/defect_templates/                             │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  3단계: 증강 데이터 생성 (핵심)                                │
│  ├─ 학습된 ControlNet 모델 로드                               │
│  ├─ 호환 가능한 배경-결함 쌍 샘플링                            │
│  ├─ 크기 변형을 가진 합성 마스크 생성                          │
│  ├─ 다중 채널 힌트 생성 (결함 + 경계선 + 텍스처)               │
│  ├─ ControlNet 추론 실행                                      │
│  └─ 출력: data/augmented/                                     │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  4단계: 품질 검증                                              │
│  ├─ 블러 검사 (Laplacian 분산)                                │
│  ├─ 아티팩트 감지 (그래디언트 분석)                            │
│  ├─ 색상 일관성 검증 (LAB 색 공간)                            │
│  ├─ 결함 메트릭 일관성 확인                                   │
│  └─ 출력: data/augmented/validation/                         │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  5단계: 데이터셋 병합                                          │
│  ├─ 증강된 마스크를 RLE 형식으로 변환                          │
│  ├─ 원본 train.csv와 병합                                     │
│  ├─ 통계 생성 (원본 vs 증강)                                  │
│  └─ 출력: data/final_dataset/train_augmented.csv             │
└──────────────────────────────────────────────────────────────┘
```

### 모듈 의존성

파이프라인은 다음 기존 모듈을 사용 (이전 개발에서):
- `src/analysis/defect_characterization.py`: 결함 메트릭 분석
- `src/analysis/background_characterization.py`: 배경 분류
- `src/analysis/roi_suitability.py`: 호환성 매칭 규칙
- `src/preprocessing/hint_generator.py`: 다중 채널 힌트 생성
- `src/preprocessing/prompt_generator.py`: 텍스트 프롬프트 생성
- `src/utils/rle.py`: RLE 인코딩/디코딩 유틸리티

---

## 사전 요구사항

### 필수 파일

시작하기 전에 다음을 준비:

1. **원본 데이터셋**
   - `train.csv`: RLE 인코딩된 마스크가 있는 원본 훈련 레이블
   - `train_images/`: ~12,568개 훈련 이미지가 있는 디렉토리 (1600×256 픽셀)

2. **ROI 메타데이터** (이전 파이프라인에서)
   - `data/processed/roi_patches/roi_metadata.csv`
   - 생성: `scripts/extract_rois.py`

3. **학습된 ControlNet 모델**
   - `outputs/controlnet_training/best.pth`
   - 생성: `scripts/train_controlnet.py`

### 시스템 요구사항

**하드웨어**:
- GPU: NVIDIA GPU with ≥8GB VRAM (권장: RTX 3060 이상)
- RAM: ≥16GB
- 저장소: 증강 데이터용 ≥10GB 여유 공간

**소프트웨어**:
- Python: 3.8+
- CUDA: 11.0+ (GPU 가속용)
- 운영 체제: Windows/Linux/macOS

### Python 의존성

```bash
pip install numpy pandas opencv-python scikit-image torch torchvision tqdm pillow matplotlib
```

**버전 요구사항**:
- torch>=1.10.0 (CUDA 지원)
- torchvision>=0.11.0
- opencv-python>=4.5.0
- scikit-image>=0.18.0
- numpy>=1.21.0
- pandas>=1.3.0

---

## 파이프라인 실행

### 빠른 시작 (전체 파이프라인)

자동 실행 스크립트 사용:

```bash
python scripts/run_augmentation_pipeline.py \
    --train_csv train.csv \
    --image_dir train_images \
    --model_path outputs/controlnet_training/best.pth \
    --output_dir data \
    --num_samples 2500
```

### 단계별 수동 실행

#### 1단계: 깨끗한 배경 추출

```bash
python scripts/extract_clean_backgrounds.py \
    --train_csv train.csv \
    --image_dir train_images \
    --output_dir data/backgrounds \
    --patch_size 512 \
    --patches_per_image 5 \
    --min_quality 0.7 \
    --stride 256
```

**예상 소요 시간**: 10-20분
**예상 출력**: ~3,000-5,000개 배경 패치

**검증**:
```bash
ls data/backgrounds/
# 다음을 확인: smooth/, vertical_stripe/, horizontal_stripe/, textured/, complex_pattern/

python -c "import json; data=json.load(open('data/backgrounds/background_inventory.json')); print(f'총 배경 수: {len(data)}')"
```

#### 2단계: 결함 템플릿 라이브러리 구축

```bash
python scripts/build_defect_templates.py \
    --roi_metadata data/processed/roi_patches/roi_metadata.csv \
    --output_dir data/defect_templates \
    --min_suitability 0.7
```

**예상 소요 시간**: 1-2분
**예상 출력**: ~1,000-3,000개 결함 템플릿이 있는 템플릿 메타데이터

#### 3단계: 증강 데이터 생성 (핵심)

```bash
python scripts/generate_augmented_data.py \
    --model_path outputs/controlnet_training/best.pth \
    --backgrounds_dir data/backgrounds \
    --templates_dir data/defect_templates \
    --output_dir data/augmented \
    --num_samples 2500 \
    --samples_per_class '{"1":625,"2":625,"3":625,"4":625}' \
    --scale_min 0.8 \
    --scale_max 1.0 \
    --device cuda \
    --batch_size 4 \
    --save_hints
```

**예상 소요 시간**: 30-60분 (GPU 성능에 따라)
**예상 출력**: 2,500개 증강 이미지-마스크 쌍

#### 4단계: 증강 품질 검증

```bash
python scripts/validate_augmented_quality.py \
    --augmented_dir data/augmented \
    --output_dir data/augmented/validation \
    --min_quality_score 0.7 \
    --check_blur \
    --check_artifacts \
    --check_color \
    --check_defect_consistency \
    --check_defect_presence
```

**예상 소요 시간**: 5-10분
**예상 통과율**: 70-85% (1,750-2,125개 샘플)

#### 5단계: 데이터셋 병합

```bash
python scripts/merge_datasets.py \
    --original_csv train.csv \
    --original_images train_images \
    --augmented_dir data/augmented \
    --output_csv data/final_dataset/train_augmented.csv \
    --output_dir data/final_dataset \
    --use_only_passed
```

**예상 소요 시간**: 5-10분
**예상 출력**: 14,318-14,693개 총 샘플이 있는 train_augmented.csv

---

## 매개변수 참조

### extract_clean_backgrounds.py

| 매개변수 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `--train_csv` | str | 필수 | train.csv 경로 |
| `--image_dir` | str | 필수 | train_images/ 경로 |
| `--output_dir` | str | 필수 | 배경 출력 디렉토리 |
| `--patch_size` | int | 512 | 추출된 패치 크기 (512×512) |
| `--patches_per_image` | int | 5 | 이미지당 최대 패치 수 |
| `--min_quality` | float | 0.7 | 최소 품질 임계값 (0-1) |
| `--stride` | int | 256 | 슬라이딩 윈도우 보폭 |

### generate_augmented_data.py

| 매개변수 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `--model_path` | str | 필수 | 학습된 ControlNet 모델 경로 |
| `--backgrounds_dir` | str | 필수 | 배경 디렉토리 경로 |
| `--templates_dir` | str | 필수 | 결함 템플릿 경로 |
| `--output_dir` | str | 필수 | 증강 데이터 출력 디렉토리 |
| `--num_samples` | int | 2500 | 생성할 총 샘플 수 |
| `--samples_per_class` | json | 균형 | 클래스별 샘플 (JSON 딕셔너리) |
| `--scale_min` | float | 0.8 | 최소 결함 크기 스케일 인수 |
| `--scale_max` | float | 1.0 | 최대 결함 크기 스케일 인수 |
| `--device` | str | cuda | 장치 (cuda/cpu) |
| `--batch_size` | int | 4 | 추론 배치 크기 |
| `--seed` | int | 42 | 재현성을 위한 랜덤 시드 |

---

## 출력 구조

### 전체 디렉토리 트리

```
data/
├── backgrounds/
│   ├── smooth/
│   ├── vertical_stripe/
│   ├── horizontal_stripe/
│   ├── textured/
│   ├── complex_pattern/
│   └── background_inventory.json
│
├── defect_templates/
│   ├── templates_metadata.json
│   ├── template_statistics.json
│   └── matching_rules.json
│
├── augmented/
│   ├── images/                    # 2,500개 파일
│   ├── masks/                     # 2,500개 파일
│   ├── hints/ (선택)
│   ├── augmented_metadata.json
│   ├── generation_log.txt
│   └── validation/
│       ├── quality_scores.json
│       ├── passed_samples.txt
│       ├── rejected_samples.txt
│       └── validation_statistics.json
│
└── final_dataset/
    ├── train_augmented.csv
    ├── dataset_statistics.json
    └── dataset_statistics.txt
```

---

## 품질 관리

### 품질 검증 검사

#### 1. 블러 감지 (20% 가중치)
- **방법**: Laplacian 분산
- **임계값**: >100 (선명), <50 (흐림)
- **점수**: `min(variance / 200, 1.0)`

#### 2. 아티팩트 감지 (20% 가중치)
- **방법**: 그래디언트 크기 분석
- **임계값**: 95번째 백분위수 < 150
- **감지**: 비정상 경계선, 후광, 노이즈 패턴

#### 3. 색상 일관성 (15% 가중치)
- **방법**: LAB 색 공간 통계
- **검사**: 휘도 안정성, 색상 범위
- **플래그**: 비정상적인 색상 변화, 채도 문제

#### 4. 결함 메트릭 일관성 (25% 가중치)
- **방법**: 생성된 결함 메트릭 재분석
- **비교**: 예상 하위 유형 vs 실제 메트릭
- **임계값**:
  - 선형 스크래치: linearity > 0.7
  - 컴팩트 블롭: solidity > 0.7
  - 길쭉함: aspect_ratio > 2.0

#### 5. 결함 존재 (20% 가중치)
- **방법**: 마스크 영역 분석
- **범위**: 이미지 영역의 0.1% - 30%
- **플래그**: 너무 작음 (보이지 않음) 또는 너무 큼 (비현실적)

### 품질 점수 해석

| 점수 범위 | 품질 수준 | 조치 |
|-----------|----------|------|
| 0.9 - 1.0 | 우수 | 직접 사용 |
| 0.7 - 0.9 | 좋음 | 사용 (기본 임계값) |
| 0.5 - 0.7 | 경계 | 수동 검토 |
| 0.0 - 0.5 | 나쁨 | 거부 |

---

## 문제 해결

### 일반적인 문제

#### 문제 1: "CUDA 메모리 부족"

**증상**:
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**해결책**:
```bash
# 해결책 1: 배치 크기 감소
python scripts/generate_augmented_data.py --batch_size 2  # 기본값: 4

# 해결책 2: CPU 사용 (느림)
python scripts/generate_augmented_data.py --device cpu

# 해결책 3: 단계적으로 생성
python scripts/generate_augmented_data.py --num_samples 1000
```

#### 문제 2: "호환 가능한 배경-결함 쌍을 찾을 수 없음"

**증상**:
```
Error: Could not find compatible background for template_id=XXX after 100 attempts
```

**근본 원인**: 지나치게 제한적인 매칭 규칙 또는 불충분한 배경 다양성

**해결책**:
```bash
# 해결책 1: 적합성 임계값 낮추기
python scripts/build_defect_templates.py --min_suitability 0.5  # 기본값: 0.7

# 해결책 2: 더 많은 배경 추출
python scripts/extract_clean_backgrounds.py --patches_per_image 10  # 기본값: 5

# 해결책 3: 배경 품질 임계값 낮추기
python scripts/extract_clean_backgrounds.py --min_quality 0.5  # 기본값: 0.7
```

#### 문제 3: 낮은 검증 통과율 (<60%)

**해결책**:
```bash
# 해결책 1: 거부 이유 분석
cat data/augmented/validation/quality_report.txt | grep "Rejection reasons"

# 해결책 2: 품질 임계값 조정
python scripts/validate_augmented_quality.py --min_quality_score 0.6

# 해결책 3: ControlNet 훈련 개선
# 더 많은 에폭 또는 더 나은 하이퍼파라미터로 재훈련
```

---

## 성능 벤치마크

### 예상 실행 시간

**시스템 사양**: RTX 3060 (12GB VRAM), 16GB RAM, i7 CPU

| 단계 | 소요 시간 | GPU 사용 | 출력 크기 |
|------|----------|----------|----------|
| 1단계: 배경 추출 | 10-20분 | 0% | ~1-2 GB |
| 2단계: 템플릿 라이브러리 | 1-2분 | 0% | <10 MB |
| 3단계: 데이터 생성 | 30-60분 | 80-95% | ~2-4 GB |
| 4단계: 품질 검증 | 5-10분 | 0% | <50 MB |
| 5단계: 데이터셋 병합 | 5-10분 | 0% | ~100 MB |
| **합계** | **51-103분** | - | **~3-6 GB** |

### 처리량 메트릭

- **배경 추출**: ~10-20 이미지/초
- **데이터 생성**: ~0.7-1.5 샘플/초 (GPU 의존)
- **품질 검증**: ~4-8 샘플/초
- **데이터셋 병합**: ~200-400 샘플/초 (RLE 인코딩)

---

## 고급 구성

### 사용자 정의 결함 크기 분포

균일한 80-100% 스케일링 대신 사용자 정의 분포 사용:

```python
# generate_augmented_data.py에서 scale_factor 샘플링 수정:
# 교체:
scale_factor = random.uniform(args.scale_min, args.scale_max)

# 다음으로:
scale_factor = np.random.beta(a=5, b=2) * 0.2 + 0.8  # 90%에서 피크
```

### 사용자 정의 클래스 분포

불균형 증강의 경우:

```bash
python scripts/generate_augmented_data.py \
    --samples_per_class '{"1":800,"2":600,"3":600,"4":500}'
```

### 다중 GPU 생성

GPU 간 생성 분할:

```bash
# 터미널 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python scripts/generate_augmented_data.py \
    --num_samples 1250 --output_dir data/augmented_gpu0

# 터미널 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python scripts/generate_augmented_data.py \
    --num_samples 1250 --output_dir data/augmented_gpu1

# 출력 병합
python scripts/merge_augmented_outputs.py \
    --input_dirs data/augmented_gpu0 data/augmented_gpu1 \
    --output_dir data/augmented
```

---

## 부록

### 배경 유형 정의

| 유형 | 설명 | 특징 |
|------|------|------|
| `smooth` | 균일한 텍스처 | 낮은 분산, 최소 패턴 |
| `vertical_stripe` | 수직선/줄무늬 | 강한 수직 경계선 |
| `horizontal_stripe` | 수평선/줄무늬 | 강한 수평 경계선 |
| `textured` | 복잡한 텍스처 | 높은 국부 분산 |
| `complex_pattern` | 혼합 패턴 | 여러 패턴 유형 |

### 결함 하위 유형 정의

| 하위 유형 | 특징 | 일반적인 메트릭 |
|----------|------|----------------|
| `linear_scratch` | 길고 얇은 스크래치 | linearity>0.7, aspect_ratio>3 |
| `elongated` | 늘어난 결함 | aspect_ratio>2, linearity<0.7 |
| `compact_blob` | 원형/둥근 결함 | solidity>0.7, aspect_ratio<2 |
| `irregular` | 복잡한 모양 | 낮은 solidity, 다양한 메트릭 |
| `general` | 기본 범주 | 중간 메트릭 |

---

## 빠른 참조 명령어

```bash
# 전체 파이프라인 (자동)
python scripts/run_augmentation_pipeline.py --train_csv train.csv --image_dir train_images --model_path outputs/controlnet_training/best.pth

# 개별 단계
python scripts/extract_clean_backgrounds.py --train_csv train.csv --image_dir train_images --output_dir data/backgrounds
python scripts/build_defect_templates.py --roi_metadata data/processed/roi_patches/roi_metadata.csv --output_dir data/defect_templates
python scripts/generate_augmented_data.py --model_path outputs/controlnet_training/best.pth --backgrounds_dir data/backgrounds --templates_dir data/defect_templates --output_dir data/augmented
python scripts/validate_augmented_quality.py --augmented_dir data/augmented --output_dir data/augmented/validation
python scripts/merge_datasets.py --original_csv train.csv --augmented_dir data/augmented --output_csv data/final_dataset/train_augmented.csv

# 검증
ls data/backgrounds/ | wc -l
ls data/augmented/images/ | wc -l
cat data/augmented/validation/validation_statistics.json
wc -l data/final_dataset/train_augmented.csv
```
