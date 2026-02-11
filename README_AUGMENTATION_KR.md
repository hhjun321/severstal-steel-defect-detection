# CASDA 파이프라인 - 빠른 시작 가이드

## 개요

이 프로젝트는 Severstal Steel Defect Detection 챌린지를 위해 ControlNet을 사용하여 사실적인 합성 결함 이미지를 생성하는 **컨텍스트 인식 철강 결함 증강(CASDA)** 파이프라인을 구현합니다.

## 구현된 내용

### 완전한 5단계 파이프라인

✅ **1단계: 배경 추출** (`scripts/extract_clean_backgrounds.py`)
- 훈련 이미지에서 결함 없는 512×512 패치 추출
- 텍스처 유형별 배경 분류
- 블러/대비/노이즈 메트릭을 사용한 품질 점수 산출

✅ **2단계: 결함 템플릿 라이브러리** (`scripts/build_defect_templates.py`)
- 클래스, 하위 유형 및 배경별 ROI 메타데이터 인덱싱
- 호환성 매칭 규칙 계산
- 검색 가능한 템플릿 데이터베이스 생성

✅ **3단계: 증강 데이터 생성** (`scripts/generate_augmented_data.py`)
- **핵심 스크립트** - 학습된 ControlNet 모델 사용
- 80-100% 크기 변형으로 합성 결함 마스크 생성
- 다중 채널 힌트 생성 (결함 + 경계선 + 텍스처)
- GPU 가속 추론
- 클래스 균형 샘플링

✅ **4단계: 품질 검증** (`scripts/validate_augmented_quality.py`)
- 다중 메트릭 검증 (블러, 아티팩트, 색상, 일관성, 존재)
- 가중치 품질 점수
- 임계값별 필터링 (기본값: 0.7)

✅ **5단계: 데이터셋 병합** (`scripts/merge_datasets.py`)
- 증강된 마스크를 RLE 형식으로 변환
- 원본 train.csv와 병합
- 포괄적인 통계 생성

### 지원 도구

✅ **자동 실행** (`scripts/run_augmentation_pipeline.py`)
- 단일 명령으로 파이프라인 실행
- 진행 상황 추적 및 오류 처리
- 실행 시간 보고

✅ **시각화** (`scripts/visualize_augmented_samples.py`)
- 증강 샘플의 시각적 검사
- 품질 점수 분포
- 클래스 및 배경 분포
- 상세한 단일 샘플 분석

✅ **단위 테스트** (`tests/test_augmentation_pipeline.py`)
- 모든 주요 기능에 대한 테스트
- 형식 준수 검증
- RLE 인코딩/디코딩 테스트

✅ **종합 문서** (`AUGMENTATION_PIPELINE_GUIDE.md`)
- 70페이지 이상의 상세 가이드
- 아키텍처 다이어그램
- 매개변수 참조
- 문제 해결 섹션
- 성능 벤치마크

## 빠른 시작

### 사전 요구사항

**필수 파일** (준비 필요):
```
train.csv                                    # 원본 훈련 레이블
train_images/                                # 12,568개 훈련 이미지
data/processed/roi_patches/roi_metadata.csv  # extract_rois.py 출력
outputs/controlnet_training/best.pth         # 학습된 ControlNet 모델
```

**시스템 요구사항**:
- Python 3.8+
- NVIDIA GPU (≥8GB VRAM)
- 16GB RAM
- 10GB 여유 디스크 공간

**의존성 설치**:
```bash
pip install numpy pandas opencv-python scikit-image torch torchvision tqdm pillow matplotlib
```

### 옵션 1: 자동 실행 (권장)

전체 파이프라인을 단일 명령으로 실행:

```bash
python scripts/run_augmentation_pipeline.py \
    --train_csv train.csv \
    --image_dir train_images \
    --model_path outputs/controlnet_training/best.pth \
    --roi_metadata data/processed/roi_patches/roi_metadata.csv \
    --output_base data \
    --num_samples 2500
```

**예상 소요 시간**: 51-103분 (GPU 성능에 따라)

### 옵션 2: 단계별 수동 실행

각 단계를 개별적으로 실행:

```bash
# 1단계: 배경 추출 (10-20분)
python scripts/extract_clean_backgrounds.py \
    --train_csv train.csv \
    --image_dir train_images \
    --output_dir data/backgrounds

# 2단계: 템플릿 구축 (1-2분)
python scripts/build_defect_templates.py \
    --roi_metadata data/processed/roi_patches/roi_metadata.csv \
    --output_dir data/defect_templates

# 3단계: 데이터 생성 (30-60분)
python scripts/generate_augmented_data.py \
    --model_path outputs/controlnet_training/best.pth \
    --backgrounds_dir data/backgrounds \
    --templates_dir data/defect_templates \
    --output_dir data/augmented \
    --num_samples 2500

# 4단계: 검증 (5-10분)
python scripts/validate_augmented_quality.py \
    --augmented_dir data/augmented \
    --output_dir data/augmented/validation

# 5단계: 병합 (5-10분)
python scripts/merge_datasets.py \
    --original_csv train.csv \
    --augmented_dir data/augmented \
    --output_csv data/final_dataset/train_augmented.csv
```

### 옵션 3: 소규모 테스트 실행

먼저 소규모 샘플로 테스트:

```bash
python scripts/run_augmentation_pipeline.py \
    --train_csv train.csv \
    --image_dir train_images \
    --model_path outputs/controlnet_training/best.pth \
    --roi_metadata data/processed/roi_patches/roi_metadata.csv \
    --num_samples 100 \
    --batch_size 2
```

## 주요 출력 파일

성공적인 실행 후:

```
data/
├── backgrounds/background_inventory.json    # ~3,000-5,000개 배경
├── defect_templates/templates_metadata.json # ~1,000-3,000개 템플릿
├── augmented/
│   ├── images/                              # 2,500개 증강 이미지
│   ├── masks/                               # 2,500개 증강 마스크
│   ├── augmented_metadata.json              # 생성 메타데이터
│   └── validation/
│       ├── quality_scores.json              # 품질 점수
│       └── validation_statistics.json       # 통과/실패 통계
└── final_dataset/
    ├── train_augmented.csv                  # 14,318-14,693개 총 샘플
    └── dataset_statistics.txt               # 종합 통계
```

## 검증

파이프라인 성공 확인:

```bash
# 증강 이미지 수 확인
ls data/augmented/images/ | wc -l  # 2500이어야 함

# 품질 통계 확인
cat data/augmented/validation/validation_statistics.json

# 최종 데이터셋 크기 확인
wc -l data/final_dataset/train_augmented.csv  # ~14,319-14,694이어야 함

# 데이터셋 통계 확인
cat data/final_dataset/dataset_statistics.txt
```

## 시각화

증강 샘플을 시각적으로 검사:

```bash
# 20개 무작위 샘플 보기
python scripts/visualize_augmented_samples.py \
    --augmented_dir data/augmented \
    --output_dir visualizations \
    --num_samples 20

# 최고/최저 품질 샘플 보기
python scripts/visualize_augmented_samples.py \
    --augmented_dir data/augmented \
    --output_dir visualizations \
    --show_best 10 \
    --show_worst 10

# 분포 보기
python scripts/visualize_augmented_samples.py \
    --augmented_dir data/augmented \
    --output_dir visualizations \
    --distributions
```

## 테스트

단위 테스트 실행:

```bash
# 모든 테스트 실행
python tests/test_augmentation_pipeline.py

# 또는 pytest 사용 (설치된 경우)
pytest tests/test_augmentation_pipeline.py -v
```

## 훈련에서 증강 데이터 사용

훈련 스크립트에서 병합된 데이터셋 로드:

```python
import pandas as pd

# 증강 데이터셋 로드
df = pd.read_csv('data/final_dataset/train_augmented.csv')

# 이미지는 두 디렉토리에 있음:
# - 원본: train_images/
# - 증강: data/augmented/images/

# 표준 훈련 파이프라인 사용
# CSV 형식은 원본 train.csv와 동일
```

## 문제 해결

### CUDA 메모리 부족
```bash
# 배치 크기 감소
python scripts/run_augmentation_pipeline.py ... --batch_size 2

# 또는 CPU 사용 (느림)
python scripts/run_augmentation_pipeline.py ... --device cpu
```

### 낮은 품질 통과율 (<60%)
```bash
# 품질 임계값 낮추기
python scripts/validate_augmented_quality.py --min_quality_score 0.6

# 또는 검증 건너뛰기 (권장하지 않음)
python scripts/run_augmentation_pipeline.py ... --skip_quality_checks
```

### 호환 배경을 찾을 수 없음
```bash
# 적합성 임계값 낮추기
python scripts/run_augmentation_pipeline.py ... --min_suitability 0.5

# 더 많은 배경 추출
python scripts/run_augmentation_pipeline.py ... --patches_per_image 10
```

## 다음 단계

1. **소규모 샘플로 테스트** (~100개 샘플) 파이프라인 작동 확인
2. **품질 리포트 검토** ControlNet 모델 품질 평가
3. **매개변수 조정** 필요 시 (임계값, 스케일 범위 등)
4. **전체 증강 실행** 2,500개 샘플로
5. **탐지 모델 훈련** train_augmented.csv 사용
6. **개선 평가** 검증 세트에서

## 문서

자세한 정보:

- **전체 가이드**: `AUGMENTATION_PIPELINE_GUIDE.md` (70페이지 이상)
  - 아키텍처 상세
  - 매개변수 조정
  - 성능 벤치마크
  - 고급 구성

- **스크립트 도움말**:
  ```bash
  python scripts/run_augmentation_pipeline.py --help
  python scripts/generate_augmented_data.py --help
  python scripts/visualize_augmented_samples.py --help
  ```

## 설계 결정

요구사항에 따른 주요 제약:

- ✅ **회전 없음** - 결함은 방향 유지
- ✅ **밝기 조정 없음** - 색상 일관성 유지
- ✅ **80-100% 크기 스케일링** - 축소만, 확대 없음
- ✅ **품질 임계값 0.7** - 균형 잡힌 필터링
- ✅ **클래스 균형** - 클래스당 동일한 샘플 (~625개)
- ✅ **물리적 인식** - 호환 가능한 배경에만 결함 배치

## 프로젝트 구조

```
severstal-steel-defect-detection/
├── AUGMENTATION_PIPELINE_GUIDE.md        # 70페이지 상세 가이드
├── README_AUGMENTATION.md                # 이 파일
├── scripts/
│   ├── extract_clean_backgrounds.py      # 1단계
│   ├── build_defect_templates.py         # 2단계
│   ├── generate_augmented_data.py        # 3단계 (핵심)
│   ├── validate_augmented_quality.py     # 4단계
│   ├── merge_datasets.py                 # 5단계
│   ├── run_augmentation_pipeline.py      # 자동 실행
│   └── visualize_augmented_samples.py    # 시각화
├── tests/
│   └── test_augmentation_pipeline.py     # 단위 테스트
├── src/
│   ├── analysis/                         # 결함 & 배경 분석
│   ├── preprocessing/                    # 힌트 & 프롬프트 생성
│   └── utils/                            # RLE 유틸리티
└── data/                                 # 출력 디렉토리
```

## 성능 예상

| 단계 | 소요 시간 | 출력 |
|------|----------|------|
| 1단계 | 10-20분 | ~3,000-5,000개 배경 |
| 2단계 | 1-2분 | ~1,000-3,000개 템플릿 |
| 3단계 | 30-60분 | 2,500개 증강 샘플 |
| 4단계 | 5-10분 | 70-85% 통과율 예상 |
| 5단계 | 5-10분 | ~14,318-14,693개 총 샘플 |
| **합계** | **51-103분** | **~20% 증강** |

*RTX 3060 GPU 기준 시간, 하드웨어에 따라 달라질 수 있음*

## 상태

**구현**: ✅ 완료 (100%)
- 5단계 모두 구현
- 자동 실행 스크립트 준비
- 시각화 도구 준비
- 단위 테스트 준비
- 문서 완료

**실행**: ⏳ 대기 중
- train.csv 및 ControlNet 모델 필요
- extract_rois.py의 ROI 메타데이터 필요
- 사전 요구사항 준비 시 실행 가능

**테스트**: 📋 예정
- 소규모 테스트 실행 (100개 샘플)
- 전체 프로덕션 실행 (2,500개 샘플)
- 품질 평가
- 증강 데이터로 모델 훈련

## 문의

문제나 질문이 있으면 다음을 참조:
- `AUGMENTATION_PIPELINE_GUIDE.md` 상세 문서
- `tests/test_augmentation_pipeline.py` 사용 예제
- 스크립트 도움말 메시지: `python script.py --help`
