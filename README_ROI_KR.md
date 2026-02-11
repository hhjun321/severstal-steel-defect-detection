# ROI 추출 연구 구현

본 저장소는 Severstal Steel Defect Detection 데이터셋을 위해 `PROJECT(roi).md`에 설명된 통계적 지표 기반 ROI 추출 파이프라인을 구현합니다.

## 개요

`PROJECT(roi).md`의 핵심 통찰은 **고품질 증강 데이터는 "어떤 결함 하위 클래스"와 "어떤 배경 컨텍스트"를 매칭하는 것에 달려있다**는 것입니다. 본 구현은 결함이 위치한 곳뿐만 아니라 배경이 합성 결함 생성에 적합한지를 평가합니다.

## 연구 파이프라인

### 1. 결함 특성화 (4가지 통계적 지표)

**모듈**: `src/analysis/defect_characterization.py`

각 결함의 기하학적 속성을 계산합니다:

- **Linearity (직선성)**: 고유값 분석을 사용한 길쭉함 측정 (0-1)
- **Solidity (치밀도)**: 결함 면적 대 볼록 껍질 면적 비율 (0-1)
- **Extent (분산도)**: 결함 면적 대 경계 상자 면적 비율 (0-1)
- **Aspect Ratio (종횡비)**: 주축 / 부축 길이 (≥1)

이러한 지표를 기반으로 결함은 다음 하위 유형으로 분류됩니다:
- `linear_scratch`: 높은 직선성 + 높은 종횡비
- `elongated`: 높은 종횡비 + 중간 직선성
- `compact_blob`: 낮은 종횡비 + 높은 치밀도
- `irregular`: 낮은 치밀도
- `general`: 기본 분류

### 2. 배경 특성화 (그리드 기반 분석)

**모듈**: `src/analysis/background_characterization.py`

이미지를 64×64 그리드로 나누고 각 패치를 분류합니다:

**1단계: 분산 분석**
- 낮은 분산 → `smooth` (평평한 금속 표면)
- 높은 분산 → 경계선 분석으로 진행

**2단계: 경계선 방향 분석 (Sobel 필터)**
- 지배적인 수직 경계선 → `vertical_stripe`
- 지배적인 수평 경계선 → `horizontal_stripe`
- 다방향 경계선 → `complex_pattern`
- 지배적인 방향 없음 → `textured`

**3단계: 주파수 분석 (FFT)**
- 고주파 비율은 텍스처 복잡도를 나타냄

각 그리드 셀은 다음을 받습니다:
- 배경 유형 분류
- 안정성 점수 (0-1): 배경의 균일성/일관성

### 3. ROI 적합성 평가

**모듈**: `src/analysis/roi_suitability.py`

사전 정의된 규칙을 사용하여 결함-배경 매칭을 평가합니다:

| 결함 하위 유형 | 최적 배경 | 점수 |
|----------------|----------|------|
| `linear_scratch` | `vertical_stripe`, `horizontal_stripe` | 1.0 |
| `compact_blob` | `smooth` | 1.0 |
| `irregular` | `complex_pattern` | 1.0 |

**적합성 점수** = 0.5×매칭 + 0.3×연속성 + 0.2×안정성

- **매칭 점수**: 결함 유형이 배경 유형에 얼마나 잘 맞는지
- **연속성 점수**: ROI 경계 상자 내 배경 균일성
- **안정성 점수**: 평균 배경 안정성

**권장사항**:
- `suitable` (적합): 적합성 ≥ 0.7
- `acceptable` (허용): 0.5 ≤ 적합성 < 0.7
- `unsuitable` (부적합): 적합성 < 0.5

### 4. ROI 위치 최적화

결함 중심이 배경 경계(불연속성) 근처에 있으면 ROI 윈도우를 (최대 32픽셀) 이동하여 결함을 경계 내에 유지하면서 배경 연속성을 최대화합니다.

### 5. 데이터 패키징

**모듈**: `src/preprocessing/roi_extraction.py`

각 ROI에 대한 최종 출력:
- ROI 이미지 패치 (512×512)
- ROI 마스크 패치 (512×512)
- 메타데이터 CSV:
  - 이미지 ID, 클래스 ID, 영역 ID
  - 결함 메트릭 (직선성, 치밀도, 분산도, 종횡비)
  - 배경 유형 및 안정성
  - 적합성 점수
  - ControlNet 훈련을 위한 텍스트 프롬프트

## 프로젝트 구조

```
severstal-steel-defect-detection/
├── src/
│   ├── utils/
│   │   └── rle_utils.py              # RLE 인코딩/디코딩
│   ├── analysis/
│   │   ├── defect_characterization.py    # 1단계: 결함 분석
│   │   ├── background_characterization.py # 2단계: 배경 분석
│   │   └── roi_suitability.py           # 3단계: 매칭 평가
│   └── preprocessing/
│       └── roi_extraction.py           # 4-5단계: ROI 추출 파이프라인
├── scripts/
│   └── extract_rois.py                # 메인 실행 스크립트
├── data/
│   └── processed/
│       └── roi_patches/               # 출력 디렉토리
│           ├── images/                # ROI 이미지 패치
│           ├── masks/                 # ROI 마스크 패치
│           ├── roi_metadata.csv       # 전체 메타데이터
│           └── statistics.txt         # 요약 통계
├── train_images/                      # 훈련 이미지 (1600×256)
├── test_images/                       # 테스트 이미지
├── train.csv                          # RLE 인코딩된 어노테이션
└── PROJECT(roi).md                    # 연구 문서
```

## 사용법

### 의존성 설치

```bash
pip install -r requirements.txt
```

### 데이터셋에서 ROI 추출

**10개 이미지로 테스트:**
```bash
python scripts/extract_rois.py --max_images 10
```

**모든 이미지 처리:**
```bash
python scripts/extract_rois.py
```

**사용자 정의 매개변수:**
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

**메타데이터만 (패치 없이):**
```bash
python scripts/extract_rois.py --no_save_patches --max_images 100
```

## 출력

### 1. ROI 메타데이터 CSV (`roi_metadata.csv`)

열:
- `image_id`, `class_id`, `region_id`: 식별자
- `roi_bbox`: (x1, y1, x2, y2) 최적화된 ROI 위치
- `defect_bbox`: (x1, y1, x2, y2) 원본 결함 경계 상자
- `centroid`: (x, y) 결함 중심
- `area`: 픽셀 단위 결함 면적
- `linearity`, `solidity`, `extent`, `aspect_ratio`: 4가지 지표
- `defect_subtype`: 결함 분류
- `background_type`: 배경 분류
- `suitability_score`: 전체 매치 품질 (0-1)
- `matching_score`: 결함-배경 매칭 (0-1)
- `continuity_score`: 배경 균일성 (0-1)
- `stability_score`: 배경 안정성 (0-1)
- `recommendation`: `suitable` / `acceptable` / `unsuitable`
- `prompt`: ControlNet을 위한 텍스트 설명
- `roi_image_path`: 저장된 이미지 패치 경로
- `roi_mask_path`: 저장된 마스크 패치 경로

### 2. 이미지/마스크 패치

PNG 파일로 저장:
- 이미지: `{image_id}_class{class_id}_region{region_id}.png`
- 마스크: 동일한 명명 규칙

### 3. 통계 요약

출력 예시:
```
추출된 총 ROI 수: 1234

클래스별 ROI:
  클래스 1: 456
  클래스 2: 321
  클래스 3: 289
  클래스 4: 168

결함 하위 유형별 ROI:
  linear_scratch: 512
  compact_blob: 334
  elongated: 245
  irregular: 143

배경 유형별 ROI:
  smooth: 423
  vertical_stripe: 389
  horizontal_stripe: 267
  textured: 155

적합성 분포:
  suitable (≥0.7): 756 (61.3%)
  acceptable (0.5-0.7): 389 (31.5%)
  unsuitable (<0.5): 89 (7.2%)
```

## 매개변수

### 명령줄 인수

| 인수 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--image_dir` | str | train_images | 훈련 이미지 디렉토리 |
| `--train_csv` | str | train.csv | RLE 어노테이션 CSV |
| `--output_dir` | str | data/processed/roi_patches | 출력 디렉토리 |
| `--roi_size` | int | 512 | ROI 패치 크기 (정사각형) |
| `--grid_size` | int | 64 | 배경 분석 그리드 크기 |
| `--min_suitability` | float | 0.5 | 최소 적합성 임계값 |
| `--max_images` | int | None | 처리할 최대 이미지 수 |
| `--no_save_patches` | flag | False | 이미지/마스크 패치 저장 건너뛰기 |
| `--verbose` | flag | False | 상세 로깅 활성화 |

## 구현 세부사항

### 결함 하위 유형 분류 규칙

```python
if linearity > 0.7 and aspect_ratio > 3.0:
    subtype = "linear_scratch"
elif aspect_ratio > 2.0 and linearity < 0.7:
    subtype = "elongated"
elif aspect_ratio < 2.0 and solidity > 0.7:
    subtype = "compact_blob"
elif solidity < 0.5:
    subtype = "irregular"
else:
    subtype = "general"
```

### 배경 분류 알고리즘

```python
# 1단계: 분산 검사
if variance < smooth_threshold:
    return "smooth"

# 2단계: 경계선 방향 분석
vertical_edges = sobel_vertical(patch)
horizontal_edges = sobel_horizontal(patch)

if vertical_edges > horizontal_edges * edge_ratio:
    return "vertical_stripe"
elif horizontal_edges > vertical_edges * edge_ratio:
    return "horizontal_stripe"

# 3단계: 주파수 분석
fft_high_freq_ratio = compute_fft_ratio(patch)
if fft_high_freq_ratio > complex_threshold:
    return "complex_pattern"
else:
    return "textured"
```

### 적합성 점수 계산

```python
# 매칭 점수: 결함-배경 호환성
matching_rules = {
    ("linear_scratch", "vertical_stripe"): 1.0,
    ("linear_scratch", "horizontal_stripe"): 1.0,
    ("compact_blob", "smooth"): 1.0,
    ("irregular", "complex_pattern"): 1.0,
    # ... 기타 조합은 부분 점수
}

matching_score = matching_rules.get((defect_subtype, bg_type), 0.5)

# 연속성 점수: ROI 내 배경 균일성
bg_grid_types = [grid.bg_type for grid in roi_grids]
continuity_score = mode_frequency(bg_grid_types)

# 안정성 점수: 평균 그리드 안정성
stability_score = mean([grid.stability for grid in roi_grids])

# 최종 적합성
suitability = 0.5*matching + 0.3*continuity + 0.2*stability
```

## 출력 예시

### ROI 메타데이터 예시

```csv
image_id,class_id,region_id,roi_bbox,defect_bbox,centroid,area,linearity,solidity,extent,aspect_ratio,defect_subtype,background_type,suitability_score,matching_score,continuity_score,stability_score,recommendation,prompt,roi_image_path,roi_mask_path
00a0f53fb,1,0,"[100,50,612,562]","[250,180,450,380]","[350,280]",12450,0.87,0.65,0.31,4.2,linear_scratch,vertical_stripe,0.89,1.0,0.85,0.78,suitable,"a high-linearity linear scratch on vertical striped background",images/00a0f53fb_class1_region0.png,masks/00a0f53fb_class1_region0.png
```

## 성능

### 처리 속도

- **이미지당 평균 시간**: ~2-3초
- **전체 데이터셋 (12,568 이미지)**: ~7-10시간
- **병목 현상**:
  - FFT 계산 (~30% 시간)
  - 그리드 기반 배경 분석 (~40% 시간)
  - 결함 메트릭 계산 (~20% 시간)

### 메모리 사용량

- **이미지당 피크 메모리**: ~50-100 MB
- **권장 시스템 RAM**: 8GB 이상

## 검증 및 품질 관리

### 통계적 검증

스크립트는 자동으로 다음을 확인합니다:
- 클래스 분포 균형
- 결함 하위 유형 커버리지
- 배경 유형 다양성
- 적합성 점수 분포

### 시각적 검사

수동 검증을 위해:
1. 적합성이 높은 샘플 샘플링
2. 경계 케이스 (적합성 ~0.5) 검토
3. 각 결함 하위 유형의 대표 예시 확인

### 알려진 제한사항

1. **그리드 크기 민감도**: 64×64 그리드는 일부 패턴을 놓칠 수 있음
2. **경계 효과**: ROI 가장자리 근처의 결함은 최적으로 배치되지 않을 수 있음
3. **다중 배경**: 단일 ROI 내 혼합 배경은 가장 지배적인 유형으로 분류됨

## 다음 단계

ROI 추출 후:
1. `scripts/prepare_controlnet_data.py`를 사용하여 ControlNet 훈련 데이터 준비
2. Multi-channel hint 이미지 생성
3. ControlNet 모델 훈련
4. 합성 결함 생성으로 데이터셋 증강

## 참조

- **연구 문서**: `PROJECT(roi).md`
- **ControlNet 준비**: `PROJECT(prepare_control).md`
- **구현 요약**: `IMPLEMENTATION_SUMMARY_KR.md`

## 기여

이슈 및 개선 제안을 환영합니다. PR을 제출하기 전에 기존 코드 스타일을 따라주세요.

## 라이선스

본 프로젝트는 Severstal Steel Defect Detection 챌린지의 일부입니다.
