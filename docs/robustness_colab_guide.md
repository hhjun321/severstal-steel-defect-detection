# Robustness Experiments Colab 실행 가이드

실행 순서: **P3 → P1 → P2**

> P3는 학습 없이 이미지 품질 지표만 계산 → 가장 빠름.  
> P1은 9번 학습 (seed 3 × group 3) → T4 기준 약 10–15시간.  
> P2는 P1 결과 + P3 FID 결과를 받아 통계 검정 → 수 초.

---

## 0. 환경 설정

### 0-1. Drive 마운트 & 레포 클론

```python
# 셀 1: Drive 마운트
from google.colab import drive
drive.mount('/content/drive')
```

```python
# 셀 2: 레포 클론 (없으면 클론, 있으면 pull)
import os
if not os.path.isdir('/content/CASDA'):
    os.system('git clone https://github.com/<your-repo>/severstal-steel-defect-detection.git /content/CASDA')
else:
    os.system('git -C /content/CASDA pull')
```

### 0-2. 경로 환경변수 설정

> **이 셀을 런타임 재시작 후 항상 먼저 실행하세요.**  
> `os.environ`으로 설정한 변수는 같은 세션의 `!` 명령에서 `$VAR`로 참조됩니다.

```python
# 셀 3: 경로 환경변수 (실제 경로에 맞게 수정)
import os

DRIVE = '/content/drive/MyDrive/data/Severstal'

os.environ['SCRIPTS']           = '/content/CASDA/scripts'
os.environ['CONFIG']            = '/content/CASDA/configs/benchmark_experiment.yaml'
os.environ['DRIVE']             = DRIVE
os.environ['TRAIN_IMAGES']      = f'{DRIVE}/train_images'
os.environ['TRAIN_CSV']         = f'{DRIVE}/train.csv'
os.environ['AUG_DATASET']       = f'{DRIVE}/augmented_dataset'       # casda_composed/, copypaste_baseline/ 포함
os.environ['AUG_IMAGES']        = f'{DRIVE}/augmented_images_v5.5'   # generated/ 포함
os.environ['ROI_DIR']           = f'{DRIVE}/roi_patches_v5.1'        # roi_metadata.csv 포함
os.environ['YOLO_DATASETS']     = f'{DRIVE}/yolo_datasets'
os.environ['BENCHMARK_RESULTS'] = f'{DRIVE}/benchmark_results'
os.environ['FID_RESULTS']       = f'{DRIVE}/fid_results'
os.environ['LOCAL_IMAGES']      = '/content/dataset_local/train_images'

print("환경변수 설정 완료:")
for k in ['SCRIPTS', 'DRIVE', 'AUG_DATASET', 'AUG_IMAGES', 'ROI_DIR',
          'BENCHMARK_RESULTS', 'FID_RESULTS']:
    print(f"  {k} = {os.environ[k]}")
```

### 0-3. 의존성 설치

```python
# 셀 4: P3 LPIPS 계산용 라이브러리
!pip install lpips -q
```

---

## P3 — 이미지 품질 지표 (KID / LPIPS / FID)

> **학습 없음.** 기존 생성 이미지 + ROI 메타데이터만 있으면 실행 가능.

### P3-1. KID + LPIPS 계산 (CASDA 생성 ROI vs Real ROI)

```python
# 셀 P3-1
!python $SCRIPTS/run_image_quality_metrics.py \
  --casda-roi-dir $AUG_IMAGES/generated \
  --roi-meta      $ROI_DIR/roi_metadata.csv \
  --metrics       kid lpips \
  --output-dir    $FID_RESULTS \
  --cache-dir     $FID_RESULTS/kid_cache \
  --device        cuda \
  --kid-subsets   10 \
  --kid-subset-size 1000 \
  --lpips-pairs   500 \
  --lpips-img-size 64
```

**출력:**
```
fid_results/
  kid_results.json            # KID (전체 + Class별, ×10⁻³)
  lpips_results.json          # LPIPS realism + diversity (Class별)
  quality_metrics_table.md    # 논문 삽입용 통합 테이블
  quality_metrics_table.tex   # LaTeX 버전
  kid_cache/                  # InceptionV3 feature 디스크 캐시
```

### P3-2. FID 계산 — CASDA (H5 가설 검정용)

> `fid_results/fid_results.json`이 이미 있으면 건너뜀.

```python
# 셀 P3-2
!python $SCRIPTS/run_fid.py \
  --config     $CONFIG \
  --data-dir   $TRAIN_IMAGES \
  --csv        $TRAIN_CSV \
  --casda-dir  $AUG_DATASET \
  --output-dir $FID_RESULTS \
  --mode       composed \
  --per-class \
  --device     cuda
```

### P3-3. FID 계산 — CopyPaste (H5 가설 검정용)

```python
# 셀 P3-3
!python $SCRIPTS/run_fid.py \
  --config     $CONFIG \
  --data-dir   $TRAIN_IMAGES \
  --csv        $TRAIN_CSV \
  --casda-dir  $AUG_DATASET \
  --casda-dir-override $AUG_DATASET/copypaste_baseline \
  --output-dir $FID_RESULTS/copypaste \
  --mode       composed \
  --per-class \
  --device     cuda
```

> 출력: `fid_results/copypaste/fid_results.json` → P2의 `--copypaste-fid-results` 입력

---

## P1 — Multi-Seed 반복 실험

> **이 단계가 가장 오래 걸립니다.**  
> seed 3개 × group 3개 = 9번 학습. T4 기준 약 13–18시간.

### P1-1. 학습 이미지 로컬 디스크 복사 (Drive I/O 병목 해소)

```python
# 셀 P1-1
!mkdir -p $LOCAL_IMAGES
!rsync -a --progress $TRAIN_IMAGES/ $LOCAL_IMAGES/
```

### P1-2. Seed 3개 × 3 그룹 학습

```python
# 셀 P1-2-seed42
!python $SCRIPTS/run_benchmark.py \
  --config    $CONFIG \
  --data-dir  $LOCAL_IMAGES \
  --groups    baseline_raw casda_composed_pruning copypaste \
  --casda-dir $AUG_DATASET \
  --yolo-dir  $YOLO_DATASETS \
  --seed      42 \
  --no-fid \
  --output-dir $BENCHMARK_RESULTS/multiseed/seed_42
```

```python
# 셀 P1-2-seed123
!python $SCRIPTS/run_benchmark.py \
  --config    $CONFIG \
  --data-dir  $LOCAL_IMAGES \
  --groups    baseline_raw casda_composed_pruning copypaste \
  --casda-dir $AUG_DATASET \
  --yolo-dir  $YOLO_DATASETS \
  --seed      123 \
  --no-fid \
  --output-dir $BENCHMARK_RESULTS/multiseed/seed_123
```

```python
# 셀 P1-2-seed456
!python $SCRIPTS/run_benchmark.py \
  --config    $CONFIG \
  --data-dir  $LOCAL_IMAGES \
  --groups    baseline_raw casda_composed_pruning copypaste \
  --casda-dir $AUG_DATASET \
  --yolo-dir  $YOLO_DATASETS \
  --seed      456 \
  --no-fid \
  --output-dir $BENCHMARK_RESULTS/multiseed/seed_456
```

> **중단 후 재시작 시** `--resume` 추가:
> ```python
> !python $SCRIPTS/run_benchmark.py ... --resume \
>   --output-dir $BENCHMARK_RESULTS/multiseed/seed_42
> ```

### P1-3. Seed 결과 집계 (mean ± std 계산)

```python
# 셀 P1-3
!python $SCRIPTS/aggregate_multiseed_results.py \
  --results-dirs \
    $BENCHMARK_RESULTS/multiseed/seed_42 \
    $BENCHMARK_RESULTS/multiseed/seed_123 \
    $BENCHMARK_RESULTS/multiseed/seed_456 \
  --output-dir $BENCHMARK_RESULTS/multiseed_aggregated
```

**출력:**
```
benchmark_results/multiseed_aggregated/
  aggregated_results.json   # (model, dataset) 별 seed값 + mean/std
  table_mean_std.md         # Markdown 테이블
  table_mean_std.tex        # LaTeX 테이블 (논문용)
```

---

## P2 — 통계 유의성 검정

> P1-3과 P3-2, P3-3이 완료된 후 실행.

### P2-1. 가설 검정 실행 (H3–H6)

```python
# 셀 P2-1
!python $SCRIPTS/run_statistical_tests.py \
  --aggregated-results    $BENCHMARK_RESULTS/multiseed_aggregated/aggregated_results.json \
  --fid-results           $FID_RESULTS/fid_results.json \
  --copypaste-fid-results $FID_RESULTS/copypaste/fid_results.json \
  --output-dir            $BENCHMARK_RESULTS/statistical_tests \
  --alpha 0.05
```

> `--copypaste-fid-results`가 없으면 H5는 N/A로 처리됨.

**출력:**
```
benchmark_results/statistical_tests/
  hypothesis_test_results.json   # 원시 검정 결과 (stat, p_raw, p_adj, Cohen's d)
  significance_table.md          # 논문 삽입용 요약 테이블
  significance_table.tex         # LaTeX 버전
```

---

## 결과 파일 위치 요약

| 단계 | 파일 | Drive 경로 |
|------|------|------------|
| P3 | KID 결과 | `fid_results/kid_results.json` |
| P3 | LPIPS 결과 | `fid_results/lpips_results.json` |
| P3 | 품질 지표 테이블 | `fid_results/quality_metrics_table.md/tex` |
| P3 | CASDA FID | `fid_results/fid_results.json` |
| P3 | CopyPaste FID | `fid_results/copypaste/fid_results.json` |
| P1 | seed별 학습 결과 | `benchmark_results/multiseed/seed_{42,123,456}/` |
| P1 | 집계 결과 | `benchmark_results/multiseed_aggregated/aggregated_results.json` |
| P1 | mean±std 테이블 | `benchmark_results/multiseed_aggregated/table_mean_std.md/tex` |
| P2 | 검정 결과 | `benchmark_results/statistical_tests/hypothesis_test_results.json` |
| P2 | 유의성 테이블 | `benchmark_results/statistical_tests/significance_table.md/tex` |

---

## 참고

### n 수가 작은 경우 주의

Wilcoxon 검정의 최소 p-value는 n에 따라 제한된다.

| n | 최소 p-value (단측) |
|---|-------------------|
| 3 | 0.125 |
| 9 (3 seed × 3 model) | ~0.004 |

H4, H6는 3 seed × 3 모델 = **n=9 관측값** → α=0.05 달성 가능.

### KID 해석

- KID는 ×10⁻³ 단위로 보고 (예: `3.21 ± 0.45`)
- 낮을수록 생성 분포가 실제 분포와 가까움
- CopyPaste는 실제 패치를 복사하므로 FID/KID ≈ 0 → **LPIPS realism이 경계 아티팩트를 정량화하는 핵심 지표**

### 스크립트 파일 위치 (`/content/CASDA/scripts/`)

| 스크립트 | 역할 |
|----------|------|
| `run_image_quality_metrics.py` | P3: KID + LPIPS |
| `run_fid.py` | P3: FID (CASDA / CopyPaste) |
| `run_benchmark.py` | P1: 학습 (`--seed`, `--no-fid`, `--resume`) |
| `aggregate_multiseed_results.py` | P1: seed 집계 → mean±std |
| `run_statistical_tests.py` | P2: 가설 검정 H3–H6 |
