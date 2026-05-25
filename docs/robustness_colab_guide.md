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
# 셀 P3-2  (H5 per-class 비교용 → per_class 유지)
# --workers 0  : Colab multiprocessing 오류 방지
# --max-images 500 : 기본 1000 → 500으로 줄여 약 2배 속도 향상 (FID 신뢰도 충분)
# --batch-size 128 : GPU 메모리 여유 시 64→128 (InceptionV3 추론 2배 빠름)
!python $SCRIPTS/run_fid.py \
  --config      $CONFIG \
  --data-dir    $TRAIN_IMAGES \
  --csv         $TRAIN_CSV \
  --casda-dir   $AUG_DATASET \
  --output-dir  $FID_RESULTS \
  --fid-mode    composed \
  --max-images  500 \
  --batch-size  128 \
  --workers     0 \
  --device      cuda
```

> **per-class가 필요 없으면** (H5 검정 생략 시) `--no-per-class` 추가 → 약 4배 빠름:
> ```python
> !python $SCRIPTS/run_fid.py ... --no-per-class
> ```

### P3-3. FID 계산 — CopyPaste (H5 가설 검정용)

```python
# 셀 P3-3
!python $SCRIPTS/run_fid.py \
  --config              $CONFIG \
  --data-dir            $TRAIN_IMAGES \
  --csv                 $TRAIN_CSV \
  --casda-dir           $AUG_DATASET \
  --casda-composed-dir  $AUG_DATASET/copypaste_baseline \
  --output-dir          $FID_RESULTS/copypaste \
  --fid-mode            composed \
  --max-images          500 \
  --batch-size          128 \
  --workers             0 \
  --device              cuda
```

> 출력: `fid_results/copypaste/fid_results.json` (참고용; H5는 이제 LPIPS 기반)

### P3-4. CopyPaste LPIPS 저장 (H5 가설 검정용)

> P3-1 이후 CopyPaste에 대해 동일한 LPIPS 계산을 수행한다.  
> CopyPaste의 "생성" 패치는 `copypaste_baseline/` 폴더의 합성 이미지에서 추출된 ROI를 사용한다.

```python
# 셀 P3-4: CopyPaste LPIPS 계산
# CopyPaste composed 이미지를 --casda-roi-dir 로 지정
# (copypaste_baseline/ 이 CASDA generated/ 와 동일한 구조이면 그대로 사용)
!python $SCRIPTS/run_image_quality_metrics.py \
  --casda-roi-dir $AUG_DATASET/copypaste_baseline \
  --roi-meta      $ROI_DIR/roi_metadata.csv \
  --metrics       lpips \
  --output-dir    $FID_RESULTS/copypaste \
  --device        cuda \
  --lpips-pairs   500 \
  --lpips-img-size 64
```

> 출력: `fid_results/copypaste/lpips_results.json` → P2의 `--copypaste-lpips-results` 입력

**copypaste_baseline/ 구조가 다를 경우 대안 (이미 실험값을 알고 있는 경우):**

```python
# 셀 P3-4b: 기존 실험값에서 JSON 직접 생성
import json, os
# P3 실험에서 직접 얻은 CopyPaste LPIPS 값 (2026-05-23 Colab 실험 결과)
copypaste_lpips = {
    "realism": {
        "per_class": {"Class1": 0.4975, "Class2": 0.4280, "Class3": 0.4838, "Class4": 0.4993},
        "overall": 0.4907
    },
    "diversity": {
        "per_class": {"Class1": 0.3984, "Class2": 0.3861, "Class3": 0.4119, "Class4": 0.3940},
        "overall": 0.3937
    }
}
out_dir = os.environ['FID_RESULTS'] + '/copypaste'
os.makedirs(out_dir, exist_ok=True)
with open(f"{out_dir}/lpips_results.json", 'w') as f:
    json.dump(copypaste_lpips, f, indent=2)
print(f"Saved: {out_dir}/lpips_results.json")
```

---

## P1 — Multi-Seed 반복 실험

> **이 단계가 가장 오래 걸립니다.**  
> model 2개(yolo_mfd, eb_yolov8) × group 3개 × seed 3개 = 18번 학습. T4 기준 약 9–12시간.  
> DeepLabV3+는 논문에서 제거됨 — Detection 2종(YOLO-MFD, EB-YOLOv8)만 사용.  
> H3은 Friedman 대신 방향 일치 정성 확인으로 대체 — §참고 참조.

### P1-1. 학습 이미지 로컬 디스크 복사 (Drive I/O 병목 해소)

```python
# 셀 P1-1
!mkdir -p $LOCAL_IMAGES
!rsync -a --progress $TRAIN_IMAGES/ $LOCAL_IMAGES/
```

### P1-2. Seed 3개 × 3 그룹 학습

```python
# 셀 P1-2-seed42
# --models yolo_mfd eb_yolov8: 논문 대상 모델 2종 (DeepLabV3+는 논문에서 제거됨)
!python $SCRIPTS/run_benchmark.py \
  --config    $CONFIG \
  --data-dir  $LOCAL_IMAGES \
  --groups    baseline_raw casda_composed_pruning copypaste \
  --models    yolo_mfd eb_yolov8 \
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
  --models    yolo_mfd eb_yolov8 \
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
  --models    yolo_mfd eb_yolov8 \
  --casda-dir $AUG_DATASET \
  --yolo-dir  $YOLO_DATASETS \
  --seed      456 \
  --no-fid \
  --output-dir $BENCHMARK_RESULTS/multiseed/seed_456
```

> **중단 후 재시작 시** `--resume` 추가:
> ```python
> !python $SCRIPTS/run_benchmark.py ... \
>   --models yolo_mfd eb_yolov8 --resume \
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
# H5 is LPIPS realism superiority (FID is biased toward CopyPaste — see p3_image_quality_results.md)
!python $SCRIPTS/run_statistical_tests.py \
  --aggregated-results      $BENCHMARK_RESULTS/multiseed_aggregated/aggregated_results.json \
  --lpips-results           $FID_RESULTS/lpips_results.json \
  --copypaste-lpips-results $FID_RESULTS/copypaste_lpips_results.json \
  --output-dir              $BENCHMARK_RESULTS/statistical_tests \
  --alpha 0.05
```

> `--copypaste-lpips-results`가 없으면 H5는 N/A로 처리됨.  
> H5 재정의: FID(CASDA) < FID(CopyPaste) → LPIPS Realism(CASDA) < LPIPS Realism(CopyPaste).  
> CopyPaste는 실제 패치를 복사하므로 FID ≈ 0이 되어 단독 평가 지표로 부적합함.

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
| P3 | CASDA LPIPS 결과 | `fid_results/lpips_results.json` |
| P3 | 품질 지표 테이블 | `fid_results/quality_metrics_table.md/tex` |
| P3 | CASDA FID | `fid_results/fid_results.json` |
| P3 | CopyPaste FID | `fid_results/copypaste/fid_results.json` |
| P3 | CopyPaste LPIPS (H5용) | `fid_results/copypaste/lpips_results.json` |
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
| 6 (3 seed × 2 model) | ~0.016 |
| 9 (3 seed × 3 model) | ~0.004 |

DeepLabV3+ 제외 후: H4, H6는 3 seed × 2 모델 = **n=6 관측값** → 단측 최소 p≈0.016, α=0.05 달성 가능.

**H3 Friedman 검정 재설계 필요:**  
Friedman test는 3개 이상의 그룹(모델)이 필요하다. 모델이 2개(yolo_mfd, eb_yolov8)로 줄어들면 Friedman 대신 **Wilcoxon signed-rank (paired)** 로 대체한다.  
비교 쌍: (casda − baseline) 차이를 yolo_mfd vs eb_yolov8 간 paired 검정.  
H3 주장("아키텍처 독립적 향상")은 두 모델 모두에서 방향이 일치하는지 확인으로 서술 가능.

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
