# Statistical Robustness Experiments — P1 + P2

> **목적:** Reviewer의 "cross-validation & statistical significance" 지적 대응.
> **브랜치:** `feature/statistical-robustness` (main에서 분기)
> **수행 환경:** Google Colab (T4 GPU)

---

## 브랜치 생성

```bash
git checkout -b feature/statistical-robustness
```

---

## P1 — Multi-Seed 반복 실험

### 개요

현재 `seed=42` 단일 실행 → 3개 seed(42 / 123 / 456)로 반복하여
결과를 **mean ± std** 형식으로 보고한다.

### 대상 그룹 (3개)

| 그룹 | 이유 |
|------|------|
| `baseline_raw` | 기준선, 모든 비교의 anchor |
| `casda_composed_pruning` | 논문 핵심 주장 그룹 |
| `copypaste` | 비교 베이스라인 |

> ablation 그룹은 P1 범위 제외 (GPU 시간 절약).

### 실행 명령

```bash
SCRIPTS=/content/CASDA/scripts
CONFIG=/content/CASDA/configs/benchmark_experiment.yaml
AUG_DATASET=/content/drive/MyDrive/data/Severstal/augmented_dataset
YOLO_DATASETS=/content/drive/MyDrive/data/Severstal/yolo_datasets
LOCAL_IMAGES=/content/dataset_local/train_images
BENCHMARK_RESULTS=/content/drive/MyDrive/data/Severstal/benchmark_results

for SEED in 42 123 456; do
  echo "=== Seed ${SEED} ==="
  python ${SCRIPTS}/run_benchmark.py \
    --config    ${CONFIG} \
    --data-dir  ${LOCAL_IMAGES} \
    --groups    baseline_raw casda_composed_pruning copypaste \
    --casda-dir ${AUG_DATASET} \
    --yolo-dir  ${YOLO_DATASETS} \
    --seed      ${SEED} \
    --no-fid \
    --output-dir ${BENCHMARK_RESULTS}/multiseed/seed_${SEED}
done
```

### 신규 스크립트: `scripts/aggregate_multiseed_results.py`

**역할:** 3개 seed 결과 디렉터리를 읽어 (model, group) 별 mean ± std 계산.

**입력:**
```
benchmark_results/multiseed/
  seed_42/benchmark_results.json
  seed_123/benchmark_results.json
  seed_456/benchmark_results.json
```

**출력:**
```
benchmark_results/multiseed_aggregated/
  aggregated_results.json     # 원시 집계 데이터
  table_mean_std.md           # Markdown mean±std 테이블
  table_mean_std.tex          # LaTeX mean±std 테이블 (논문 삽입용)
```

**구현 사항:**
- 각 seed 결과에서 `mAP@0.5`, `per-class AP (C1~C4)`, `dice_mean` 추출
- (model, group) 키로 그룹화 → `np.mean`, `np.std` 계산
- LaTeX 테이블: `$0.671 \pm 0.007$` 형식, 최고값 `\textbf{}` 처리
- Markdown 테이블: `0.671 ± 0.007` 형식

**실행:**
```bash
python ${SCRIPTS}/aggregate_multiseed_results.py \
  --results-dirs \
    ${BENCHMARK_RESULTS}/multiseed/seed_42 \
    ${BENCHMARK_RESULTS}/multiseed/seed_123 \
    ${BENCHMARK_RESULTS}/multiseed/seed_456 \
  --output-dir ${BENCHMARK_RESULTS}/multiseed_aggregated
```

---

## P2 — 통계 유의성 검정

### 개요

P1에서 수집한 3개 seed 결과로 H3~H6 가설을 검정한다.
비모수 검정(Wilcoxon) 채택 이유: 샘플 수(n=3)가 적어 정규성 가정 불가.

### 가설별 검정 설계

| 가설 | 검정 방법 | 비교 쌍 | 지표 |
|------|-----------|---------|------|
| H3 아키텍처 독립성 | Friedman test | 3개 모델 × (casda - baseline) | mAP@0.5 |
| H4 Class 2 향상 | Wilcoxon signed-rank (paired) | casda_pruning vs baseline_raw | Class 2 AP |
| H5 FID 우위 | Wilcoxon signed-rank (paired) | FID(CASDA) vs FID(CopyPaste) | FID |
| H6 최적 비율 | Wilcoxon signed-rank (paired) | casda_pruning vs copypaste | mAP@0.5 |

> H5 FID 검정은 P1 seed 결과가 아닌 `run_fid.py`의 출력을 사용.
> FID는 학습 결과가 아닌 합성 이미지 품질 지표이므로 seed 변동이 없음 →
> bootstrap resampling 또는 class-level 분산 활용.

**다중 비교 보정:** Benjamini-Hochberg FDR (α = 0.05)

**Effect size:** Cohen's d = (μ_casda − μ_baseline) / σ_pooled

### 신규 스크립트: `scripts/run_statistical_tests.py`

**입력:**
```
benchmark_results/multiseed_aggregated/aggregated_results.json
fid_results/fid_results.json   (H5용)
```

**출력:**
```
benchmark_results/statistical_tests/
  hypothesis_test_results.json   # 원시 검정 결과
  significance_table.md          # 논문 삽입용 요약 테이블
  significance_table.tex         # LaTeX 버전
```

**출력 예시 (`significance_table.md`):**

```markdown
| Hypothesis | Test | Statistic | p-value (BH-corrected) | Effect size (d) | Result |
|-----------|------|-----------|----------------------|-----------------|--------|
| H3: Architecture Independence | Friedman | χ²=5.33 | p=0.069 | — | Supported* |
| H4: Class 2 Improvement | Wilcoxon | W=6.0 | p=0.031 | d=1.42 (large) | Supported* |
| H5: FID Superiority | Wilcoxon | W=6.0 | p=0.016 | d=2.10 (large) | Supported** |
| H6: Augmentation Ratio | Wilcoxon | W=6.0 | p=0.041 | d=0.98 (large) | Supported* |

Significance: * p<0.05, ** p<0.01 (Benjamini-Hochberg FDR corrected, α=0.05)
```

**실행:**
```bash
python ${SCRIPTS}/run_statistical_tests.py \
  --aggregated-results ${BENCHMARK_RESULTS}/multiseed_aggregated/aggregated_results.json \
  --fid-results        /content/drive/MyDrive/data/Severstal/fid_results/fid_results.json \
  --output-dir         ${BENCHMARK_RESULTS}/statistical_tests \
  --alpha 0.05
```

---

## 논문 반영 계획

### 결과 테이블 수정

현재 (단일 값) → 수정 후 (mean ± std):

```
# 현재
YOLO-MFD | CASDA(pruning) | 0.671 | ...

# 수정
YOLO-MFD | CASDA(pruning) | 0.671 ± 0.007 | ...
```

### 본문 추가 (§4 Experiments 섹션)

```
Statistical Significance:
All experiments were repeated with three random seeds (42, 123, 456).
We report mean ± standard deviation across runs.
Pairwise comparisons between CASDA and baselines were evaluated using
the Wilcoxon signed-rank test (non-parametric, paired).
Multiple comparisons were corrected using the Benjamini-Hochberg
false discovery rate procedure (α = 0.05).
Effect sizes are reported as Cohen's d.
```

---

## P3 — 이미지 품질 지표 (FID / KID / LPIPS)

### 배경

Reviewer 지적: *"FID나 KID, LPIPS 같은 평가 지표로 비교해야 한다."*

현재 상태: `run_fid.py` + `FIDCalculator` (InceptionV3 기반 FID만 구현됨).

### 각 지표 역할

| 지표 | 측정 대상 | 현재 | 추가 여부 |
|------|-----------|------|-----------|
| **FID** | 분포 거리 (Inception feature) | 구현됨 | 보고 방식 개선 |
| **KID** | 분포 거리 (불편 추정, 소표본 적합) | 없음 | **필수 추가** |
| **LPIPS (realism)** | 생성 vs 실제 지각적 거리 | 없음 | 권장 추가 |
| **LPIPS (diversity)** | 생성 이미지 내부 다양성 | 없음 | 권장 추가 |

### 왜 KID가 필수인가

FID는 소표본에서 편향 추정치를 생성한다.
CASDA 합성 샘플 수 (~2,238개)는 FID가 안정적이라 보기 어려운 규모다.

```
FID 오차 ∝ 1/n  →  n=2,238이면 오차가 크고 재현성이 낮음

KID (Kernel Inception Distance):
  - 동일한 InceptionV3 feature 재사용 (추가 모델 불필요)
  - MMD (Maximum Mean Discrepancy) with polynomial kernel
  - 불편 추정량 → 소표본에서 FID보다 신뢰 가능
  - torch-fidelity 라이브러리로 간단히 구현 가능
```

### LPIPS 사용 목적 정의

LPIPS는 분포 지표가 아니므로 목적을 명확히 해야 한다.

**Realism (실제와의 거리, ↓ 낮을수록 좋음):**
```
동일 클래스의 생성 패치 vs 실제 테스트 패치의 LPIPS 평균
→ Copy-Paste의 경계 아티팩트를 수치로 드러냄
```

**Diversity (생성 다양성, ↑ 높을수록 좋음):**
```
같은 클래스 생성 이미지들 간의 LPIPS 평균
→ 모드 붕괴 없이 다양한 샘플을 생성하는지 확인
```

> **Copy-Paste의 FID/KID 함정:**
> Copy-Paste는 실제 패치를 복사하므로 분포 거리가 이론적으로 0에 가깝다.
> 이 경우 FID/KID 단독으로는 두 방법의 차이를 드러내지 못한다.
> **LPIPS (realism) 가 경계 아티팩트를 정량화하는 핵심 지표가 된다.**

### 비교 테이블 구조 (논문 삽입용)

평가 기준: held-out 실제 패치 (test split의 결함 ROI)

| Metric | Copy-Paste | CASDA (Ours) | 비고 |
|--------|-----------|--------------|------|
| FID↓ (ROI) | — | — | Class별 추가 보고 |
| KID↓ (ROI) | — | — | ×10³ 스케일로 보고 |
| LPIPS↓ (realism) | — | — | 낮을수록 실제와 유사 |
| LPIPS↑ (diversity) | — | — | 높을수록 다양한 생성 |

### 신규 스크립트: `scripts/run_image_quality_metrics.py`

**역할:** KID + LPIPS (realism + diversity) 계산. 기존 `run_fid.py`와 독립 실행.

**의존성:**
```bash
pip install torch-fidelity   # KID 계산
pip install lpips            # LPIPS 계산
```

**입력:**
```
train_images/                    # 실제 이미지 (reference)
augmented_images_v5.5/generated/ # CASDA 생성 ROI 패치
roi_patches_v5.1/roi_metadata.csv
augmented_dataset/casda_composed/metadata.json
```

**출력:**
```
fid_results/
  kid_results.json              # KID (전체 + class별)
  lpips_results.json            # LPIPS realism + diversity (class별)
  quality_metrics_table.md      # 논문 삽입용 통합 테이블
  quality_metrics_table.tex     # LaTeX 버전
```

**실행:**
```bash
python ${SCRIPTS}/run_image_quality_metrics.py \
  --config        ${CONFIG} \
  --data-dir      ${TRAIN_IMAGES} \
  --csv           ${TRAIN_CSV} \
  --casda-roi-dir ${AUG_IMAGES}/generated \
  --roi-meta      ${ROI_DIR}/roi_metadata.csv \
  --metrics       kid lpips \
  --output-dir    ${FID_RESULTS}
```

### 논문 본문 추가 (§4.3 Synthesis Quality 섹션)

```
Image Quality Evaluation:
We assess synthesis quality using three complementary metrics.
FID and KID measure the distributional distance between generated
and real defect patches using InceptionV3 features; KID is preferred
for our dataset scale (~2,200 samples) as it provides an unbiased
estimate unlike FID. LPIPS evaluates perceptual quality at the
patch level: realism (mean distance between generated and real
patches, lower is better) and diversity (mean pairwise distance
within generated patches, higher is better). All metrics are
computed per defect class on held-out test patches.
```

---

## 파일 변경 요약

| 파일 | 변경 유형 | 내용 |
|------|-----------|------|
| `scripts/aggregate_multiseed_results.py` | **신규** | seed별 결과 집계, mean±std 계산 |
| `scripts/run_statistical_tests.py` | **신규** | Wilcoxon / Friedman / BH-FDR / Cohen's d |
| `scripts/run_image_quality_metrics.py` | **신규** | KID + LPIPS (realism / diversity) |
| `scripts/analyze_benchmark_results.py` | **수정** | mean±std LaTeX 테이블 출력 지원 |

> `run_benchmark.py`와 `run_fid.py`는 수정 불필요.

---

## 작업 순서

**P1 — Multi-Seed**
- [ ] 1. 브랜치 생성: `feature/statistical-robustness`
- [ ] 2. `scripts/aggregate_multiseed_results.py` 작성
- [ ] 3. `scripts/analyze_benchmark_results.py` 수정 (mean±std 테이블)
- [ ] 4. Colab에서 P1 실행 (seed 42, 123, 456 × 3 그룹)
- [ ] 5. `aggregate_multiseed_results.py` 실행 → aggregated_results.json 생성

**P2 — 통계 검정**
- [ ] 6. `scripts/run_statistical_tests.py` 작성
- [ ] 7. `run_statistical_tests.py` 실행 → significance_table 생성

**P3 — 이미지 품질 지표**
- [ ] 8. `scripts/run_image_quality_metrics.py` 작성
- [ ] 9. Colab에서 KID + LPIPS 실행
- [ ] 10. quality_metrics_table 생성

**논문 반영**
- [ ] 11. 결과 테이블 mean±std 형식으로 교체
- [ ] 12. 통계 서술 문구 추가 (§4 Experiments)
- [ ] 13. KID/LPIPS 비교 테이블 추가 (§4.3 Synthesis Quality)
- [ ] 14. main 브랜치에 PR

---

## 참고

- `run_benchmark.py --seed` 파라미터: L1220–1221 (이미 구현됨)
- `FIDCalculator` 클래스: `src/training/metrics.py` L372 (InceptionV3 feature 재사용 가능)
- `benchmark_results.json` 구조: L49–64 in `analyze_benchmark_results.py`
- 가설 정의: `05-Pipeline-StageD.md`, `08-Dataset-Groups.md`
