[연구 리포트] CASDA 시스템의 성능 비교 실험 설계 및 분석
================================================================================

1. 실험 환경 및 벤치마크 설계 개요
--------------------------------------------------------------------------------

본 연구의 핵심은 검출 모델의 구조적 변경 없이, **데이터의 질적 개선(Data-centric AI)**
만으로 얼마나 성능을 향상시킬 수 있는지 입증하는 데 있다.

이를 위해 학계에서 검증된 최신 SOTA 모델들을 '고정된 측정 도구'로 활용하며,
학습 데이터셋의 구성만을 변수로 두어 비교 실험을 수행한다.

### 1.1 비교 대상 모델 (Benchmark Models)

증강 데이터의 범용성을 입증하기 위해 서로 다른 아키텍처 특성을 가진 2종의 Detection 모델을 선정한다.

| 모델         | 연도 | 특징                                                        |
|--------------|------|-------------------------------------------------------------|
| YOLO-MFD     | 2025 | 다중 스케일 엣지 특징 강화(MEFE) 모듈 탑재, 미세 결함 특화  |
| EB-YOLOv8    | 2025 | BiFPN 기반 복합 특징 융합, Severstal 주요 벤치마크 모델      |

**모델 선정 근거:**
- 서로 다른 백본 및 Neck 구조(MEFE vs BiFPN)를 가진 Detection 모델 2종으로 아키텍처 독립성 검증
- Severstal 데이터셋에서 이미 벤치마크 결과가 공개된 모델만 선정 (재현성 확보)
- 각 모델의 아키텍처를 고정하고, 동일 하이퍼파라미터로 학습하여 데이터 효과만 비교

### 1.2 데이터셋 비교 그룹 (Control Groups)

| 그룹                     | 구성                                                     | 목적                          |
|--------------------------|----------------------------------------------------------|-------------------------------|
| Baseline (Raw)           | Severstal 원본 데이터셋 (6,666매 결함 이미지)            | 증강 없는 기준선              |
| Baseline (Trad)          | 원본 + 전통적 기하 변환 (회전, 반전, 크기 조절) 증강     | 전통 증강 대비 효과 측정      |
| Proposed (CASDA-Full)    | 원본 + CASDA 생성 합성 이미지 5,000매 전수 사용          | CASDA 전체 출력의 효과        |
| Proposed (CASDA-Pruning) | 원본 + CASDA 생성 이미지 중 S_적합도 상위 2,000매 선별   | 품질 기반 선별의 효과         |

**CASDA-Pruning의 선별 기준:**
- `S_적합도(Suitability Score)` 상위 2,000매 선별
- Suitability Score는 matching_score, continuity_score, stability_score의 종합 지표
- 품질 검증(Stage 4) 통과 기준: 5-metric scoring 임계값 0.7 이상 (통과율 약 83%)


2. 성능 지표 및 측정 도구
--------------------------------------------------------------------------------

실험 결과는 Severstal(SSDD) 벤치마크의 표준 지표를 중심으로 분석한다.

| 지표                          | 용도                                     | 비고                         |
|-------------------------------|------------------------------------------|------------------------------|
| mAP (Mean Average Precision)  | 검출 모델 성능 비교의 주요 지표          | IoU 임계값 0.5 기준          |
| Dice Score                    | 세그멘테이션 정확도 측정                 | 클래스별 개별 산출 후 평균   |
| FID (Frechet Inception Distance) | 생성 이미지의 물리적 타당성 측정      | 낮을수록 실제 분포에 근접    |

**추가 분석 지표:**
- 클래스별 AP (Class 1~4 개별 분석) : 클래스 불균형 영향 확인
- Precision / Recall 곡선 : 증강 데이터의 false positive 유발 여부 확인


3. CASDA 합성 데이터 생성 파이프라인
--------------------------------------------------------------------------------

실험에 사용되는 합성 데이터는 다음 5단계 파이프라인을 통해 생성된다.

### 3.1 파이프라인 개요

| Stage | 단계                     | 환경  | 소요 시간 | 출력                                |
|-------|--------------------------|-------|-----------|-------------------------------------|
| 1     | ROI Extraction           | CPU   | ~8h       | 3,247 ROI 패치 + 4-metric 메타데이터|
| 2     | ControlNet Data Prep     | CPU   | ~37min    | 멀티채널 hint + 하이브리드 프롬프트  |
| 3     | Augmentation Generation  | GPU   | ~50min    | 2,500 합성 이미지                   |
| 4     | Quality Validation       | CPU   | ~8min     | 품질 검증 (통과율 ~83%)             |
| 5     | Dataset Merging          | CPU   | ~6min     | 원본 + 합성 통합 데이터셋           |

### 3.2 ControlNet 학습 설정

합성 이미지 생성을 위한 ControlNet 모델의 학습 설정:

| 항목                         | 값                              |
|------------------------------|---------------------------------|
| Base Model                   | Stable Diffusion v1.5           |
| ControlNet Init              | lllyasviel/sd-controlnet-canny  |
| Resolution                   | 512                             |
| Batch Size                   | 1                               |
| Epochs                       | 100                             |
| Gradient Accumulation Steps  | 4                               |
| Learning Rate                | 1e-5                            |
| LR Scheduler                 | constant_with_warmup (50 steps) |
| Mixed Precision              | fp16                            |
| SNR Gamma                    | 5.0                             |
| Early Stopping Patience      | 20 epochs                       |
| Seed                         | 42                              |

### 3.3 멀티채널 Hint 구조

ControlNet에 입력되는 hint 이미지는 3채널 RGB로 구성된다:

| 채널  | 내용                                         |
|-------|----------------------------------------------|
| R     | Defect mask (4-indicator enhancement 적용)   |
| G     | Background structure lines (엣지 정보)       |
| B     | Background fine texture (미세 질감)          |

### 3.4 프롬프트 구조

하이브리드 텍스트 프롬프트 형식:

```
"a {defect_subtype} surface defect on {background_type} metal surface
 with {surface_condition}, steel defect class {class_id}"
```

Negative prompt (공통):
```
"blurry, low quality, artifacts, noise, distorted, warped,
 unrealistic, oversaturated, cartoon, painting, text, watermark, logo"
```


4. 학습 품질 개선 사항 (v4)
--------------------------------------------------------------------------------

v4 학습에서는 이전 버전에서 확인된 RGB 색상 아티팩트 문제를 해결하기 위해
6개의 개선 방안을 적용하였다.

### 4.1 추론(Inference) 측 개선

| ID  | 개선 내용                            | 적용 파일               | 설정값                  |
|-----|--------------------------------------|-------------------------|-------------------------|
| A1  | Grayscale 후처리 추가                | test_controlnet.py      | --grayscale_postprocess |
|     | `image.convert("L").convert("RGB")` |                         |                         |
| A2  | conditioning_scale 기본값 하향       | test_controlnet.py      | 1.0 -> 0.7             |
|     |                                      | run_validation_phases.py|                         |
|     |                                      | train_controlnet.py     |                         |

### 4.2 학습(Training) 측 개선

| ID  | 개선 내용                            | 적용 파일               | 설정값                      |
|-----|--------------------------------------|-------------------------|-----------------------------|
| B1  | Target 이미지 grayscale 강제 변환    | train_controlnet.py     | --force_grayscale_target    |
|     | `convert("L") -> merge("RGB",[g,g,g])` |                      |                             |
| B2  | Grayscale 일관성 loss 추가           | train_controlnet.py     | --gray_loss_lambda 0.1      |
|     | `L_gray = MSE(ch0-ch1) + MSE(ch1-ch2)` |                      |                             |
| B3  | Min-SNR Weighting 적용              | train_controlnet.py     | snr_gamma=5.0               |
|     | Early stopping patience 확대         |                         | patience=20                 |
|     | 데이터 증강 추가                     |                         | --augment                   |
|     | (horizontal flip 50%,                |                         |                             |
|     |  brightness/contrast jitter 0.9~1.1) |                         |                             |


5. 실험 실행 계획
--------------------------------------------------------------------------------

### 5.1 Phase 1: ControlNet 학습 및 합성 데이터 생성

```
[수정 적용] -> [v4 학습 실행] -> [Validation] -> [합성 데이터 생성]
```

1. 경로 해석 수정 적용
   - `_resolve_hint_path()`에 data_dir.parent 기준 해석 추가 (수정 A)
   - `create_train_jsonl()`의 base_dir를 output_dir로 변경 (수정 B)
2. Colab 환경에서 v4 ControlNet 학습 실행
3. 4단계 Validation 실행
   - Phase 1: 기본 생성 품질 비교
   - Phase 2: 하이퍼파라미터 스윕 (guidance scale, inference steps, seed)
   - Phase 3: Unseen 데이터 일반화 테스트
   - Phase 4: 학습 통계 리포트
4. RGB 아티팩트 해소 여부 확인
5. 품질 검증 통과 합성 이미지 5,000매 생성

### 5.2 Phase 2: 벤치마크 모델 학습 및 평가

각 벤치마크 모델(YOLO-MFD, EB-YOLOv8)에 대해:

1. 4개 데이터셋 그룹별로 독립 학습 실행 (총 8회)
   - Baseline (Raw) / Baseline (Trad) / CASDA-Full / CASDA-Pruning
2. 동일 테스트 세트에 대해 추론 실행
3. 성능 지표(mAP, Dice Score) 산출
4. 클래스별 AP 분석 및 Precision/Recall 곡선 생성

### 5.3 Phase 3: 합성 데이터 품질 평가

1. FID Score 산출: 원본 vs CASDA 합성 이미지
2. 클래스별 FID 비교 (Class 1~4)
3. 생성 이미지의 결함 형태 분포 분석 (linearity, solidity, extent, aspect_ratio)


6. 기대 결과 및 분석 프레임워크
--------------------------------------------------------------------------------

### 6.1 성능 비교 테이블 (목표 형식)

| 모델       | 데이터셋         | mAP@0.5 | Class1 AP | Class2 AP | Class3 AP | Class4 AP |
|------------|------------------|---------|-----------|-----------|-----------|-----------|
| YOLO-MFD   | Baseline (Raw)   | 0.5546  | 0.5090    | 0.4525    | 0.6806    | 0.5764    |
| YOLO-MFD   | Baseline (Trad)  | 0.4880  | 0.5126    | 0.2444    | 0.6379    | 0.5572    |
| YOLO-MFD   | Copy-Paste       | 0.5257  | 0.5076    | 0.3811    | 0.6501    | 0.5641    |
| YOLO-MFD   | CASDA (Pruning)  | 0.5835  | 0.5298    | 0.5317    | 0.6886    | 0.5839    |
| EB-YOLOv8  | Baseline (Raw)   |         |           |           |           |           |
| EB-YOLOv8  | Baseline (Trad)  |         |           |           |           |           |
| EB-YOLOv8  | Copy-Paste       |         |           |           |           |           |
| EB-YOLOv8  | CASDA (Pruning)  |         |           |           |           |           |

### 6.2 핵심 검증 가설

| 가설 | 내용                                                              | 검증 방법                    |
|------|-------------------------------------------------------------------|------------------------------|
| H1   | CASDA 증강 데이터는 전통 증강보다 mAP를 유의미하게 향상시킨다     | CASDA-Full vs Baseline(Trad) |
| H2   | 품질 기반 선별(Pruning)이 전수 사용보다 효율적이다                | CASDA-Pruning vs CASDA-Full  |
| H3   | CASDA 증강 효과는 모델 아키텍처에 독립적이다 (범용성)             | YOLO-MFD·EB-YOLOv8 양쪽에서 방향 일치 확인 |
| H4   | CASDA는 소수 클래스(Class 3, 4)의 성능을 특히 개선한다            | 클래스별 AP 비교             |
| H5   | CASDA 합성 이미지는 물리적으로 타당하다 (FID 기준)                | FID Score 비교               |

### 6.3 분석 관점

1. **데이터 양 vs 질**: CASDA-Full(5,000매) vs CASDA-Pruning(2,000매) 비교로
   단순 양적 증가보다 질적 선별이 효과적인지 검증
2. **클래스 불균형 해소**: 소수 클래스(Class 3, 4)에 대한 증강 효과 집중 분석
3. **아키텍처 범용성**: Detection/Segmentation 양쪽에서 일관된 향상 확인
4. **물리적 타당성**: FID Score + 결함 형태 분포로 생성 품질의 객관적 평가
