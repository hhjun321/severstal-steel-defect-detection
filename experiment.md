[연구 리포트] CASDA 시스템의 성능 비교 실험 설계 및 분석
1. 실험 환경 및 벤치마크 설계 개요
본 연구의 핵심은 검출 모델의 구조적 변경 없이, **데이터의 질적 개선(Data-centric AI)**만으로 얼마나 성능을 향상시킬 수 있는지 입증하는 데 있다. 
이를 위해 학계에서 검증된 최신 SOTA 모델들을 '고정된 측정 도구'로 활용하며, 학습 데이터셋의 구성만을 변수로 두어 비교 실험을 수행한다.

1.1 비교 대상 모델 (Benchmark Models)
증강 데이터의 범용성을 입증하기 위해 서로 다른 아키텍처 특성을 가진 3종의 모델을 선정한다.
-YOLO-MFD (2025): 다중 스케일 엣지 특징 강화(MEFE) 모듈을 탑재하여 미세 결함에 특화된 최신 모델.
-EB-YOLOv8 (2025): BiFPN을 통해 복합 특징 융합 능력을 극대화한 Severstal 데이터셋의 주요 벤치마크 모델.
-DeepLabV3+ (2024): Severstal 데이터셋 분석의 표준으로 사용되는 세그멘테이션 기반 통합 시스템.

1.2 데이터셋 비교 그룹 (Control Groups)
Baseline (Raw): Severstal 원본 데이터셋(6,666매 결함 이미지)만 사용.
Baseline (Trad): 원본 + 전통적 기하 변환(회전, 반전, 크기 조절) 증강 적용.
Proposed (CASDA-Full): 원본 + CASDA로 생성된 5,000매의 합성 이미지 전수 사용.
Proposed (CASDA-Pruning): 원본 + CASDA 생성 이미지 중 $S_{적합도}$(Suitability Score) 상위 2,000매만 선별하여 사용.

2. 성능 지표 및 측정 도구
실험 결과는 Severstal(SSDD) 벤치마크의 표준 지표인 **mAP(Mean Average Precision)**와 Dice Score를 중심으로 분석한다. 
또한, 생성된 데이터의 물리적 타당성을 측정하기 위해 FID(Fréchet Inception Distance) 점수를 추가 지표로 활용한다.