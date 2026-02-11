Context-Aware Steel Defect Augmentation (CASDA)
본 프로젝트는 Severstal Steel Defect 데이터셋을 바탕으로, 결함의 기하학적 특성(Sub-class)과 철강 표면의 물리적 배경 특징(Background Context)을 결합하여 고정밀 증강 데이터를 생성하는 통합 ControlNet 프레임워크를 구축합니다.

프로젝트 아키텍처 (Overview)
기존의 단순 합성 방식에서 벗어나, **"어떤 결함(What)"**이 **"어떤 배경(Where)"**에 놓여야 하는지의 물리적 정합성을 학습합니다.

1.Defect Analysis: 4대 지표 기반의 결함 Sub-class 세분화.
2.Context Analysis: 육안 특징 및 통계 지표 기반의 배경 클러스터링.
3.Optimal ROI Mapping: 배경과 결함 지표 간의 정합성 매핑.
4.ControlNet Generation: 배경 맥락을 인지하는 통합 생성 모델 학습.

1. 선행 작업 (Pre-requisites)
구현 파이프라인 가동 전, 다음 데이터 세트가 준비되어야 합니다.

A. 결함 Sub-class 지표 (Defect Metrics)각 결함 마스크에 대해 0.0 ~ 1.0 사이로 정규화된 지표를 산출합니다.
Linearity (Eccentricity): 직선성 (Class 3 핵심)
Solidity: 치밀도 (Class 4 핵심)
Extent: 분산도 (Class 1 핵심)
Aspect Ratio: 방향성 (공정 정합성 핵심)

B. 배경 특징 분류 (Background Inventory)
결함이 포함되어있는 이미지의 통계분석을 통해 배경을 4가지 유형으로 분류합니다.
Checkered | 규칙적인 다이아몬드 패턴
Vertical | 압연 흔적 및 수직 하이라이트
Smooth | 균일한 평면
Rough | 불규칙한 박리 및 노이즈

2. 데이터 전처리 파이프라인 (Data Pipeline)
ControlNet 학습을 위한 Multi-Channel Hint 이미지를 생성합니다.

Multi-Channel Hint 구조 (3-Channel RGB)
Red (Shape): 결함의 기하학적 형태 (Mask/Skeleton).
Green (Structure): 배경의 기하학적 구조 (Canny Edge/Normal Map).
Blue (Texture): 배경의 로컬 질감 강도 (Local Variance).

Prompt Engineering
학습 시 사용되는 텍스트 캡션의 표준 포맷:

"A steel surface with [Background Type], featuring a defect with [Linearity Value] and [Solidity Value] properties."


3. ControlNet 통합 모델 설계
Architecture: 단일 통합 ControlNet (Multi-Conditioning).
Goal: 배경 패턴을 파괴하지 않으면서, 패턴 위에 입체적으로 동화된 결함 픽셀 생성.
Key Feature: 배경의 에지(Green Channel)를 조건으로 입력받아 결함과의 물리적 교차점을 자연스럽게 처리.
