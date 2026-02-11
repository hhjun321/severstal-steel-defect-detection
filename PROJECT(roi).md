증강 데이터의 품질을 결정짓는 핵심은 **"어떤 결함(Sub-class)"**을 **"어디(Background Context)"**에 생성하느냐의 조합입니다. 
즉 단순히 결함 위치를 찾는 것이 아니라, **"결함이 생성되기에 적합한 배경인가"**를 판단하는 로직을 구축하는 것이 핵심입니다.


통계적 지표 기반 ROI 추출 프로세스
1. 전처리: 데이터셋 지표 수치화 (Characterization)
모든 Severstal 결함 이미지의 Ground Truth(GT) 마스크를 분석하여 다음 데이터를 DB화(CSV)합니다.
입력: 원본 이미지 및 마스크 이미지
작업: regionprops(Python skimage 라이브러리 등)를 활용하여 각 마스크의 Linearity, Solidity, Extent, Aspect Ratio를 계산
결과: image_id, class_id, linearity, solidity, extent, aspect_ratio, bbox_coords가 포함된 메타데이터 파일 생성


배경 분석을 포함한 ROI 준비 로직
결함의 기하학적 통계뿐만 아니라, 배경의 물리적 특성을 데이터셋에 입히는 과정입니다.
1. 그리드 기반 배경 타입 분류 (Grid-based Labeling)
전체 이미지(1600x256)를 64x64 또는 128x128 그리드로 나눈 뒤, 각 패치의 텍스처를 분석합니다.
분산(Variance) 분석: 분산이 낮으면 'Smooth(평면)', 높으면 **'Textured(패턴)'**로 1차 분류합니다.
주파수/엣지 분석 (FFT or Sobel): 특정 방향의 엣지 강도가 높으면 'Vertical/Horizontal Stripe', 사방으로 높으면 **'Complex Pattern'**으로 분류합니다.
결과: 이미지마다 [Grid_ID, Background_Type, Stability_Score] 정보를 가진 배경 맵(Background Map)을 생성합니다.

2. ROI 적합도 판정 (Suitability Score)
단순히 결함이 있는 위치만 따지는 것이 아니라, 배경 맵과 대조하여 ROI의 적합성을 평가합니다.
패턴 연속성(Continuity): ROI로 선정된 영역 내에 배경 패턴의 급격한 단절(Edge)이나 노이즈가 없는지 확인합니다.
결함-배경 매칭: * 예: 'Linear'한 결함(스크래치)은 'Vertical Stripe' 배경의 결을 따라 발생할 때 가장 자연스럽습니다.
이 매칭 정보를 ControlNet 학습 시 프롬프트(예: a linear scratch on vertical striped metal surface)에 포함시킵니다.

3. 최종 ROI 추출 로직 (Integrated Selection)
이제 두 가지 정보를 병합하여 최종 패치를 자릅니다.
기준 A: 결함의 4대 지표(Sub-class 선정)
기준 B: 배경의 안정성 및 타입(ROI 위치 보정)
작업: 결함 중심점으로 패치를 자를 때, 만약 중심점이 배경 경계면에 걸쳐 있다면 배경 패턴이 균일한 쪽으로 패치 윈도우를 미세 이동(Shift) 시켜 배경의 연속성을 확보합니다.

수정된 파이프라인 워크플로우
배경 분석: 전체 이미지를 그리드로 나눠 배경 타입 및 안정성 맵 생성.
결함 분석: GT 마스크에서 4대 지표를 계산하여 Sub-class 정의.
ROI 합성: 배경 맵과 결함 위치를 대조하여 최적의 512x512 ROI 패치 확정.
데이터 패키징: [ROI Image] + [Multi-channel Hint] + [Prompt(Sub-class + Background)]