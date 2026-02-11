이제 추출된 ROI 데이터를 실제 학습이 가능한 형태로 가공하고, 모델에 주입하는 [데이터 패키징 및 학습 진입] 단계를 진행해야 합니다. 절차는 다음과 같습니다.

1. Multi-Channel Hint 이미지 제작 (Conditioning)
ControlNet에 입력할 '힌트' 이미지를 생성합니다. 단순히 흑백 마스크를 쓰는 것보다 배경과 결함의 관계를 정의한 3채널 힌트가 훨씬 효과적입니다.
Red 채널: 4대 지표(Linearity, Solidity 등)가 반영된 결함의 정밀 마스크.
Green 채널: 배경 분석 단계에서 파악된 배경의 주요 구조선(Stripe 등 엣지 정보).
Blue 채널: 배경의 미세 질감(Texture) 또는 노이즈 밀도.

2. 하이브리드 프롬프트 생성 (Prompt Engineering)
배경 타입과 결함 Sub-class를 결합하여 ControlNet에 줄 설명을 자동 생성합니다.
구조: [Sub-class 특성 결함] + [배경 타입] + [표면 상태]
예시: "A high-linearity scratch on a vertical striped metal surface with smooth texture."
이 프롬프트는 train.jsonl 파일에 각 ROI 이미지 경로와 함께 저장됩니다.

3. 학습 데이터셋 최종 검수 (Sanity Check)
학습에 들어가기 전, 생성된 ROI 패치들이 물리적으로 타당한지 검수합니다.
Distribution Check: 특정 Sub-class나 특정 배경 타입에 데이터가 너무 쏠려 있지 않은지 확인합니다. (불균형 시 증강/언더샘플링 결정)
Visual Check: ROI 패치 내에서 배경 패턴이 끊기거나 결함이 너무 구석에 치우치지 않았는지 샘플링 검사를 수행합니다.

4. ControlNet 학습 설정 (Configuration)
이 단계부터는 PROJECT(control_net).md 파일을 참고한다.
Base Model 선정: Stable Diffusion v1.5 등 적합한 베이스 모델 연결.
Hyperparameter 설정: Learning Rate, Batch Size, 그리고 배경 유지를 위한 ControlNet Weight 설정.

전체 요약 파이프라인
[현재 단계] 배경 분석 + 4대 지표 기반 ROI 추출 & Sub-class 분류.
[다음 단계] 3채널 힌트 제작 및 하이브리드 프롬프트 구성.
[준비 완료] train.jsonl 및 ROI 패치 데이터셋 구축.
[학습 시작] ControlNet 본격 학습 (다른 채팅방에서 진행).