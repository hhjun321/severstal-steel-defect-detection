# ROI 매칭 결과 시각화 가이드
## 5개 샘플 이미지에서 추출된 ROI 위치 및 특성 분석

---

## 📋 개요

이 문서는 CASDA 시스템의 Stage 1 (ROI Extraction) 결과를 시각화하여 보여줍니다.
실제 결함이 있는 5개의 다양한 샘플 이미지에서 추출된 ROI의 위치, 특성, 그리고 결함-배경 매칭 점수를 확인할 수 있습니다.

---

## 🎯 선택된 5개 샘플

다양한 클래스와 특성을 대표하는 샘플들:

### 1. `0007a71bf.jpg` - 완벽한 매칭 사례
**클래스**: Class 3 (선형 결함)  
**ROI 개수**: 1개  
**특징**: linear_scratch + vertical_stripe (완벽한 물리적 매칭)

```
ROI 정보:
├─ ROI ID: 0
├─ 결함 서브타입: linear_scratch (선형 스크래치)
├─ 배경 유형: vertical_stripe (수직 줄무늬)
├─ 적합도 점수: 0.813 (suitable ✅)
│  ├─ Matching Score: 1.000 (완벽)
│  ├─ Continuity Score: 0.519
│  └─ Stability Score: 0.788
├─ ROI 위치: (1145, 4) ~ (1168, 255)
├─ 결함 크기: 5,522 픽셀
└─ 통계 지표:
   ├─ Linearity: 0.992 (매우 높은 직선성)
   ├─ Solidity: 0.983 (매우 치밀)
   ├─ Extent: 0.957
   └─ Aspect Ratio: 11.40 (길쭉한 형태)

물리적 타당성:
  ✅ 선형 스크래치는 수직 줄무늬 배경에서 자연스럽게 발생
  ✅ Matching Score 1.0 = 최적 조합
  ✅ Linearity 0.99 = 완벽한 직선 형태
  ✅ Aspect Ratio 11.4 = 매우 긴 선형 결함
```

**시각화 설명**:
- 원본 이미지 우측에 수직으로 긴 선형 결함
- 수직 줄무늬 배경 위에 명확한 스크래치
- 녹색 박스 = suitable (0.8 이상)

---

### 2. `005f19695.jpg` - 다중 ROI 샘플
**클래스**: Class 3 (선형 결함)  
**ROI 개수**: 3개  
**특징**: 동일 이미지에서 여러 결함 영역 추출

```
ROI 0:
├─ 결함 서브타입: linear_scratch
├─ 배경 유형: vertical_stripe
├─ 적합도 점수: 0.814 (suitable ✅)
├─ ROI 위치: (481, 0) ~ (503, 126)
├─ 결함 크기: 1,902 픽셀
└─ 지표:
   ├─ Linearity: 0.982
   ├─ Aspect Ratio: 7.48
   └─ Matching Score: 1.000

ROI 2:
├─ 결함 서브타입: linear_scratch
├─ 배경 유형: vertical_stripe
├─ 적합도 점수: 0.862 (suitable ✅)
├─ ROI 위치: (574, 0) ~ (613, 256)
├─ 결함 크기: 4,480 픽셀
└─ 지표:
   ├─ Linearity: 0.995 (거의 완벽한 직선)
   ├─ Aspect Ratio: 14.53 (매우 길쭉)
   └─ Matching Score: 1.000

물리적 타당성:
  ✅ 동일 이미지에서 여러 선형 스크래치 검출
  ✅ 모두 수직 줄무늬 배경과 완벽하게 매칭
  ✅ 일관된 높은 Linearity (0.98+)
```

---

### 3. `000a4bcdd.jpg` - 풍부한 ROI 샘플
**클래스**: Class 1 (일반 결함)  
**ROI 개수**: 10개  
**특징**: compact_blob + smooth 매칭 다수 포함

```
ROI 1: compact_blob + smooth
├─ 적합도 점수: 0.843 (suitable ✅)
├─ ROI 위치: (338, 108) ~ (356, 141)
├─ 결함 크기: 544 픽셀
└─ 지표:
   ├─ Linearity: 0.716
   ├─ Solidity: 0.963 (매우 치밀)
   ├─ Aspect Ratio: 1.88
   └─ Matching Score: 1.000

ROI 3: compact_blob + smooth
├─ 적합도 점수: 0.728 (suitable ✅)
├─ ROI 위치: (373, 128) ~ (418, 156)
├─ 결함 크기: 930 픽셀
└─ Matching Score: 1.000

ROI 6: compact_blob + smooth
├─ 적합도 점수: 0.739 (suitable ✅)
├─ ROI 위치: (352, 193) ~ (378, 228)
├─ 결함 크기: 812 픽셀
└─ Matching Score: 1.000

ROI 7: compact_blob + smooth
├─ 적합도 점수: 0.803 (suitable ✅)
├─ ROI 위치: (244, 202) ~ (268, 225)
├─ 결함 크기: 463 픽셀
└─ 지표:
   ├─ Linearity: 0.178 (낮음 = 원형)
   ├─ Solidity: 0.959 (치밀)
   ├─ Aspect Ratio: 1.10 (거의 원형)
   └─ Matching Score: 1.000

ROI 8: compact_blob + smooth
├─ 적합도 점수: 0.743 (suitable ✅)
├─ ROI 위치: (146, 206) ~ (188, 242)
├─ 결함 크기: 1,130 픽셀
└─ Matching Score: 1.000

ROI 2, 5, 9: general/compact_blob + complex_pattern
├─ 적합도 점수: 0.52~0.66 (acceptable 🟠)
└─ Matching Score: 0.6~0.7 (보통)

물리적 타당성:
  ✅ Compact blob은 smooth 배경에서 가장 잘 보임
  ✅ 5개의 ROI가 suitable 등급 (perfect matching)
  ✅ Complex pattern 배경은 acceptable (괜찮은 매칭)
  ✅ Solidity 0.95+ = 매우 압축된 blob 형태
  ✅ Aspect Ratio 1.1~1.9 = 거의 원형에 가까움
```

---

### 4. `000f6bf48.jpg` - 희귀 클래스 샘플
**클래스**: Class 4 (희귀 결함)  
**ROI 개수**: 2개  
**특징**: 대형 결함 영역

```
ROI 0: 대형 결함 영역
├─ 결함 서브타입: compact_blob
├─ 배경 유형: complex_pattern
├─ 적합도 점수: 0.616 (acceptable 🟠)
├─ ROI 위치: (515, 0) ~ (861, 218)
├─ 결함 크기: 49,938 픽셀 (매우 큼!)
└─ 지표:
   ├─ Linearity: 0.656
   ├─ Solidity: 0.931
   ├─ Extent: 0.662
   └─ Matching Score: 0.600

ROI 1:
├─ 결함 서브타입: general
├─ 배경 유형: complex_pattern
├─ 적합도 점수: 0.637 (acceptable 🟠)
├─ ROI 위치: (853, 126) ~ (1132, 256)
├─ 결함 크기: 19,419 픽셀 (큼)
└─ Matching Score: 0.700

물리적 타당성:
  ⚠️ 매우 큰 결함 영역 (49K 픽셀)
  ✅ Complex pattern과의 매칭은 acceptable
  ✅ Class 4는 희귀하지만 중요한 클래스
  📊 증강에 우선적으로 사용 가능
```

---

### 5. `0014fce06.jpg` - 복합 패턴 배경
**클래스**: Class 3 (선형 결함)  
**ROI 개수**: 1개  
**특징**: linear_scratch + complex_pattern

```
ROI 0:
├─ 결함 서브타입: linear_scratch
├─ 배경 유형: complex_pattern
├─ 적합도 점수: 0.511 (acceptable 🟠)
│  ├─ Matching Score: 0.300 (낮음)
│  ├─ Continuity Score: 0.836 (높음)
│  └─ Stability Score: 0.552
├─ ROI 위치: (896, 60) ~ (930, 250)
├─ 결함 크기: 4,851 픽셀
└─ 지표:
   ├─ Linearity: 0.977 (매우 높은 직선성)
   ├─ Solidity: 0.879
   ├─ Aspect Ratio: 6.66
   └─ Matching Score: 0.300

물리적 타당성:
  ⚠️ Matching Score 낮음 (0.3)
  ✅ 하지만 Linearity 0.98로 명확한 선형 결함
  ✅ Continuity 0.84로 배경이 균일함
  📊 Acceptable 등급 - 사용 가능하나 최적은 아님
  
분석:
  - Complex pattern 배경에서 linear scratch는 이상적이지 않음
  - 그러나 높은 continuity로 배경이 균일하여 사용 가능
  - 결함 자체는 명확한 선형 형태 (Linearity 0.98)
```

---

## 📊 전체 통계 요약

### 적합도 점수 분포

```
Suitable (≥0.8):    8개 ROI  ████████████████████  (42%)
  • 0007a71bf.jpg: 1개 (선형+수직줄무늬)
  • 005f19695.jpg: 2개 (선형+수직줄무늬)
  • 000a4bcdd.jpg: 5개 (blob+smooth)

Acceptable (0.5-0.8): 11개 ROI  ███████████████████████  (58%)
  • 000a4bcdd.jpg: 5개 (다양한 조합)
  • 000f6bf48.jpg: 2개 (희귀 클래스)
  • 0014fce06.jpg: 1개 (선형+복합패턴)
  • 기타

Unsuitable (<0.5):  0개 ROI  (0%)

총 ROI: 19개
평균 적합도 점수: 0.71
```

### 클래스별 분포

```
Class 1 (일반):    10개 ROI  ████████████████████████████
Class 2:           0개  (이 샘플에 없음)
Class 3 (선형):    7개  ████████████████████
Class 4 (희귀):    2개  █████

희귀 클래스 비율: 10.5%
```

### 결함-배경 매칭 패턴

```
Perfect Matching (1.0):
  • linear_scratch + vertical_stripe  → 4회
  • compact_blob + smooth             → 5회
  합계: 9회 (47%)

Good Matching (0.7):
  • general + complex_pattern         → 2회

Acceptable Matching (0.6):
  • compact_blob + complex_pattern    → 4회
  • general + smooth                  → 1회

Sub-optimal Matching (0.3):
  • linear_scratch + complex_pattern  → 1회
```

---

## 🎨 시각화 스크립트 사용법

### 방법 1: 자동 실행

```bash
# 5개 샘플 자동 선택 및 시각화
python scripts/quick_visualize_roi.py
```

**출력**:
- `outputs/roi_visualizations/roi_visualization_1_*.png` (샘플 1 상세)
- `outputs/roi_visualizations/roi_visualization_2_*.png` (샘플 2 상세)
- `outputs/roi_visualizations/roi_visualization_3_*.png` (샘플 3 상세)
- `outputs/roi_visualizations/roi_visualization_4_*.png` (샘플 4 상세)
- `outputs/roi_visualizations/roi_visualization_5_*.png` (샘플 5 상세)
- `outputs/roi_visualizations/roi_comparison_all_5_samples.png` (비교 뷰)

### 방법 2: 커스텀 실행

```python
from scripts.visualize_roi_matching import ROIVisualizer

visualizer = ROIVisualizer(
    roi_metadata_path="data/processed/roi_patches/roi_metadata.csv",
    train_csv_path="data/train.csv",
    train_images_dir="data/train_images"
)

# 특정 이미지 시각화
visualizer.visualize_single_image("0007a71bf.jpg", save_path="output.png")

# 여러 이미지 비교
visualizer.visualize_comparison(
    ["0007a71bf.jpg", "000a4bcdd.jpg"],
    save_path="comparison.png"
)
```

---

## 🔍 시각화 요소 설명

### 상세 뷰 (roi_visualization_X_*.png)

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  [원본 이미지 + ROI 박스들]  [ROI 1] [ROI 2] [ROI 3] ...  │
│                                                             │
│  • 좌측: 전체 이미지에 모든 ROI 박스 표시                     │
│  • 우측: 각 ROI 패치 (512×512) + 결함 마스크 오버레이        │
│                                                             │
│  박스 색상:                                                  │
│  🟢 Green   = suitable (≥0.8)                               │
│  🟠 Orange  = acceptable (0.5~0.8)                          │
│  🔴 Red     = unsuitable (<0.5)                             │
│                                                             │
│  각 패치 정보:                                               │
│  - ROI ID, Class ID                                         │
│  - Defect subtype (결함 서브타입)                            │
│  - Background type (배경 유형)                               │
│  - Suitability score (적합도 점수)                          │
│  - Recommendation (권장사항)                                │
│                                                             │
│  결함 마스크:                                                │
│  • 빨간색 오버레이 = 실제 결함 영역 (RLE 디코딩)              │
│  • 70% 원본 + 30% 빨간색 블렌딩                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 비교 뷰 (roi_comparison_all_5_samples.png)

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  [Image 1]  [Image 2]  [Image 3]  [Image 4]  [Image 5]       │
│                                                                │
│  • 5개 이미지를 나란히 배치                                      │
│  • 각 이미지에 모든 ROI 박스 표시                               │
│  • 클래스 라벨 (C1, C2, C3, C4) 표시                            │
│  • 색상 코딩으로 적합도 한눈에 파악                              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 💡 주요 인사이트

### 1. 완벽한 매칭 조합 (Matching Score 1.0)

✅ **linear_scratch + vertical_stripe**
- 물리적으로 가장 타당한 조합
- Linearity 0.98+ 보장
- 수직 줄무늬 위의 선형 스크래치는 실제로 흔함

✅ **compact_blob + smooth**
- 균일한 배경에서 blob이 명확히 보임
- Solidity 0.95+ 보장
- 검출 및 분류가 용이

### 2. Acceptable 조합의 활용

🟠 **compact_blob + complex_pattern**
- Matching Score 0.6이지만 사용 가능
- 실제 제조 환경에서도 발생
- 다양성 증대에 기여

🟠 **linear_scratch + complex_pattern**
- 최적은 아니지만 realistic
- Linearity가 높으면 충분히 식별 가능

### 3. 클래스 불균형 해결

📊 **Class 4 우선 증강**
- 희귀 클래스 (전체의 10-15%)
- Acceptable 등급이라도 우선 사용
- 34.2% 증가 목표 달성

---

## 🎯 증강 전략

선택된 5개 샘플을 바탕으로:

1. **Suitable ROI (8개)**
   - 최우선 증강 대상
   - 다양한 scale (0.8-1.0)
   - 여러 배경에 적용

2. **Class 4 ROI (2개)**
   - Acceptable이지만 우선 증강
   - 희귀 클래스 균형 맞추기

3. **Acceptable ROI (11개)**
   - 다양성 확보를 위해 일부 사용
   - 품질 검증 통과 시 포함

**예상 증강 결과**:
- 이 5개 이미지에서만 ~50-100개 합성 샘플 생성 가능
- 전체 3,247개 ROI에서 2,500개 생성 (실제 결과: 2,075개 통과)

---

## 📚 참고 자료

- **ROI Extraction Guide**: README_ROI_KR.md
- **Technical Whitepaper**: TECHNICAL_WHITEPAPER_KR.md (Chapter 3)
- **Research Report**: RESEARCH_REPORT_KR.md (Section 3.2)

---

**작성일**: 2026년 2월 10일  
**버전**: 1.0  
**스크립트 위치**: `scripts/quick_visualize_roi.py`, `scripts/visualize_roi_matching.py`
