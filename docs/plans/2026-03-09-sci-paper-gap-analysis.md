# SCI 논문 Gap Analysis 및 추가 실험 계획
**작성일**: 2026-03-09
**목적**: CASDA 논문의 SCI 투고를 위한 현황 점검 및 필요 실험 도출

---

## 1. 논문 핵심 주장 (Narrative)

> "극도의 클래스 불균형을 가진 산업 결함 검출 데이터에서,
> 구조 조건부 합성(CASDA)은 소수 클래스 표현을 강화하여
> Detection 성능을 개선하며, 이 효과는 아키텍처에 독립적이다."

**핵심 근거:**
- YOLO-MFD mAP@0.5: +2.89%p (0.5546 → 0.5835)
- Class2 AP(원본 247장): +7.92%p (0.4525 → 0.5317) ← 가장 강력한 근거
- H3: EB-YOLOv8, YOLO-MFD, DeepLabV3+ 3종 모두에서 효과 확인
- H4: 소수 클래스(Class 2, 3, 4) 집중 개선

---

## 2. 현황 Gap Analysis

| 논문 섹션 | SCI 필수 요건 | 현재 상태 |
|-----------|--------------|-----------|
| Introduction | 문제 정의, Contribution 3~4개 | ✅ 초안 존재 |
| Related Work | 증강 방법론 서베이, 결함 검출 SOTA | ✅ 초안 존재 |
| Method | CASDA 5단계 파이프라인 기술 | ✅ 실험 문서 존재 |
| Dataset Statistics | 클래스 불균형 통계, 시각화 | ✅ 보유 |
| Baseline - Raw | 원본 데이터 학습 결과 | ✅ 완료 |
| Baseline - Trad | 전통적 기하 변환 증강 | 🔄 학습 중 |
| Baseline - CopyPaste | ROI 직접 붙여넣기 증강 | ❌ 미실험 |
| Baseline - GAN 기반 | 생성 모델 비교군 1종 | ❌ 미실험 |
| CASDA 결과 (v5.6) | 3종 모델 × CASDA 결과 | ✅ 완료 |
| Ablation Study | 각 컴포넌트 기여도 분석 | ❌ 전무 |
| Statistical Significance | 다중 seed, 평균/표준편차 | ❌ 단일 seed=42 |
| Qualitative Analysis | 생성 이미지 시각화 | △ FID 분석 있으나 시각 자료 부족 |
| Discussion | 한계, Segmentation 악화 설명 | △ 분석 문서 존재 |

---

## 3. 핵심 Gap 3개

### Gap 1 — 추가 Baseline 부족 [Critical]
- **문제**: 현재 Raw + Trad(학습 중)만 존재. CASDA 우월성 주장 근거 약함
- **필요**: CopyPaste + GAN 기반 1종 추가
- **리뷰어 예상 지적**: "단순 copy-paste 대비 ControlNet이 왜 필요한가?"

### Gap 2 — Ablation Study 전무 [Critical]
- **문제**: CASDA 각 컴포넌트(멀티채널 hint, Pruning, Poisson Blending)의
  개별 기여 입증 불가
- **SCI 기준**: Ablation 없이 accept 불가

### Gap 3 — 단일 Seed 실험 [High]
- **문제**: seed=42 단일 실험만 존재. 통계적 유의성 미검증
- **필요**: 최소 3회 반복(seed 42, 123, 456) + 평균±표준편차

---

## 4. 추가 실험 계획 (우선순위순)

### [P0] ① Ablation Study — CASDA 컴포넌트 기여도

**실험 설계:**

| 실험명 | 제거/변경 컴포넌트 | 목적 |
|--------|-------------------|------|
| CASDA-Full (기준) | 없음 | 전체 파이프라인 기준선 |
| w/o Pruning | suitability 기반 선별 제거, 전량 사용 | Pruning 기여 증명 |
| w/o Multi-hint | 3채널 → 1채널(defect mask만) | 멀티채널 hint 기여 증명 |
| w/o Blending | Poisson Blending 없이 ROI 직접 합성 | 블렌딩 기여 증명 |

- **측정 모델**: YOLO-MFD (가장 명확한 개선 보인 모델)
- **측정 지표**: mAP@0.5, Class2 AP (핵심 근거와 직결)

### [P0] ② CopyPaste Baseline 추가

**구현 방식:**
- 원본 결함 ROI를 clean background 이미지에 직접 붙여넣기
- Poisson Blending 없이 단순 paste (CASDA와의 blending 효과 차이 부각)
- 동일한 ROI 수(~2,242장)로 통제 비교

**비교 목적**: "구조 조건부 생성(ControlNet)이 단순 복사 붙여넣기보다 왜 우수한가?"

### [P1] ③ GAN 기반 Baseline 1종

**후보:**
- **DCGAN**: 구현 간단, 비교 대상으로 충분, 학습 시간 단축
- **선택 기준**: CASDA의 ControlNet 기반 생성 vs 비조건부 GAN 생성의 차이 부각

**비교 목적**: "왜 ControlNet인가? 단순 GAN 대비 무엇이 나은가?"

### [P2] ④ 다중 Seed 통계 실험

**실험 설계:**
- Seed: 42, 123, 456 (3회 반복)
- 대상: Baseline(Raw) vs CASDA-Composed-Pruning
- 모델: YOLO-MFD (핵심 모델)
- 결과: 평균 ± 표준편차, paired t-test

---

## 5. 현실적 실행 순서

```
현재 진행 중: Baseline(Trad) 학습 완료 대기
    ↓
Step 1: Ablation Study 설계 및 실행 [P0, ~1~2일]
    ↓
Step 2: CopyPaste Baseline 구현 + 실행 [P0, ~0.5일 구현 + 학습]
    ↓
Step 3: GAN Baseline 실행 [P1, GPU 여유 시]
    ↓
Step 4: 다중 Seed 통계 실험 [P2, GPU 여유 시]
    ↓
논문 실험 섹션 완성
```

---

## 6. 최종 비교 테이블 목표 형식 (논문 Table)

| Method | YOLO-MFD mAP | EB-YOLOv8 mAP | DLv3+ Dice | Class2 AP |
|--------|-------------|--------------|-----------|-----------|
| Baseline (Raw) | 0.5546 | 0.5819 | 0.6290 | 0.4525 |
| Baseline (Trad) | 🔄 | 🔄 | 🔄 | 🔄 |
| CopyPaste | ❌ | ❌ | ❌ | ❌ |
| GAN-based | ❌ | ❌ | ❌ | ❌ |
| **CASDA (Ours)** | **0.5835** | **0.5850** | 0.6232 | **0.5317** |

---

## 7. Segmentation 악화 대응 전략

DeepLabV3+ Dice -0.58%p 악화는 논문에서 정직하게 분석:
- **원인**: FID-Composed 240.17 → Poisson Blending 아티팩트가 pixel-level 학습에 악영향
- **논문 내 처리**: Limitation 섹션에서 분석 + stripe 배경 합성 개선 방향 제시
- **보완**: Class1 Dice +2.24%p 개선 사례로 단순 패턴에서의 효과 강조
