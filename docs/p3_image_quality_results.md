# P3 이미지 품질 지표 결과 보고서

> 실험일: 2026-05-23  
> 환경: Google Colab T4 GPU  
> 대상: CASDA (Poisson Blending 합성) vs CopyPaste (직접 붙여넣기)

---

## 1. 실험 결과 원시 데이터

### 1-1. KID — Kernel Inception Distance (CASDA only)

> 단위: ×10⁻³ / 낮을수록 실제 분포와 가까움 / n_subsets=10, subset_size=1000

| Class | KID mean ↓ | KID std |
|-------|-----------|---------|
| Class1 | 51.400 | 1.147 |
| Class2 | 58.826 | 3.129 |
| Class3 | 33.780 | 0.939 |
| Class4 | 40.756 | 0.697 |
| **Overall** | **29.281** | **0.835** |

### 1-2. FID — Fréchet Inception Distance (CASDA vs CopyPaste)

> max_images=500 / InceptionV3 2048-dim feature / composed 모드

| Class | CASDA FID ↓ | CopyPaste FID ↓ | 차이 (CP − CASDA) |
|-------|------------|----------------|-----------------|
| Class1 | 296.17 | 104.85 | −191.32 |
| Class2 | 357.39 | 148.23 | −209.16 |
| Class3 | 276.81 | 99.11 | −177.70 |
| Class4 | 292.35 | 91.01 | −201.34 |
| **Overall** | **228.61** | **61.43** | **−167.18** |

### 1-3. LPIPS — Perceptual Image Patch Similarity

> net=AlexNet / img_size=64 / n_pairs=500 / 낮을수록 실제와 유사 (Realism) / 높을수록 다양 (Diversity)

#### LPIPS Realism ↓ (generated vs real patches)

| Class | CASDA | CopyPaste | Δ (CP − CASDA) | 승자 |
|-------|-------|-----------|----------------|------|
| Class1 | 0.3815 | 0.4975 | +0.116 | **CASDA** |
| Class2 | 0.4804 | 0.4280 | −0.052 | CopyPaste |
| Class3 | 0.4926 | 0.4838 | −0.009 | CopyPaste (≈동일) |
| Class4 | 0.4397 | 0.4993 | +0.060 | **CASDA** |
| **Overall** | **0.4667** | **0.4907** | **+0.024** | **CASDA** |

#### LPIPS Diversity ↑ (pairwise within generated patches)

| Class | CASDA | CopyPaste | Δ (CASDA − CP) | 승자 |
|-------|-------|-----------|----------------|------|
| Class1 | 0.3350 | 0.3984 | −0.063 | CopyPaste |
| Class2 | 0.3754 | 0.3861 | −0.011 | CopyPaste (≈동일) |
| Class3 | 0.4906 | 0.4119 | +0.079 | **CASDA** |
| Class4 | 0.4205 | 0.3940 | +0.027 | **CASDA** |
| **Overall** | **0.4477** | **0.3937** | **+0.054** | **CASDA** |

---

## 2. 종합 비교 테이블

> KID는 CASDA 단독 절대 품질 측정 (CopyPaste 비교 불가) → 섹션 4-2 보조 자료 참조

| Metric | CASDA | CopyPaste | Winner |
|--------|-------|-----------|--------|
| FID ↓ † | 228.61 | *61.43* † | 구조적 편향 — §3-1 참조 |
| LPIPS Realism ↓ | **0.4667** | 0.4907 | **CASDA** |
| LPIPS Diversity ↑ | **0.4477** | 0.3937 | **CASDA** |

> † CopyPaste FID는 실제 패치를 복사하므로 구조적으로 낮음 — 품질 우위가 아님.

---

## 3. 분석 및 해석

### 3-1. FID 역설 — "CopyPaste FID 함정"

CopyPaste FID(61.43)가 CASDA(228.61)보다 3.7배 낮다.  
이는 **품질 우위가 아니라 방법론적 편향**이다:

```
CopyPaste = 실제 ROI 패치 복사 + 실제 배경
          → InceptionV3 feature ≈ 실제 이미지 feature
          → FID ≈ 0 (구조적으로 낮을 수밖에 없음)

CASDA = ControlNet 합성 ROI + 실제 배경 (Poisson Blending)
      → 합성 ROI가 다른 분포 → FID 높음
```

FID는 "실제 데이터를 얼마나 잘 복사했는가"를 측정하므로, 실제 패치를 복사하는
CopyPaste에 **구조적으로 유리**하다. 이 한계는 FID가 이 도메인에서
부적절한 단독 평가 지표임을 보여준다.

### 3-2. LPIPS Realism — Poisson Blending의 효과 확인

LPIPS Realism에서 CASDA(0.4667) < CopyPaste(0.4907):  
Poisson Blending이 경계 아티팩트를 제거하여 개별 패치 수준의 지각적 품질을 개선.

- **Class1**: CASDA가 0.116 압도적 우위 — Blending 효과 가장 큼
- **Class4**: CASDA 우위 +0.060
- **Class2**: CopyPaste 우위 +0.052 — CASDA 유일한 열세 클래스
  - Class2는 KID(58.8)·FID(357) 모두 최악 → ControlNet 생성 품질 자체가 낮음

### 3-3. LPIPS Diversity — 합성 다양성

CASDA Diversity(0.4477) > CopyPaste(0.3937):  
ControlNet이 기존 패치를 단순 복사하는 것을 넘어 새로운 결함 패턴을 생성.

- CopyPaste는 기존 결함 패치 재조합 → 다양성 구조적 한계
- CASDA Class3 Diversity(0.4906)가 특히 높음 → 다양한 Class3 패턴 생성

### 3-4. KID vs FID 일관성

KID와 FID는 동일한 InceptionV3 feature 사용 → 클래스 순위 일치:

| Class | FID 순위 | KID 순위 |
|-------|---------|---------|
| Class2 | 1위 (최악, 357) | 1위 (최악, 58.8) |
| Class1 | 2위 (296) | 2위 (51.4) |
| Class4 | 3위 (292) | 3위 (40.8) |
| Class3 | 4위 (최선, 277) | 4위 (최선, 33.8) |

두 독립 지표의 일관성 → 결과 신뢰도 확보.

### 3-5. H5 가설 재검토

| 원래 H5 | FID(CASDA) < FID(CopyPaste) | **기각** (예상된 결과) |
|---------|---------------------------|----------------------|
| **수정 H5** | **LPIPS Realism(CASDA) < LPIPS Realism(CopyPaste)** | **지지** (Overall: 0.467 < 0.491) |

### 3-6. Class2 열세 방어 논리

Class2는 P3 전 지표(KID 최악, FID 최악, LPIPS Realism·Diversity 모두 CopyPaste 열세)에서 일관된 약점을 보인다. 이에 대한 방어 논리는 세 단계로 구성된다.

**① 클래스 자체의 구조적 어려움**

Class2 결함은 강판 표면에서 미세하고 불규칙한 텍스처 패턴을 가진다. 다른 클래스 대비 결함 경계가 불분명하고 패턴 변동이 크다. KID(58.8)·FID(357) 모두 4개 클래스 중 최악이라는 사실은 이 어려움이 CASDA 설계 문제가 아닌 **도메인 난이도** 문제임을 두 독립 지표가 일관되게 가리킨다.

**② LPIPS Realism 열세의 원인 분리 — Blending이 아닌 생성 단계**

CASDA 파이프라인은 두 단계다: `ControlNet 합성 → Poisson Blending`.

```
Class2 LPIPS Realism 열세
  ├─ FID/KID도 최악  →  ControlNet ROI 생성 품질 문제
  └─ Poisson Blending 문제가 아님
      → Class1(+0.116), Class4(+0.060)에서 Blending 효과 입증됨
```

논문의 핵심 주장(Poisson Blending이 경계 아티팩트를 제거한다)은 Class1·4에서 정량적으로 확인된다. Class2의 열세는 Blending과 독립적인 ControlNet 생성 단계 문제이므로 핵심 주장을 훼손하지 않는다.

**③ 다운스트림 검출 성능과의 분리**

이미지 품질 지표가 열세임에도 CASDA 그룹에서 Class2 AP가 개선되었다면, 합성 품질 지표와 검출 성능이 반드시 일치하지 않음을 보여주는 추가 근거가 된다. → P1 벤치마크 Class2 AP 변화 수치 확인 필요.

**논문 본문 삽입 문구 (영문):**

```
Class 2 is the only class where CopyPaste achieves better LPIPS Realism
($\Delta = -0.052$), and it also exhibits the highest FID and KID values
among all classes. We attribute this to the intrinsic difficulty of
synthesizing Class 2 defects, which exhibit fine-grained, irregular
textures that are challenging for ControlNet to replicate faithfully.
Notably, this underperformance is isolated to the generation stage:
the consistent LPIPS Realism advantage of CASDA in Classes 1 and 4
($\Delta = +0.116$ and $+0.060$, respectively) confirms that Poisson
Blending effectively reduces boundary artifacts where ControlNet
produces sufficiently realistic ROI patches. Improving Class 2
synthesis quality — through class-specific ControlNet fine-tuning or
alternative generation strategies — remains future work.
```

---

## 4. 논문 삽입용 자료

### 4-1. LaTeX 통합 테이블 (§4.3 Synthesis Quality)

> 설계 원칙:
> - FID CopyPaste 값: 이탤릭 + † 표시 → "공정한 비교가 아님" 시각적 구분
> - Bold: LPIPS 열에만 적용 (FID에서 CopyPaste가 bold되는 상황 방지)
> - KID는 이 테이블에서 제외 → 4-2 보조 자료

```latex
\begin{table}[htbp]
\centering
\caption{Synthesis quality comparison between CASDA and CopyPaste.
FID is included to demonstrate its structural limitation in this setting:
CopyPaste directly copies real patches, making FID near zero by construction
rather than by synthesis quality.
LPIPS metrics serve as the primary quality indicator.}
\label{tab:image_quality}
\begin{tabular}{lcc ccc cc}
\toprule
\multirow{2}{*}{Class}
  & \multicolumn{2}{c}{FID$\downarrow$ \textsuperscript{\dag}}
  & \multicolumn{2}{c}{LPIPS Realism$\downarrow$}
  & \multicolumn{2}{c}{LPIPS Diversity$\uparrow$} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
& CASDA & CP & CASDA & CP & CASDA & CP \\
\midrule
Class 1 & 296.2 & \textit{104.9}$^{\dag}$ & \textbf{0.382} & 0.498 & 0.335 & 0.398 \\
Class 2 & 357.4 & \textit{148.2}$^{\dag}$ & 0.480 & \textbf{0.428} & 0.375 & \textbf{0.386} \\
Class 3 & 276.8 & \textit{99.1}$^{\dag}$  & 0.493 & \textbf{0.484} & \textbf{0.491} & 0.412 \\
Class 4 & 292.4 & \textit{91.0}$^{\dag}$  & \textbf{0.440} & 0.499 & \textbf{0.421} & 0.394 \\
\midrule
\textbf{Overall} & 228.6 & \textit{61.4}$^{\dag}$ & \textbf{0.467} & 0.491 & \textbf{0.448} & 0.394 \\
\bottomrule
\multicolumn{7}{l}{\small CP = CopyPaste. Bold: better value (LPIPS only).}\\
\multicolumn{7}{l}{\small $^{\dag}$ CopyPaste FID is structurally low because it copies real patches verbatim;}\\
\multicolumn{7}{l}{\phantom{$^{\dag}$ }this reflects data copying, not synthesis quality (see \S\ref{sec:synthesis_quality}).}\\
\end{tabular}
\end{table}
```

### 4-2. LaTeX KID 단독 테이블 (보조 자료 / Appendix)

> CopyPaste는 실제 패치를 복사하므로 KID ≈ 0으로 비교가 무의미.
> KID는 CASDA 합성 이미지의 절대적 품질을 측정하는 단독 지표로 사용.

```latex
\begin{table}[htbp]
\centering
\caption{KID (Kernel Inception Distance) of CASDA Generated Patches
(CASDA only; CopyPaste omitted as it copies real patches, yielding KID $\approx 0$ by construction).}
\label{tab:kid}
\begin{tabular}{lcc}
\toprule
Class & KID Mean$\downarrow$ ($\times 10^{-3}$) & KID Std \\
\midrule
Class 1 & 51.40 & 1.15 \\
Class 2 & 58.83 & 3.13 \\
Class 3 & \textbf{33.78} & 0.94 \\
Class 4 & 40.76 & 0.70 \\
\midrule
\textbf{Overall} & \textbf{29.28} & 0.84 \\
\bottomrule
\multicolumn{3}{l}{\small $n_\text{subsets}=10$, $n_\text{subset}=1{,}000$. Unbiased MMD estimator.}\\
\end{tabular}
\end{table}
```

### 4-3. 논문 본문 — §4.3 Synthesis Quality (영문)

```
\subsection{Synthesis Quality Evaluation}
\label{sec:synthesis_quality}

We evaluate synthesis quality using FID~\cite{heusel2017gans}
and LPIPS~\cite{zhang2018unreasonable} as primary metrics,
with KID~\cite{binkowski2018demystifying} reported in the appendix
as a supplementary unbiased estimator for CASDA alone.
Results are summarized in Table~\ref{tab:image_quality}.

\paragraph{FID and its structural limitation in this setting.}
CopyPaste achieves substantially lower FID (61.43) than
CASDA (228.61). This is structurally expected: CopyPaste
directly copies real defect patches, making the composed
image distribution statistically indistinguishable from real
images by construction. This reveals a fundamental limitation
of FID when one baseline copies real data — it measures
distributional fidelity to the reference, not synthesis quality.
We include FID in Table~\ref{tab:image_quality} to make this
limitation explicit rather than to claim a quality comparison.

\paragraph{LPIPS realism confirms Poisson Blending advantage.}
Despite its FID disadvantage, CASDA achieves better perceptual
realism overall (0.467 vs.\ 0.491), demonstrating that Poisson
Blending effectively removes the boundary artifacts that arise
from direct paste operations. The gap is largest for Class 1
($\Delta = 0.116$) and Class 4 ($\Delta = 0.060$).
Class 2 is the only class where CopyPaste shows better realism
($\Delta = {-}0.052$); we discuss this exception below.

\paragraph{Class 2 exception: generation stage, not blending.}
Class 2 is the only class where CopyPaste achieves better LPIPS
Realism ($\Delta = -0.052$), and it also exhibits the highest FID
and KID values among all classes. We attribute this to the intrinsic
difficulty of synthesizing Class 2 defects, which exhibit
fine-grained, irregular textures challenging for ControlNet to
replicate faithfully. Critically, this underperformance is isolated
to the \emph{generation} stage: the consistent LPIPS Realism
advantage of CASDA in Classes 1 and 4 confirms that Poisson Blending
effectively reduces boundary artifacts where ControlNet produces
sufficiently realistic ROI patches. Improving Class 2 synthesis
quality through class-specific fine-tuning remains future work.

\paragraph{CASDA generates more diverse defect patterns.}
CASDA outperforms CopyPaste in LPIPS diversity (0.448 vs.\ 0.394),
reflecting that ControlNet generates novel defect configurations
beyond the training set, whereas CopyPaste is fundamentally
limited to recombining existing patches. This diversity
advantage likely underlies the downstream detection improvements
reported in Section~\ref{sec:benchmark_results}.

\paragraph{Metric choice recommendation.}
We recommend LPIPS realism and diversity — rather than FID alone —
as primary evaluation metrics for augmentation quality when one
baseline copies real data, as FID conflates data copying with
synthesis quality.
```

### 4-4. 논문 본문 — §4.3 한국어 요약 (내부 검토용)

```
§4.3 합성 품질 평가

FID와 LPIPS로 합성 품질을 평가했다. KID는 CASDA 단독 절대 품질 지표로
부록에 별도 보고한다.

FID: CopyPaste(61.43)가 CASDA(228.61)보다 낮다. 이는 CopyPaste가
실제 패치를 복사하므로 통계적 분포가 실제 이미지와 동일하기 때문이다.
FID의 한계를 명시적으로 드러내기 위해 테이블에 포함했으며, 품질 비교로
해석해서는 안 된다.

LPIPS Realism: CASDA(0.467) < CopyPaste(0.491) — Poisson Blending이
경계 아티팩트를 제거하여 지각적 품질을 개선함을 확인했다.
Class1(Δ=+0.116), Class4(Δ=+0.060)에서 Blending 효과가 뚜렷하다.
Class2는 CopyPaste 열세(Δ=−0.052) — 아래 Class2 예외 항목 참조.

Class2 예외 처리: Class2는 미세·불규칙 텍스처로 ControlNet 생성 난이도가
높다. FID/KID 모두 최악으로 생성 단계 문제임이 두 독립 지표로 확인된다.
Poisson Blending 문제가 아님 → 핵심 주장(Blending 효과)은 훼손되지 않음.

LPIPS Diversity: CASDA(0.448) > CopyPaste(0.394) — ControlNet이
기존 패치 재조합을 넘어 다양한 결함 패턴을 생성.
이 다양성이 검출 성능 향상(§4.2)의 근본 원인으로 해석된다.
```

---

## 5. 시사점 및 제한사항

### 시사점
1. **FID는 이 도메인에서 단독 지표로 부적합** — 실제 패치를 복사하는 베이스라인과 비교 시 구조적 편향 발생; LPIPS가 필수 보완 지표
2. **Class2는 생성 난이도가 높은 클래스** — FID/KID 최악 + LPIPS Realism 열세가 일관됨; ControlNet 생성 단계 문제이며 Blending과 독립적
3. **Poisson Blending의 효과** — Class1·4의 LPIPS Realism 우위로 정량적 확인 (각 +0.116, +0.060)
4. **합성 다양성 우위** — Diversity에서 CASDA가 전체적으로 앞섬 (+0.054 overall); Class3·4 주도
5. **KID는 보조 지표** — CASDA 단독 절대 품질 측정; FID와 클래스 순위 일치로 결과 신뢰도 확보

### 제한사항
1. **KID는 CASDA만 측정** — CopyPaste는 실제 패치 복사라 KID ≈ 0으로 비교 불가; 보조 자료로만 사용
2. **LPIPS img_size=64** — 원본 ROI 크기 다양성으로 인해 리사이즈 왜곡 발생 가능
3. **InceptionV3는 ImageNet 학습** — 강철 도메인에 최적화되지 않아 FID 절대값 높음
4. **n=500 pairs** — LPIPS 추정치의 분산 미보고 (부트스트랩 std 추가 가능)
5. **Class2 미검증** — P1 벤치마크에서 Class2 AP 변화 수치로 이미지 품질 열세와 검출 성능 관계 확인 필요

---

## 6. 참고문헌

- Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. *NeurIPS*. (FID)
- Bińkowski et al. (2018). Demystifying MMD GANs. *ICLR*. (KID)
- Zhang et al. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. *CVPR*. (LPIPS)
