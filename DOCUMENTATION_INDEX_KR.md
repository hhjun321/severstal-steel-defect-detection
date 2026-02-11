# 📚 CASDA 프로젝트 문서 인덱스
## Context-Aware Steel Defect Augmentation - 전체 문서 가이드

---

## 📋 목차

- [문서 개요](#문서-개요)
- [문서 유형별 분류](#문서-유형별-분류)
- [독자별 추천 문서](#독자별-추천-문서)
- [읽기 순서 가이드](#읽기-순서-가이드)
- [문서 요약](#문서-요약)

---

## 문서 개요

본 프로젝트는 **12개의 종합 문서**로 구성되어 있으며, 총 **500+ 페이지**에 달하는 상세한 기술 문서를 제공합니다.

### 문서 통계

| 문서 유형 | 개수 | 총 페이지 | 언어 |
|----------|------|----------|------|
| 연구 논문 | 2 | 150+ | KR, EN |
| 기술 문서 | 6 | 200+ | KR, EN |
| 가이드 | 4 | 150+ | KR, EN |
| 총계 | 12 | 500+ | 2개 언어 |

---

## 문서 유형별 분류

### 1️⃣ 연구 문서 (Academic Papers)

학술적 깊이와 방법론을 중심으로 작성된 연구 논문 형식의 문서입니다.

| 문서명 | 페이지 | 언어 | 대상 독자 |
|--------|--------|------|-----------|
| **RESEARCH_REPORT_KR.md** | 80+ | 한국어 | 연구원, 학계 |
| **RESEARCH_REPORT_EN.md** | 80+ | English | Researchers, Academia |

**내용**:
- 초록, 서론, 관련 연구
- 방법론 상세 (5단계 파이프라인)
- 실험 결과 및 분석
- 토론 및 결론
- 참고문헌 및 부록

**읽기 시간**: 각 2-3시간

---

### 2️⃣ 기술 문서 (Technical Documentation)

구현 상세와 API 레퍼런스를 포함한 개발자용 문서입니다.

#### A. 기술 백서

| 문서명 | 페이지 | 언어 | 대상 독자 |
|--------|--------|------|-----------|
| **TECHNICAL_WHITEPAPER_KR.md** | 100+ | 한국어 | 아키텍트, 엔지니어 |

**내용**:
- 시스템 아키텍처
- 핵심 알고리즘 상세
- 코드 레벨 구현
- API 레퍼런스
- 성능 최적화
- 배포 가이드
- 문제 해결
- 확장성

**읽기 시간**: 3-4시간

#### B. 구현 요약

| 문서명 | 페이지 | 언어 | 주제 |
|--------|--------|------|------|
| **IMPLEMENTATION_SUMMARY_KR.md** | 10 | 한국어 | Stage 1: ROI 추출 |
| **IMPLEMENTATION_CONTROLNET_PREP_KR.md** | 10 | 한국어 | Stage 2: ControlNet 준비 |

**내용**:
- 각 단계별 구현 요약
- 모듈 구조
- 사용 방법
- 출력 형식

**읽기 시간**: 각 20-30분

#### C. 전체 파이프라인

| 문서명 | 페이지 | 언어 |
|--------|--------|------|
| **README_COMPLETE_PIPELINE_KR.md** | 25 | 한국어 |
| **README_COMPLETE_PIPELINE.md** | 25 | English |

**내용**:
- 5단계 파이프라인 통합
- 빠른 시작 가이드
- 프로젝트 구조
- 주요 기능
- 실행 세부사항
- 문제 해결

**읽기 시간**: 각 1시간

---

### 3️⃣ 사용자 가이드 (User Guides)

실무 적용을 위한 실용적인 가이드입니다.

#### A. 단계별 가이드

| 문서명 | 페이지 | 언어 | 주제 |
|--------|--------|------|------|
| **README_ROI_KR.md** | 20 | 한국어 | ROI 추출 가이드 |
| **README_ROI.md** | 20 | English | ROI Extraction Guide |
| **README_AUGMENTATION_KR.md** | 20 | 한국어 | 증강 파이프라인 |
| **README_AUGMENTATION.md** | 20 | English | Augmentation Pipeline |

**내용**:
- 개요 및 빠른 시작
- 상세 사용 방법
- 옵션 및 파라미터
- 출력 검증
- 시각화
- 테스트

**읽기 시간**: 각 45-60분

#### B. 종합 가이드

| 문서명 | 페이지 | 언어 |
|--------|--------|------|
| **AUGMENTATION_PIPELINE_GUIDE_KR.md** | 30 | 한국어 |
| **AUGMENTATION_PIPELINE_GUIDE.md** | 30 | English |

**내용**:
- 5단계 상세 설명
- 매개변수 참조
- 품질 관리
- 문제 해결
- 성능 벤치마크
- 고급 구성

**읽기 시간**: 각 1.5-2시간

---

### 4️⃣ 경영 문서 (Executive Documents)

의사결정자를 위한 요약 문서입니다.

| 문서명 | 페이지 | 언어 | 대상 독자 |
|--------|--------|------|-----------|
| **EXECUTIVE_SUMMARY_KR.md** | 15 | 한국어 | 경영진, 의사결정자 |

**내용**:
- 프로젝트 개요
- 비즈니스 성과
- 비용 효율성
- ROI 분석
- 리스크 및 제약사항
- 권장 사항

**읽기 시간**: 30-45분

---

### 5️⃣ 프레젠테이션 (Presentations)

발표용 슬라이드 형식 문서입니다.

| 문서명 | 슬라이드 | 언어 | 발표 시간 |
|--------|----------|------|-----------|
| **PRESENTATION_SLIDES_KR.md** | 22 | 한국어 | 45-60분 |

**내용**:
- 문제 인식
- 솔루션 개요
- 핵심 기술 (5개)
- 실험 결과
- 비교 분석
- 로드맵 및 실행 계획

**읽기 시간**: 1시간 (발표 + Q&A)

---

## 독자별 추천 문서

### 👔 경영진 / 의사결정자

**필수 문서** (총 1-2시간):
1. **EXECUTIVE_SUMMARY_KR.md** (30-45분)
   - 비즈니스 가치 및 ROI
2. **PRESENTATION_SLIDES_KR.md** (45-60분)
   - 시각적 요약 및 주요 결과

**선택 문서**:
- **RESEARCH_REPORT_KR.md** - 1장 (서론) 및 6장 (결론)

---

### 🔬 연구원 / 학계

**필수 문서** (총 4-5시간):
1. **RESEARCH_REPORT_KR.md** 또는 **EN.md** (2-3시간)
   - 완전한 방법론 및 실험 결과
2. **TECHNICAL_WHITEPAPER_KR.md** - 3장 (핵심 알고리즘) (1-2시간)
   - 수학적 정의 및 알고리즘

**선택 문서**:
- **IMPLEMENTATION_SUMMARY_KR.md**
- **IMPLEMENTATION_CONTROLNET_PREP_KR.md**

---

### 💻 AI/ML 엔지니어

**필수 문서** (총 5-6시간):
1. **TECHNICAL_WHITEPAPER_KR.md** (3-4시간)
   - 시스템 아키텍처 및 구현 상세
2. **README_COMPLETE_PIPELINE_KR.md** (1시간)
   - 전체 파이프라인 이해
3. **AUGMENTATION_PIPELINE_GUIDE_KR.md** (1.5-2시간)
   - 실전 가이드

**선택 문서**:
- **RESEARCH_REPORT_KR.md** - 3장 (방법론)
- **IMPLEMENTATION_SUMMARY_KR.md**

---

### 🛠️ 소프트웨어 엔지니어 / 운영

**필수 문서** (총 3-4시간):
1. **README_COMPLETE_PIPELINE_KR.md** (1시간)
   - 빠른 시작 및 프로젝트 구조
2. **AUGMENTATION_PIPELINE_GUIDE_KR.md** (1.5-2시간)
   - 상세 실행 가이드
3. **TECHNICAL_WHITEPAPER_KR.md** - 7장 (배포) 및 9장 (문제 해결) (1시간)

**선택 문서**:
- **README_ROI_KR.md**
- **README_AUGMENTATION_KR.md**

---

### 🎓 학생 / 입문자

**추천 순서** (총 4-5시간):
1. **EXECUTIVE_SUMMARY_KR.md** (30분)
   - 프로젝트 개요 파악
2. **PRESENTATION_SLIDES_KR.md** (1시간)
   - 시각적 이해
3. **README_COMPLETE_PIPELINE_KR.md** (1시간)
   - 전체 구조 학습
4. **README_ROI_KR.md** (1시간)
   - Stage 1 상세 학습
5. **README_AUGMENTATION_KR.md** (1.5시간)
   - Stage 3-5 상세 학습

**선택 문서**:
- **RESEARCH_REPORT_KR.md** - 2장 (관련 연구)

---

## 읽기 순서 가이드

### 🎯 목적별 추천 순서

#### 빠른 이해 (1-2시간)

```
1. EXECUTIVE_SUMMARY_KR.md (30분)
   ↓
2. PRESENTATION_SLIDES_KR.md - 슬라이드 1-10 (30분)
   ↓
3. README_COMPLETE_PIPELINE_KR.md - 빠른 시작 섹션 (30분)
```

#### 실무 적용 (3-4시간)

```
1. README_COMPLETE_PIPELINE_KR.md (1시간)
   ↓
2. AUGMENTATION_PIPELINE_GUIDE_KR.md (1.5-2시간)
   ↓
3. TECHNICAL_WHITEPAPER_KR.md - 9장 (문제 해결) (30분)
   ↓
4. 실행 및 테스트
```

#### 심층 연구 (8-10시간)

```
1. RESEARCH_REPORT_KR.md (2-3시간)
   ↓
2. TECHNICAL_WHITEPAPER_KR.md (3-4시간)
   ↓
3. IMPLEMENTATION_SUMMARY_KR.md (30분)
   ↓
4. IMPLEMENTATION_CONTROLNET_PREP_KR.md (30분)
   ↓
5. 전체 가이드 문서 검토 (2-3시간)
```

#### 전체 마스터 (15-20시간)

```
Phase 1: 개념 이해 (3-4시간)
  1. EXECUTIVE_SUMMARY_KR.md
  2. PRESENTATION_SLIDES_KR.md
  3. README_COMPLETE_PIPELINE_KR.md

Phase 2: 학술적 이해 (2-3시간)
  4. RESEARCH_REPORT_KR.md

Phase 3: 기술적 심화 (3-4시간)
  5. TECHNICAL_WHITEPAPER_KR.md

Phase 4: 구현 상세 (2-3시간)
  6. IMPLEMENTATION_SUMMARY_KR.md
  7. IMPLEMENTATION_CONTROLNET_PREP_KR.md
  8. README_ROI_KR.md

Phase 5: 실무 가이드 (4-5시간)
  9. AUGMENTATION_PIPELINE_GUIDE_KR.md
  10. README_AUGMENTATION_KR.md

Phase 6: 실습 (1-2시간)
  11. 코드 실행 및 테스트
```

---

## 문서 요약

### 📄 RESEARCH_REPORT_KR.md / EN.md

**유형**: 연구 논문  
**페이지**: 80+  
**대상**: 연구원, 학계

**장점**:
- ✅ 완전한 학술 형식 (초록, 서론, 방법론, 결과, 결론)
- ✅ 수학적 정의 및 증명
- ✅ 참고문헌 포함
- ✅ 재현 가능한 실험 설계

**핵심 내용**:
- 4가지 통계 지표 기반 결함 특성화
- 그리드 기반 배경 분석
- 결함-배경 매칭 시스템
- 5단계 파이프라인
- 실험 결과 및 분석

---

### 📄 TECHNICAL_WHITEPAPER_KR.md

**유형**: 기술 백서  
**페이지**: 100+  
**대상**: 아키텍트, AI/ML 엔지니어

**장점**:
- ✅ 코드 레벨 구현 상세
- ✅ 알고리즘 의사코드 포함
- ✅ API 레퍼런스
- ✅ 성능 최적화 기법
- ✅ 배포 가이드
- ✅ 문제 해결 섹션

**핵심 내용**:
- 시스템 아키텍처 (모듈 구조)
- 핵심 알고리즘 (Linearity, Solidity 등)
- 단계별 구현 (코드 포함)
- 성능 최적화 (병렬 처리, GPU)
- 품질 보장 (단위 테스트, 회귀 테스트)
- 배포 (Docker, Kubernetes)

---

### 📄 EXECUTIVE_SUMMARY_KR.md

**유형**: 경영 보고서  
**페이지**: 15  
**대상**: 경영진, 의사결정자

**장점**:
- ✅ 비즈니스 가치 중심
- ✅ ROI 분석 포함
- ✅ 리스크 평가
- ✅ 실행 계획 제시
- ✅ 간결한 요약

**핵심 내용**:
- 비즈니스 과제 및 솔루션
- 정량적 성과 (16.5% 증가, 83% 품질)
- 비용 효율성 ($35K-$166K 절감)
- ROI 분석 (189%, 1년 내 회수)
- 리스크 및 완화 전략
- 권장 사항 (3단계 실행 계획)

---

### 📄 PRESENTATION_SLIDES_KR.md

**유형**: 프레젠테이션  
**슬라이드**: 22  
**대상**: 전체 (발표용)

**장점**:
- ✅ 시각적 표현
- ✅ 핵심만 집중
- ✅ 비기술적 청중도 이해 가능
- ✅ 45-60분 발표 최적화

**핵심 내용**:
- 문제 인식 (3가지 과제)
- 솔루션 개요
- 5가지 핵심 기술
- 실험 결과 (그래프 포함)
- 비교 분석 (vs 기존 기법)
- 성공 사례
- 로드맵 및 실행 계획

---

### 📄 README_COMPLETE_PIPELINE_KR.md / EN.md

**유형**: 통합 가이드  
**페이지**: 25  
**대상**: 전체 (입문용)

**장점**:
- ✅ 전체 파이프라인 조망
- ✅ 빠른 시작 가이드
- ✅ 프로젝트 구조 명확
- ✅ 실행 예제 포함

**핵심 내용**:
- 프로젝트 개요
- 5단계 파이프라인 설명
- 빠른 시작 (설치 및 실행)
- 프로젝트 구조
- 주요 기능
- 데이터셋 통계
- 문제 해결
- 다음 단계

---

### 📄 AUGMENTATION_PIPELINE_GUIDE_KR.md / EN.md

**유형**: 상세 가이드  
**페이지**: 30  
**대상**: 실무자

**장점**:
- ✅ 단계별 상세 설명
- ✅ 매개변수 완전 참조
- ✅ 품질 관리 상세
- ✅ 문제 해결 섹션
- ✅ 성능 벤치마크

**핵심 내용**:
- 5단계 상세 가이드
- 빠른 시작 vs 수동 실행
- 매개변수 참조표
- 출력 구조
- 품질 검증 기준
- 문제 해결 (일반적 오류)
- 성능 벤치마크
- 고급 구성

---

### 📄 README_ROI_KR.md / EN.md

**유형**: 단계별 가이드  
**페이지**: 20  
**대상**: Stage 1 학습자

**장점**:
- ✅ ROI 추출 집중
- ✅ 실행 예제 다양
- ✅ 출력 형식 상세
- ✅ 메타데이터 설명

**핵심 내용**:
- ROI 추출 연구 개요
- 4가지 지표 설명
- 배경 분석 (그리드 기반)
- 적합도 평가
- 사용법 (CLI)
- 출력 데이터 형식
- 통계 요약 예시

---

### 📄 README_AUGMENTATION_KR.md / EN.md

**유형**: 단계별 가이드  
**페이지**: 20  
**대상**: Stage 3-5 학습자

**장점**:
- ✅ 증강 생성 집중
- ✅ 빠른 시작 3가지 옵션
- ✅ 검증 방법 상세
- ✅ 시각화 도구 설명

**핵심 내용**:
- 증강 파이프라인 개요 (5단계)
- 사전 요구사항 체크리스트
- 빠른 시작 (자동/수동/테스트)
- 주요 출력 파일
- 검증 방법
- 시각화
- 테스트
- 훈련에서 사용법

---

### 📄 IMPLEMENTATION_SUMMARY_KR.md

**유형**: 구현 요약  
**페이지**: 10  
**대상**: Stage 1 개발자

**핵심 내용**:
- ROI 추출 파이프라인
- 모듈 구조 (3개 분석 모듈)
- 사용 방법
- 출력 데이터
- 핵심 기여

---

### 📄 IMPLEMENTATION_CONTROLNET_PREP_KR.md

**유형**: 구현 요약  
**페이지**: 10  
**대상**: Stage 2 개발자

**핵심 내용**:
- ControlNet 데이터 준비
- 다중 채널 힌트 생성
- 하이브리드 프롬프트
- 데이터셋 검증
- 패키징 형식

---

## 📌 빠른 참조

### 즉시 시작하고 싶다면?

```
1. README_COMPLETE_PIPELINE_KR.md - 빠른 시작 섹션 (5분)
2. 코드 실행 (2시간)
3. 결과 확인 (30분)
```

### 논문 작성 참고가 필요하다면?

```
RESEARCH_REPORT_KR.md 또는 EN.md
```

### API 레퍼런스가 필요하다면?

```
TECHNICAL_WHITEPAPER_KR.md - 8장 (API 레퍼런스)
```

### 문제 해결이 필요하다면?

```
TECHNICAL_WHITEPAPER_KR.md - 9장 (문제 해결)
AUGMENTATION_PIPELINE_GUIDE_KR.md - 문제 해결 섹션
```

### 비즈니스 가치 설명이 필요하다면?

```
EXECUTIVE_SUMMARY_KR.md - 3장 (비즈니스 성과)
```

### 발표 준비가 필요하다면?

```
PRESENTATION_SLIDES_KR.md (22 슬라이드)
```

---

## 📊 문서 통계 요약

| 항목 | 수치 |
|------|------|
| 총 문서 수 | 12개 |
| 총 페이지 수 | 500+ |
| 지원 언어 | 한국어, English |
| 코드 예제 | 100+ |
| 다이어그램 | 50+ |
| 표 및 그래프 | 80+ |
| 참고문헌 | 10+ |

---

## 🔄 문서 업데이트

**최종 업데이트**: 2026년 2월 9일  
**버전**: 1.0  
**상태**: 완료 (100%)

**업데이트 예정**:
- Phase 2: 성능 개선 후 벤치마크 업데이트 (3-6개월)
- Phase 3: 실제 배포 후 케이스 스터디 추가 (6-12개월)

---

## 📞 문의

**문서 관련 문의**:
- 이메일: [documentation@casda-project.org]
- GitHub Issues: [저장소 URL]/issues

**기술 지원**:
- 이메일: [technical-support@casda-project.org]

---

**문서 작성**: CASDA Project Team  
**문서 버전**: 1.0  
**라이선스**: MIT License
