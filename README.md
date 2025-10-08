# 한국어 의료 상담 챗봇을 위한 경량화 LLM 구현 및 최적화

## 🎯 프로젝트 개요
42dot LLM을 베이스로 하여 의료 도메인에 특화된 경량화 모델을 구현하고, PEFT(LoRA) 기법을 활용한 효율적인 파인튜닝을 수행하는 프로젝트입니다.

## 🛠️ 기술적 접근법

### 1. 베이스 모델 선택
- **42dot/42dot_LLM-SFT-1.3B**: 한국어 특화 LLM 모델 활용
- 1.3B 파라미터로 적절한 크기와 성능의 균형

### 2. PEFT(LoRA) 기반 경량화
- **Parameter Efficient Fine-Tuning**: 전체 모델 대신 적응기만 학습
- **메모리 효율성**: M2 Pro Mac에서도 안정적 학습 가능
- **도메인 특화**: 의료 데이터로 효율적 파인튜닝

### 3. 의료 도메인 특화
- **AI Hub 의료 데이터**: 필수의료 + 전문의료 데이터 활용
- **주관식 데이터 필터링**: 객관식 제외하고 주관식만 학습
- **안전성 우선**: 의료 챗봇의 안전성 강조

## 📊 데이터 구성

### 의료 도메인 특화 데이터 (100%)
- **필수의료 의학지식 데이터**: 기본 의료 상담 능력 학습
- **전문 의학지식 데이터**: 고도의 전문 의학 지식 학습
- **주관식 데이터만 활용**: 객관식 제외하고 주관식 Q&A만 학습
- **Train/Validation/Test 분할**: 8:1:1 비율로 데이터 분할

## ✅ 완료된 작업

### Phase 1: 데이터 수집 및 전처리 ✅
- [x] AI Hub에서 의료 데이터 다운로드 및 품질 검증
- [x] 데이터 통합, 정제, 학습/검증/테스트 분할
- [x] 주관식 데이터 필터링 (객관식 제외)

### Phase 2: PEFT 기반 파인튜닝 ✅
- [x] 42dot LLM 모델 로드 및 성능 검증
- [x] PEFT(LoRA) 기법을 활용한 효율적 파인튜닝
- [x] M2 Pro Mac 환경에서 메모리 최적화

### Phase 3: 프롬프트 엔지니어링 및 평가 ✅
- [x] 3가지 프롬프트 전략 비교 (Basic, Improved, Optimized)
- [x] BERTScore 및 의료 특화 메트릭으로 성능 평가
- [x] 안전성 우선 프롬프트 최적화

## 📊 최종 성능 결과

### 🏆 프롬프트 성능 비교 (3가지 전략)

| 메트릭 | Basic | Improved | Optimized | 최고 성능 |
|--------|-------|----------|-----------|-----------|
| **BERTScore F1** | 0.6590 | 0.6527 | **0.6563** | 🥇 **Optimized** |
| **응답 길이** | **228.3자** | 282.3자 | 275.0자 | 🥇 **Basic** |
| **의료 용어 포함도** | 0.1058 | **0.1381** | 0.1159 | 🥇 **Improved** |
| **안전성 점수** | **0.814** | 0.704 | 0.740 | 🥇 **Basic** |
| **응답 품질** | 0.978 | 0.994 | **0.996** | 🥇 **Optimized** |

### 🎯 핵심 성과
- **PEFT 기법**: M2 Pro Mac에서 안정적 학습 완료
- **안전성 우선**: Basic 프롬프트가 가장 안전 (0.814)
- **정확도**: Optimized 프롬프트가 Basic에 근접한 성능
- **메모리 효율성**: 전체 모델 대신 적응기만 학습으로 메모리 절약

### 💡 최종 권장사항
**Basic 프롬프트 사용** - 의료 챗봇에서 안전성이 최우선이므로 가장 안전한 Basic 프롬프트를 권장합니다.

## 🚀 프로젝트 차별화 포인트

1. **PEFT 기법 활용**: 메모리 효율적인 파인튜닝으로 개인 환경에서도 학습 가능
2. **안전성 우선**: 의료 챗봇의 안전성을 최우선으로 고려한 프롬프트 설계
3. **체계적 평가**: BERTScore와 의료 특화 메트릭을 통한 정량적 성능 평가
4. **실용적 접근**: 복잡한 프롬프트보다 간단하고 안전한 프롬프트가 더 효과적임을 입증

## 📁 프로젝트 구조
```
chat/
├── data/                    # 데이터 저장소
│   ├── raw/                # 원본 데이터 (tar 파일)
│   ├── processed/          # 전처리된 데이터
│   └── splits/             # 학습/검증/테스트 분할
├── models/                 # 모델 저장소
│   └── medical_finetuned_peft/  # PEFT 파인튜닝된 모델
├── scripts/                # 실행 스크립트
│   ├── peft_medical_finetuning.py      # PEFT 파인튜닝
│   ├── peft_medical_evaluation.py      # 모델 평가
│   ├── improved_prompt_evaluation.py   # 개선된 프롬프트 평가
│   ├── optimized_prompt_evaluation.py  # 최적화된 프롬프트 평가
│   └── triple_prompt_comparison.py     # 3개 프롬프트 비교
├── results/                # 평가 결과
│   ├── peft_model_evaluation.json      # Basic 프롬프트 결과
│   ├── improved_prompt_evaluation.json # Improved 프롬프트 결과
│   └── optimized_prompt_evaluation.json # Optimized 프롬프트 결과
└── README.md               # 프로젝트 문서
```

## 🛠️ 설치 및 실행

### 환경 설정
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install torch transformers peft bert-score numpy tqdm
```

### 모델 파인튜닝
```bash
# PEFT 기반 파인튜닝
python scripts/peft_medical_finetuning.py
```

### 모델 평가
```bash
# Basic 프롬프트 평가
python scripts/peft_medical_evaluation.py

# Improved 프롬프트 평가
python scripts/improved_prompt_evaluation.py

# Optimized 프롬프트 평가
python scripts/optimized_prompt_evaluation.py

# 3개 프롬프트 비교
python scripts/triple_prompt_comparison.py
```

## 📈 성능 모니터링
- BERTScore를 통한 정확도 측정
- 의료 특화 메트릭 (안전성, 의료 용어 포함도 등)
- 3가지 프롬프트 전략 비교 분석

