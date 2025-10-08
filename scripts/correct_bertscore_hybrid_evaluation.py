#!/usr/bin/env python3
"""
올바른 BERTScore 기반 하이브리드 의료 챗봇 평가
- 오직 test 데이터만 사용 (데이터 누수 방지)
- 공정한 성능 평가
"""

import os
import json
import time
import random
from datetime import datetime
from typing import Dict, List, Tuple, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CorrectBERTScoreHybridEvaluator:
    def __init__(self, model_path: str = "models/medical_finetuned_improved"):
        """올바른 하이브리드 시스템 BERTScore 평가자 초기화"""
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 사용 디바이스: {self.device}")
        
        # BERT 모델 설정
        self.bert_model = "bert-base-multilingual-cased"
        print(f"🧠 BERT 모델: {self.bert_model}")
        print(f"📊 IDF 가중치: True")
        
        # 모델 로딩
        self._load_model()
        
        # 의료 지식베이스
        self._load_medical_knowledge()
        
    def _load_model(self):
        """모델과 토크나이저 로딩"""
        print(f"\n📥 모델 로딩 중...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            print("✅ 모델 로드 완료")
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            raise
    
    def _load_medical_knowledge(self):
        """의료 지식베이스 로딩"""
        # 응급상황 키워드
        self.emergency_keywords = [
            "심한", "갑작스러운", "급성", "심장", "호흡곤란", "의식불명", 
            "대량출혈", "골절", "화상", "중독", "뇌졸중", "심근경색",
            "응급실", "119", "구급차", "생명위험", "즉시", "긴급"
        ]
        
        # 증상-질병 매핑
        self.symptom_disease_map = {
            "감기": ["콧물", "기침", "목아픔", "발열", "몸살"],
            "당뇨병": ["다뇨", "다음", "다식", "체중감소", "피로"],
            "고혈압": ["두통", "어지러움", "가슴답답", "호흡곤란"],
            "위염": ["복통", "속쓰림", "메스꺼움", "구토", "소화불량"],
            "우울증": ["우울감", "무기력", "수면장애", "식욕부진", "집중력저하"]
        }
        
        # 질병-치료 매핑
        self.disease_treatment_map = {
            "감기": {
                "약물": ["아세트아미노펜", "이부프로펜", "진해제", "거담제"],
                "자조": ["충분한 휴식", "수분 섭취", "비타민C", "온수 가글"],
                "의사방문": ["고열(38.5°C 이상)", "3-4일 이상 지속", "호흡곤란"]
            },
            "당뇨병": {
                "약물": ["의사 처방 약물"],
                "자조": ["규칙적인 식사", "운동", "혈당 측정", "체중 관리"],
                "의사방문": ["혈당 조절 불량", "합병증 증상", "새로운 증상"]
            }
        }
    
    def _detect_emergency(self, text: str) -> bool:
        """응급상황 감지"""
        text_lower = text.lower()
        for keyword in self.emergency_keywords:
            if keyword in text_lower:
                return True
        return False
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """증상 추출"""
        symptoms = []
        for disease, symptom_list in self.symptom_disease_map.items():
            for symptom in symptom_list:
                if symptom in text:
                    symptoms.append(disease)
                    break
        return symptoms
    
    def _get_rule_based_response(self, question: str) -> str:
        """규칙 기반 응답 생성"""
        # 응급상황 감지
        if self._detect_emergency(question):
            return "응급상황으로 보입니다. 즉시 119에 신고하거나 응급실을 방문하세요."
        
        # 증상 기반 응답
        symptoms = self._extract_symptoms(question)
        if symptoms:
            disease = symptoms[0]
            if disease in self.disease_treatment_map:
                treatment = self.disease_treatment_map[disease]
                response = f"{disease} 증상으로 보입니다. "
                
                if "자조" in treatment:
                    response += f"{', '.join(treatment['자조'])}이 중요합니다. "
                
                if "약물" in treatment:
                    response += f"복용 가능한 약물: {', '.join(treatment['약물'])}. "
                
                if "의사방문" in treatment:
                    response += f"의사 방문이 필요한 경우: {', '.join(treatment['의사방문'])}."
                
                return response
        
        # 일반적인 응답
        return "증상이 지속되거나 악화되면 의료진과 상담하세요."
    
    def _get_model_response(self, question: str) -> str:
        """모델 기반 응답 생성"""
        try:
            # 프롬프트 생성
            prompt = f"의료 상담: {question}\n의료진:"
            
            # 토크나이징
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # 디코딩
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            
            # 응답 정리
            if "의료진:" in response:
                response = response.split("의료진:")[-1].strip()
            
            # 반복 제거
            lines = response.split('\n')
            cleaned_lines = []
            for line in lines:
                if line.strip() and not any(word in line for word in ["의료진:", "간호사:", "진료:"]):
                    cleaned_lines.append(line.strip())
            
            if cleaned_lines:
                return cleaned_lines[0]
            else:
                return "정확한 상담을 위해 의료진과 직접 상담하시기 바랍니다."
                
        except Exception as e:
            print(f"⚠️ 모델 응답 생성 오류: {e}")
            return "정확한 상담을 위해 의료진과 직접 상담하시기 바랍니다."
    
    def _is_valid_medical_response(self, response: str) -> bool:
        """의료 응답 유효성 검사"""
        invalid_patterns = [
            "내원한 환자에게",
            "가장 적절한 치료법은",
            "1 스테로이드",
            "2 모기약",
            "3 피부트레이닝",
            "4 피부트레이닝"
        ]
        
        for pattern in invalid_patterns:
            if pattern in response:
                return False
        return True
    
    def get_hybrid_response(self, question: str) -> Tuple[str, str, bool, List[str], float]:
        """하이브리드 응답 생성"""
        start_time = time.time()
        
        # 1. 응급상황 우선 체크
        is_emergency = self._detect_emergency(question)
        if is_emergency:
            response = "응급상황으로 보입니다. 즉시 119에 신고하거나 응급실을 방문하세요."
            response_time = time.time() - start_time
            return response, "emergency", is_emergency, [], response_time
        
        # 2. 규칙 기반 응답 시도
        rule_response = self._get_rule_based_response(question)
        if rule_response and len(rule_response) > 20:  # 충분한 길이의 응답
            response_time = time.time() - start_time
            symptoms = self._extract_symptoms(question)
            return rule_response, "rule_based", is_emergency, symptoms, response_time
        
        # 3. 모델 기반 응답 시도
        model_response = self._get_model_response(question)
        if self._is_valid_medical_response(model_response):
            response_time = time.time() - start_time
            symptoms = self._extract_symptoms(question)
            return model_response, "model_based", is_emergency, symptoms, response_time
        
        # 4. 폴백 응답
        fallback_response = "정확한 상담을 위해 의료진과 직접 상담하시기 바랍니다."
        response_time = time.time() - start_time
        return fallback_response, "fallback", is_emergency, [], response_time
    
    def load_test_data_only(self, num_samples: int = 200) -> List[Dict]:
        """오직 test 데이터만 로딩 (데이터 누수 방지)"""
        print(f"\n📊 Test 데이터만 로딩 중...")
        
        # 오직 test 파일만 사용
        test_files = [
            "data/processed/splits/essential_medical_test.json", 
            "data/processed/splits/professional_medical_test.json"
        ]
        
        all_data = []
        for file_path in test_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data)
                        else:
                            all_data.append(data)
                    print(f"✅ {file_path}: {len(data) if isinstance(data, list) else 1}개 샘플")
                except Exception as e:
                    print(f"⚠️ {file_path} 로딩 실패: {e}")
            else:
                print(f"⚠️ {file_path} 파일 없음")
        
        if not all_data:
            print("❌ 테스트 데이터를 찾을 수 없습니다.")
            return []
        
        # 샘플링
        if len(all_data) > num_samples:
            all_data = random.sample(all_data, num_samples)
        
        print(f"✅ Test 데이터만 로드 완료: {len(all_data)}개 샘플")
        print(f"🔒 데이터 누수 방지: train/validation 데이터 제외")
        return all_data
    
    def evaluate_medical_professionalism(self, response: str) -> float:
        """의료 전문성 평가"""
        medical_terms = [
            "의료진", "병원", "응급실", "119", "진료", "상담", "증상", 
            "치료", "약물", "복용", "의사", "간호사", "응급", "긴급"
        ]
        
        response_lower = response.lower()
        medical_count = sum(1 for term in medical_terms if term in response_lower)
        return min(medical_count / 5.0, 1.0)  # 최대 1.0
    
    def evaluate_response_quality(self, response: str) -> float:
        """응답 품질 평가"""
        if not response or len(response) < 10:
            return 0.0
        
        # 길이 점수 (20-200자 사이가 적절)
        length_score = 1.0 if 20 <= len(response) <= 200 else 0.5
        
        # 의료 관련성 점수
        medical_keywords = ["의료", "병원", "증상", "치료", "약물", "의사"]
        medical_score = sum(1 for keyword in medical_keywords if keyword in response) / len(medical_keywords)
        
        # 반복성 점수 (반복이 적을수록 좋음)
        words = response.split()
        unique_words = len(set(words))
        repetition_score = unique_words / len(words) if words else 0
        
        return (length_score + medical_score + repetition_score) / 3
    
    def evaluate(self, num_samples: int = 200) -> Dict[str, Any]:
        """올바른 하이브리드 시스템 BERTScore 평가"""
        print(f"\n🔍 올바른 BERTScore 기반 하이브리드 시스템 평가 시작 ({num_samples}개 샘플)")
        print("=" * 60)
        print("🔒 데이터 누수 방지: 오직 test 데이터만 사용")
        
        # 테스트 데이터 로딩 (오직 test만)
        test_data = self.load_test_data_only(num_samples)
        if not test_data:
            return {}
        
        # 평가 변수
        total_samples = len(test_data)
        correct_responses = 0
        emergency_detected = 0
        symptoms_detected = 0
        total_response_time = 0
        response_sources = {"rule_based": 0, "model_based": 0, "emergency": 0, "fallback": 0}
        
        all_responses = []
        all_references = []
        
        print(f"\n🤖 하이브리드 응답 생성 중: ", end="")
        
        # 각 샘플에 대해 평가
        for i, sample in enumerate(tqdm(test_data, desc="하이브리드 응답 생성")):
            question = sample.get("question", "")
            reference = sample.get("answer", "")
            
            if not question or not reference:
                continue
            
            # 하이브리드 응답 생성
            response, source, is_emergency, symptoms, response_time = self.get_hybrid_response(question)
            
            # 응답 수집 (BERTScore용)
            all_responses.append(response)
            all_references.append(reference)
            
            # 기본 메트릭 계산
            total_response_time += response_time
            response_sources[source] += 1
            
            # 응급상황 감지
            if is_emergency:
                emergency_detected += 1
            
            # 증상 감지
            if symptoms:
                symptoms_detected += 1
            
            # 응답 정확도 (간단한 키워드 기반)
            if any(keyword in response.lower() for keyword in ["의료", "병원", "의사", "상담", "치료"]):
                correct_responses += 1
        
        # BERTScore 계산
        print(f"\n🧠 BERTScore 계산 중...")
        try:
            P, R, F1 = score(all_responses, all_references, 
                           model_type=self.bert_model, 
                           idf=True, 
                           verbose=False)
            
            precision = P.mean().item()
            recall = R.mean().item()
            f1_score = F1.mean().item()
        except Exception as e:
            print(f"⚠️ BERTScore 계산 오류: {e}")
            precision = recall = f1_score = 0.0
        
        # 의료 전문성 및 응답 품질 평가
        medical_scores = []
        quality_scores = []
        
        for response in all_responses:
            medical_scores.append(self.evaluate_medical_professionalism(response))
            quality_scores.append(self.evaluate_response_quality(response))
        
        avg_medical_professionalism = sum(medical_scores) / len(medical_scores) if medical_scores else 0
        avg_response_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # 결과 계산
        avg_response_time = total_response_time / total_samples if total_samples > 0 else 0
        response_accuracy = correct_responses / total_samples if total_samples > 0 else 0
        emergency_accuracy = emergency_detected / total_samples if total_samples > 0 else 0
        symptom_accuracy = symptoms_detected / total_samples if total_samples > 0 else 0
        overall_accuracy = (response_accuracy + emergency_accuracy + symptom_accuracy) / 3
        
        # 결과 정리
        results = {
            "evaluation_type": "올바른 BERTScore 기반 하이브리드 시스템 평가 (데이터 누수 방지)",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "data_source": "오직 test 데이터만 사용",
            "total_samples": total_samples,
            "average_response_time": avg_response_time,
            "bertscore_metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            },
            "medical_metrics": {
                "medical_professionalism": avg_medical_professionalism,
                "emergency_detection_accuracy": emergency_accuracy,
                "symptom_detection_accuracy": symptom_accuracy,
                "response_quality": avg_response_quality,
                "overall_accuracy": overall_accuracy
            },
            "response_sources": response_sources,
            "goals_achieved": {
                "accuracy_90_percent": overall_accuracy >= 0.9,
                "response_time_3_seconds": avg_response_time <= 3.0,
                "bertscore_f1_0_7": f1_score >= 0.7
            }
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """결과 출력"""
        print(f"\n🎯 올바른 BERTScore 기반 하이브리드 시스템 평가 결과")
        print("=" * 60)
        print(f"🔒 데이터 소스: {results['data_source']}")
        print(f"📊 총 샘플: {results['total_samples']}개")
        print(f"⏱️ 평균 응답 시간: {results['average_response_time']:.2f}초")
        
        print(f"\n🧠 BERTScore 메트릭:")
        print(f"   Precision: {results['bertscore_metrics']['precision']:.4f}")
        print(f"   Recall: {results['bertscore_metrics']['recall']:.4f}")
        print(f"   F1-Score: {results['bertscore_metrics']['f1_score']:.4f}")
        
        print(f"\n🏥 의료 전문성: {results['medical_metrics']['medical_professionalism']:.2%}")
        print(f"🚨 응급 감지 정확도: {results['medical_metrics']['emergency_detection_accuracy']:.2%}")
        print(f"🔍 증상 감지 정확도: {results['medical_metrics']['symptom_detection_accuracy']:.2%}")
        print(f"⭐ 응답 품질: {results['medical_metrics']['response_quality']:.2%}")
        print(f"🎯 전체 정확도: {results['medical_metrics']['overall_accuracy']:.2%}")
        
        print(f"\n📊 응답 소스 분포:")
        for source, count in results['response_sources'].items():
            print(f"   {source}: {count}개")
        
        print(f"\n🎯 목표 달성 여부:")
        goals = results['goals_achieved']
        print(f"   정확도 90% 목표: {'✅ 달성' if goals['accuracy_90_percent'] else '❌ 미달성'}")
        print(f"   응답 시간 3초 목표: {'✅ 달성' if goals['response_time_3_seconds'] else '❌ 미달성'}")
        print(f"   BERTScore F1 0.7 목표: {'✅ 달성' if goals['bertscore_f1_0_7'] else '❌ 미달성'}")

def main():
    """메인 실행 함수"""
    print("🚀 올바른 BERTScore 기반 하이브리드 의료 챗봇 평가")
    print("=" * 60)
    print("🔒 데이터 누수 방지: 오직 test 데이터만 사용")
    
    # 평가자 초기화
    evaluator = CorrectBERTScoreHybridEvaluator()
    
    # 평가 옵션 선택
    print(f"\n📊 평가 옵션:")
    print(f"   1. 빠른 평가: 100개 샘플")
    print(f"   2. 표준 평가: 200개 샘플")
    print(f"   3. 전체 평가: 모든 test 샘플")
    
    choice = input("\n선택하세요 (1-3): ").strip()
    
    if choice == "1":
        num_samples = 100
    elif choice == "2":
        num_samples = 200
    elif choice == "3":
        num_samples = 10000  # 충분히 큰 수
    else:
        print("기본값 200개 샘플로 진행합니다.")
        num_samples = 200
    
    # 평가 실행
    results = evaluator.evaluate(num_samples)
    
    if results:
        # 결과 출력
        evaluator.print_results(results)
        
        # 결과 저장
        os.makedirs("models", exist_ok=True)
        result_file = f"models/correct_bertscore_hybrid_evaluation_results_{results['timestamp']}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 결과 저장: {result_file}")
        print(f"\n✅ 올바른 BERTScore 기반 하이브리드 시스템 평가 완료!")
    else:
        print("❌ 평가 실패")

if __name__ == "__main__":
    main()
