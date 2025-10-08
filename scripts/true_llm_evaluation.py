#!/usr/bin/env python3
"""
진짜 LLM 챗봇 평가 스크립트
- test 데이터만 사용 (data leakage 방지)
- 실제 모델 성능 측정
- 정량적/정성적 평가
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

class TrueLLMEvaluator:
    def __init__(self, model_path: str = "models/true_llm_chatbot"):
        """진짜 LLM 챗봇 평가자 초기화"""
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 사용 디바이스: {self.device}")
        
        # BERT 모델 설정
        self.bert_model = "bert-base-multilingual-cased"
        print(f"🧠 BERT 모델: {self.bert_model}")
        
        # 모델 로딩
        self._load_model()
        
    def _load_model(self):
        """모델과 토크나이저 로딩"""
        print(f"\n📥 모델 로딩 중: {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print("✅ 모델 로드 완료")
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            print("기본 모델을 사용합니다.")
            self._load_base_model()
    
    def _load_base_model(self):
        """기본 모델 로딩"""
        print(f"\n📥 기본 모델 로딩 중: skt/kogpt2-base-v2")
        
        self.tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
        self.model = AutoModelForCausalLM.from_pretrained(
            "skt/kogpt2-base-v2",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("✅ 기본 모델 로드 완료")
    
    def generate_response(self, question: str) -> str:
        """모델 기반 응답 생성"""
        try:
            # 프롬프트 생성
            prompt = f"사용자: {question}\n의료진:"
            
            # 토크나이징
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 200,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs),
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
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
                line = line.strip()
                if line and not any(word in line for word in ["사용자:", "의료진:", "간호사:", "진료:"]):
                    cleaned_lines.append(line)
            
            if cleaned_lines:
                return cleaned_lines[0]
            else:
                return "죄송합니다. 질문을 이해하지 못했습니다."
                
        except Exception as e:
            print(f"⚠️ 응답 생성 오류: {e}")
            return "죄송합니다. 일시적인 오류가 발생했습니다."
    
    def load_test_data(self, num_samples: int = 200) -> List[Dict]:
        """test 데이터만 로딩 (data leakage 방지)"""
        print(f"\n📊 Test 데이터만 로딩 중...")
        
        # 오직 test 파일만 사용
        test_files = [
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
            "의료", "병원", "의사", "진료", "상담", "증상", "치료", "약물", 
            "복용", "응급", "긴급", "진단", "질병", "환자", "건강"
        ]
        
        response_lower = response.lower()
        medical_count = sum(1 for term in medical_terms if term in response_lower)
        return min(medical_count / 5.0, 1.0)
    
    def evaluate_response_quality(self, response: str) -> float:
        """응답 품질 평가"""
        if not response or len(response) < 5:
            return 0.0
        
        # 길이 점수 (10-300자 사이가 적절)
        length_score = 1.0 if 10 <= len(response) <= 300 else 0.5
        
        # 의료 관련성 점수
        medical_keywords = ["의료", "병원", "증상", "치료", "약물", "의사"]
        medical_score = sum(1 for keyword in medical_keywords if keyword in response) / len(medical_keywords)
        
        # 반복성 점수 (반복이 적을수록 좋음)
        words = response.split()
        unique_words = len(set(words))
        repetition_score = unique_words / len(words) if words else 0
        
        return (length_score + medical_score + repetition_score) / 3
    
    def evaluate(self, num_samples: int = 200) -> Dict[str, Any]:
        """진짜 LLM 챗봇 평가"""
        print(f"\n🔍 진짜 LLM 챗봇 평가 시작 ({num_samples}개 샘플)")
        print("=" * 60)
        print("🔒 데이터 누수 방지: 오직 test 데이터만 사용")
        
        # 테스트 데이터 로딩
        test_data = self.load_test_data(num_samples)
        if not test_data:
            return {}
        
        # 평가 변수
        total_samples = len(test_data)
        total_response_time = 0
        all_responses = []
        all_references = []
        
        print(f"\n🤖 LLM 응답 생성 중: ", end="")
        
        # 각 샘플에 대해 평가
        for i, sample in enumerate(tqdm(test_data, desc="LLM 응답 생성")):
            question = sample.get("question", "")
            reference = sample.get("answer", "")
            
            if not question or not reference:
                continue
            
            # LLM 응답 생성
            start_time = time.time()
            response = self.generate_response(question)
            response_time = time.time() - start_time
            
            # 응답 수집
            all_responses.append(response)
            all_references.append(reference)
            total_response_time += response_time
        
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
        
        # 결과 정리
        results = {
            "evaluation_type": "진짜 LLM 챗봇 평가 (data leakage 방지)",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "data_source": "오직 test 데이터만 사용",
            "total_samples": total_samples,
            "average_response_time": avg_response_time,
            "bertscore_metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            },
            "quality_metrics": {
                "medical_professionalism": avg_medical_professionalism,
                "response_quality": avg_response_quality
            },
            "goals_achieved": {
                "bertscore_f1_0_7": f1_score >= 0.7,
                "response_time_5_seconds": avg_response_time <= 5.0,
                "medical_professionalism_0_6": avg_medical_professionalism >= 0.6
            }
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """결과 출력"""
        print(f"\n🎯 진짜 LLM 챗봇 평가 결과")
        print("=" * 60)
        print(f"🔒 데이터 소스: {results['data_source']}")
        print(f"📊 총 샘플: {results['total_samples']}개")
        print(f"⏱️ 평균 응답 시간: {results['average_response_time']:.2f}초")
        
        print(f"\n🧠 BERTScore 메트릭:")
        print(f"   Precision: {results['bertscore_metrics']['precision']:.4f}")
        print(f"   Recall: {results['bertscore_metrics']['recall']:.4f}")
        print(f"   F1-Score: {results['bertscore_metrics']['f1_score']:.4f}")
        
        print(f"\n📊 품질 메트릭:")
        print(f"   의료 전문성: {results['quality_metrics']['medical_professionalism']:.2%}")
        print(f"   응답 품질: {results['quality_metrics']['response_quality']:.2%}")
        
        print(f"\n🎯 목표 달성 여부:")
        goals = results['goals_achieved']
        print(f"   BERTScore F1 0.7 목표: {'✅ 달성' if goals['bertscore_f1_0_7'] else '❌ 미달성'}")
        print(f"   응답 시간 5초 목표: {'✅ 달성' if goals['response_time_5_seconds'] else '❌ 미달성'}")
        print(f"   의료 전문성 60% 목표: {'✅ 달성' if goals['medical_professionalism_0_6'] else '❌ 미달성'}")

def main():
    """메인 실행 함수"""
    print("🚀 진짜 LLM 챗봇 평가")
    print("=" * 60)
    print("🔒 데이터 누수 방지: 오직 test 데이터만 사용")
    
    # 평가자 초기화
    evaluator = TrueLLMEvaluator()
    
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
        result_file = f"models/true_llm_evaluation_results_{results['timestamp']}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 결과 저장: {result_file}")
        print(f"\n✅ 진짜 LLM 챗봇 평가 완료!")
    else:
        print("❌ 평가 실패")

if __name__ == "__main__":
    main()
