import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from bert_score import score as bert_score
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 환경 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class OptimizedPromptEvaluator:
    def __init__(self, base_model_path="42dot/42dot_LLM-SFT-1.3B", 
                 peft_model_path="models/medical_finetuned_peft"):
        self.base_model_path = base_model_path
        self.peft_model_path = peft_model_path
        
        print("모델 로딩 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Base 모델 로드
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # PEFT 모델 로드
        self.model = PeftModel.from_pretrained(
            self.base_model,
            peft_model_path,
            torch_dtype=torch.float32
        )
        
        self.model.eval()
        print("✅ 모델 로딩 완료!")
    
    def create_optimized_prompt(self, question):
        """최적화된 프롬프트 생성 (Basic + 최소한의 개선)"""
        return f"""의료 상담을 도와드리겠습니다. 정확하고 안전한 정보를 제공하겠습니다.

질문: {question}

답변:"""
    
    def generate_response(self, question, max_length=256):
        """질문에 대한 응답 생성 (Basic과 동일한 파라미터)"""
        prompt = self.create_optimized_prompt(question)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs.input_ids.shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.3,  # Basic과 동일
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                top_p=0.9,  # Basic과 동일
                top_k=50    # Basic과 동일
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        return response
    
    def evaluate_bertscore(self, test_data, sample_size=500):
        """BERTScore로 성능 평가 (Basic과 동일한 방식)"""
        # 샘플 데이터 선택 (메모리 절약)
        if len(test_data) > sample_size:
            import random
            test_data = random.sample(test_data, sample_size)
        
        print(f"BERTScore 평가 시작 (샘플 데이터: {len(test_data)}개)")
        
        questions = [item["question"] for item in test_data]
        ground_truths = [item["answer"] for item in test_data]
        predictions = []
        
        print("응답 생성 중...")
        for i, question in enumerate(tqdm(questions)):
            try:
                response = self.generate_response(question)
                predictions.append(response)
            except Exception as e:
                print(f"오류 발생 (질문 {i+1}): {e}")
                predictions.append("응답 생성 실패")
        
        print("BERTScore 계산 중...")
        P, R, F1 = bert_score(predictions, ground_truths, lang="ko")
        
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
            "sample_size": len(test_data)
        }
    
    def evaluate_medical_metrics(self, test_data, sample_size=500):
        """의료 특화 메트릭 평가 (Basic과 동일한 방식)"""
        # 샘플 데이터 선택 (메모리 절약)
        if len(test_data) > sample_size:
            import random
            test_data = random.sample(test_data, sample_size)
        
        print(f"의료 메트릭 평가 시작 (샘플 데이터: {len(test_data)}개)")
        
        medical_terms = [
            "증상", "치료", "약물", "진단", "병원", "의사", "환자", "질병", "감염", "염증",
            "통증", "발열", "기침", "두통", "복통", "피로", "구토", "설사", "변비", "수면",
            "혈압", "혈당", "콜레스테롤", "심장", "폐", "간", "신장", "뇌", "신경", "근육"
        ]
        
        total_score = 0
        medical_coverage = 0
        safety_score = 0
        response_lengths = []
        
        for item in tqdm(test_data):
            question = item["question"]
            ground_truth = item["answer"]
            
            try:
                response = self.generate_response(question)
                response_lengths.append(len(response))
                
                # 의료 용어 포함도 체크
                medical_terms_in_response = sum(1 for term in medical_terms 
                                             if term in response)
                medical_coverage += medical_terms_in_response / len(medical_terms)
                
                # 안전성 체크 (위험한 표현 제거)
                dangerous_phrases = ["확실히", "100%", "절대", "무조건", "진단", "치료법"]
                if not any(phrase in response for phrase in dangerous_phrases):
                    safety_score += 1
                
                # 응답 길이 적절성 (의료 챗봇에 최적화)
                if 20 <= len(response) <= 800:
                    total_score += 1
                    
            except Exception as e:
                print(f"오류 발생: {e}")
        
        return {
            "medical_coverage": medical_coverage / len(test_data),
            "safety_score": safety_score / len(test_data),
            "response_quality": total_score / len(test_data),
            "avg_response_length": np.mean(response_lengths),
            "sample_size": len(test_data)
        }

def main():
    # 테스트 데이터 로드 (Basic과 동일한 방식)
    print("테스트 데이터 로딩 중...")
    
    test_files = [
        "data/processed/splits/essential_medical_test.json",
        "data/processed/splits/professional_medical_test.json"
    ]
    
    test_data = []
    for file_path in test_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                test_data.extend(data)
                print(f"로드됨: {file_path} ({len(data)}개)")
        else:
            print(f"파일 없음: {file_path}")
    
    if not test_data:
        print("테스트 데이터를 찾을 수 없습니다!")
        return
    
    print(f"총 테스트 데이터: {len(test_data)}개")
    
    # Optimized 프롬프트 모델 평가
    evaluator = OptimizedPromptEvaluator()
    
    # BERTScore 평가
    print("\n=== BERTScore 평가 ===")
    bertscore_results = evaluator.evaluate_bertscore(test_data)
    
    # 의료 메트릭 평가
    print("\n=== 의료 메트릭 평가 ===")
    medical_results = evaluator.evaluate_medical_metrics(test_data)
    
    # 결과 출력
    print("\n=== Optimized 프롬프트 성능 결과 ===")
    print(f"BERTScore Precision: {bertscore_results['bertscore_precision']:.4f}")
    print(f"BERTScore Recall: {bertscore_results['bertscore_recall']:.4f}")
    print(f"BERTScore F1: {bertscore_results['bertscore_f1']:.4f}")
    print(f"의료 용어 포함도: {medical_results['medical_coverage']:.4f}")
    print(f"안전성 점수: {medical_results['safety_score']:.4f}")
    print(f"응답 품질: {medical_results['response_quality']:.4f}")
    print(f"평균 응답 길이: {medical_results['avg_response_length']:.1f}자")
    print(f"평가 샘플 수: {bertscore_results['sample_size']}개")
    
    # 결과 저장
    results = {
        "model_type": "PEFT_42dot_LLM_Optimized_Prompt",
        "evaluation_strategy": "전체_test_데이터_평가",
        "temperature": 0.3,
        "bertscore": bertscore_results,
        "medical_metrics": medical_results,
        "evaluation_timestamp": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    }
    
    with open("results/optimized_prompt_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n결과가 results/optimized_prompt_evaluation.json에 저장되었습니다!")
    
    return results

if __name__ == "__main__":
    main()

