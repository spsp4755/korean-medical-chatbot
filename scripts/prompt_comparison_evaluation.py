import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time
from datetime import datetime
from tqdm import tqdm

class PromptComparisonEvaluator:
    def __init__(self, base_model_path="42dot/42dot_LLM-SFT-1.3B", 
                 peft_model_path="models/medical_finetuned_peft"):
        self.base_model_path = base_model_path
        self.peft_model_path = peft_model_path
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 사용 디바이스: {self.device}")
        self._load_model()
        
    def _load_model(self):
        """PEFT 모델 로드"""
        print("PEFT 모델 로딩 중...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        self.model = PeftModel.from_pretrained(base_model, self.peft_model_path)
        self.model.eval()
        
        print("PEFT 모델 로딩 완료!")
    
    def create_basic_prompt(self, question):
        """기본 프롬프트 (Before)"""
        return f"질문: {question}\n답변:"
    
    def create_improved_prompt(self, question):
        """개선된 프롬프트 (After)"""
        return f"""당신은 친근하고 실용적인 의료 상담 챗봇입니다. 환자의 질문에 대해 다음과 같이 답변해주세요:

1. 먼저 응급상황인지 판단하세요
2. 일반인이 이해하기 쉬운 언어로 설명하세요
3. 즉시 실행 가능한 조치를 제시하세요
4. 답변은 200자 이내로 간결하게 하세요
5. 필요시 의사 상담을 권하세요

환자 질문: {question}

의료진 답변:"""
    
    def generate_response(self, question, prompt_type="basic", max_length=200):
        """질문에 대한 응답 생성"""
        start_time = time.time()
        
        if prompt_type == "basic":
            prompt = self.create_basic_prompt(question)
        else:
            prompt = self.create_improved_prompt(question)
        
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
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                top_p=0.9,
                top_k=50
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        end_time = time.time()
        response_time = end_time - start_time
        
        return response, response_time
    
    def evaluate_response_quality(self, response):
        """응답 품질 평가"""
        # 응답 길이
        length = len(response)
        
        # 응급상황 키워드 체크
        emergency_keywords = ["응급", "즉시", "병원", "응급실", "119", "심각", "위험"]
        emergency_score = sum(1 for keyword in emergency_keywords if keyword in response)
        
        # 실용적 조치 키워드 체크
        practical_keywords = ["하세요", "드세요", "마시세요", "휴식", "수분", "의사", "상담"]
        practical_score = sum(1 for keyword in practical_keywords if keyword in response)
        
        # 일반인 친화적 언어 체크
        friendly_keywords = ["도움", "괜찮", "걱정", "안심", "확인", "체크"]
        friendly_score = sum(1 for keyword in friendly_keywords if keyword in response)
        
        # 전문 용어 체크 (너무 많으면 감점)
        medical_terms = ["항생제", "감수성", "배양", "전해질", "불균형", "합병증"]
        medical_term_count = sum(1 for term in medical_terms if term in response)
        
        return {
            "length": length,
            "emergency_score": emergency_score,
            "practical_score": practical_score,
            "friendly_score": friendly_score,
            "medical_term_count": medical_term_count,
            "is_appropriate_length": 50 <= length <= 300,
            "has_emergency_guidance": emergency_score > 0,
            "has_practical_advice": practical_score > 0,
            "is_user_friendly": friendly_score > 0 and medical_term_count <= 2
        }
    
    def compare_prompts(self, test_questions):
        """프롬프트 비교 평가"""
        print("🔍 프롬프트 비교 평가 시작...")
        
        results = {
            "basic_prompt": [],
            "improved_prompt": [],
            "comparison": {}
        }
        
        for question in tqdm(test_questions, desc="질문 평가 중"):
            # 기본 프롬프트로 평가
            basic_response, basic_time = self.generate_response(question, "basic")
            basic_quality = self.evaluate_response_quality(basic_response)
            
            # 개선된 프롬프트로 평가
            improved_response, improved_time = self.generate_response(question, "improved")
            improved_quality = self.evaluate_response_quality(improved_response)
            
            results["basic_prompt"].append({
                "question": question,
                "response": basic_response,
                "response_time": basic_time,
                "quality_metrics": basic_quality
            })
            
            results["improved_prompt"].append({
                "question": question,
                "response": improved_response,
                "response_time": improved_time,
                "quality_metrics": improved_quality
            })
        
        # 비교 분석
        basic_metrics = [item["quality_metrics"] for item in results["basic_prompt"]]
        improved_metrics = [item["quality_metrics"] for item in results["improved_prompt"]]
        
        results["comparison"] = {
            "avg_response_length": {
                "basic": sum(m["length"] for m in basic_metrics) / len(basic_metrics),
                "improved": sum(m["length"] for m in improved_metrics) / len(improved_metrics)
            },
            "avg_response_time": {
                "basic": sum(item["response_time"] for item in results["basic_prompt"]) / len(results["basic_prompt"]),
                "improved": sum(item["response_time"] for item in results["improved_prompt"]) / len(results["improved_prompt"])
            },
            "emergency_guidance_ratio": {
                "basic": sum(1 for m in basic_metrics if m["has_emergency_guidance"]) / len(basic_metrics),
                "improved": sum(1 for m in improved_metrics if m["has_emergency_guidance"]) / len(improved_metrics)
            },
            "practical_advice_ratio": {
                "basic": sum(1 for m in basic_metrics if m["has_practical_advice"]) / len(basic_metrics),
                "improved": sum(1 for m in improved_metrics if m["has_practical_advice"]) / len(improved_metrics)
            },
            "user_friendly_ratio": {
                "basic": sum(1 for m in basic_metrics if m["is_user_friendly"]) / len(basic_metrics),
                "improved": sum(1 for m in improved_metrics if m["is_user_friendly"]) / len(improved_metrics)
            },
            "appropriate_length_ratio": {
                "basic": sum(1 for m in basic_metrics if m["is_appropriate_length"]) / len(basic_metrics),
                "improved": sum(1 for m in improved_metrics if m["is_appropriate_length"]) / len(improved_metrics)
            }
        }
        
        return results

def main():
    # 테스트 질문들
    test_questions = [
        "장염에 걸린거 같은데 어떻게 해야하나요?",
        "머리가 아픈데 진통제 먹어도 될까요?",
        "열이 나는데 병원 가야 할까요?",
        "복통이 심한데 응급실 가야 하나요?",
        "감기 증상이 있는데 약국에서 약을 사도 될까요?",
        "어지러운데 뭘 해야 할까요?",
        "가슴이 답답한데 심장 문제일까요?",
        "배가 아픈데 식중독일까요?"
    ]
    
    print("🚀 프롬프트 비교 평가 시작")
    print("=" * 60)
    
    evaluator = PromptComparisonEvaluator()
    results = evaluator.compare_prompts(test_questions)
    
    # 결과 출력
    print("\n📊 프롬프트 비교 결과")
    print("=" * 60)
    
    comp = results["comparison"]
    print(f"평균 응답 길이:")
    print(f"  기본 프롬프트: {comp['avg_response_length']['basic']:.1f}자")
    print(f"  개선 프롬프트: {comp['avg_response_length']['improved']:.1f}자")
    print(f"  개선도: {((comp['avg_response_length']['improved'] - comp['avg_response_length']['basic']) / comp['avg_response_length']['basic'] * 100):+.1f}%")
    
    print(f"\n응급상황 안내 비율:")
    print(f"  기본 프롬프트: {comp['emergency_guidance_ratio']['basic']:.1%}")
    print(f"  개선 프롬프트: {comp['emergency_guidance_ratio']['improved']:.1%}")
    print(f"  개선도: {((comp['emergency_guidance_ratio']['improved'] - comp['emergency_guidance_ratio']['basic']) * 100):+.1f}%p")
    
    print(f"\n실용적 조치 제시 비율:")
    print(f"  기본 프롬프트: {comp['practical_advice_ratio']['basic']:.1%}")
    print(f"  개선 프롬프트: {comp['practical_advice_ratio']['improved']:.1%}")
    print(f"  개선도: {((comp['practical_advice_ratio']['improved'] - comp['practical_advice_ratio']['basic']) * 100):+.1f}%p")
    
    print(f"\n사용자 친화적 답변 비율:")
    print(f"  기본 프롬프트: {comp['user_friendly_ratio']['basic']:.1%}")
    print(f"  개선 프롬프트: {comp['user_friendly_ratio']['improved']:.1%}")
    print(f"  개선도: {((comp['user_friendly_ratio']['improved'] - comp['user_friendly_ratio']['basic']) * 100):+.1f}%p")
    
    print(f"\n적절한 길이 답변 비율:")
    print(f"  기본 프롬프트: {comp['appropriate_length_ratio']['basic']:.1%}")
    print(f"  개선 프롬프트: {comp['appropriate_length_ratio']['improved']:.1%}")
    print(f"  개선도: {((comp['appropriate_length_ratio']['improved'] - comp['appropriate_length_ratio']['basic']) * 100):+.1f}%p")
    
    # 결과 저장
    os.makedirs("results", exist_ok=True)
    with open("results/prompt_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 결과가 results/prompt_comparison_results.json에 저장되었습니다!")

if __name__ == "__main__":
    main()
