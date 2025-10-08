#!/usr/bin/env python3
"""
고급 의료 모델 평가 스크립트
- BERTScore 기반 정량적 평가
- 의료 전문성 평가
- 응급상황 감지 평가
"""

import os
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class AdvancedMedicalEvaluator:
    def __init__(self, model_path="models/medical_finetuned_advanced"):
        """고급 의료 평가자 초기화"""
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 사용 디바이스: {self.device}")
        
        # 모델 로딩
        self._load_model()
        
    def _load_model(self):
        """모델과 토크나이저 로딩"""
        print(f"\n📥 모델 로딩 중: {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print("✅ 모델 로드 완료")
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            raise
    
    def load_test_data(self, sample_size=500):
        """테스트 데이터 로딩 (Data Leakage 방지)"""
        print(f"\n📊 테스트 데이터 로딩 중...")
        
        test_paths = [
            "data/processed/splits/essential_medical_test.json",
            "data/processed/splits/professional_medical_test.json"
        ]
        
        all_data = []
        for path in test_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.extend(data)
                    print(f"✅ {path}: {len(data)}개 샘플")
            else:
                print(f"⚠️ {path}: 파일 없음")
        
        # 샘플링
        if len(all_data) > sample_size:
            import random
            all_data = random.sample(all_data, sample_size)
            print(f"📊 샘플링: {sample_size}개")
        
        print(f"📊 총 테스트 데이터: {len(all_data)}개")
        return all_data
    
    def generate_response(self, question, max_length=200):
        """응답 생성"""
        prompt = f"환자: {question}\n의료진:"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        return response
    
    def evaluate_medical_professionalism(self, question, response):
        """의료 전문성 평가"""
        medical_keywords = [
            "의사", "병원", "진료", "처방", "약물", "치료", "증상", "진단",
            "응급", "응급실", "119", "구급차", "생명", "위험", "즉시",
            "혈압", "혈당", "체온", "맥박", "호흡", "의식", "통증"
        ]
        
        score = 0
        for keyword in medical_keywords:
            if keyword in response:
                score += 1
        
        return min(score / len(medical_keywords) * 100, 100)
    
    def evaluate_emergency_detection(self, question, response):
        """응급상황 감지 평가"""
        emergency_keywords = [
            "응급", "119", "구급차", "응급실", "생명", "위험", "즉시",
            "심장", "뇌졸중", "출혈", "의식", "호흡", "가슴", "복부"
        ]
        
        question_lower = question.lower()
        response_lower = response.lower()
        
        # 질문에 응급 키워드가 있는지 확인
        has_emergency = any(keyword in question_lower for keyword in emergency_keywords)
        
        if has_emergency:
            # 응답에 응급 대응이 있는지 확인
            has_emergency_response = any(keyword in response_lower for keyword in emergency_keywords)
            return 100 if has_emergency_response else 0
        else:
            # 응급이 아닌 경우 정상 응답인지 확인
            return 100 if "응급" not in response_lower else 0
    
    def evaluate_symptom_detection(self, question, response):
        """증상 감지 평가"""
        symptom_keywords = [
            "두통", "복통", "발열", "기침", "어지러움", "구토", "설사",
            "가슴", "복부", "머리", "목", "등", "팔", "다리", "통증"
        ]
        
        question_lower = question.lower()
        response_lower = response.lower()
        
        detected_symptoms = []
        for symptom in symptom_keywords:
            if symptom in question_lower and symptom in response_lower:
                detected_symptoms.append(symptom)
        
        return len(detected_symptoms) / len(symptom_keywords) * 100
    
    def evaluate_response_quality(self, response):
        """응답 품질 평가"""
        if not response or len(response.strip()) < 10:
            return 0
        
        # 길이 점수 (너무 짧거나 너무 길면 감점)
        length_score = min(len(response) / 100, 1.0) * 50
        
        # 완성도 점수 (문장이 완전한지)
        completeness_score = 50 if response.endswith(('.', '!', '?')) else 30
        
        return length_score + completeness_score
    
    def evaluate_with_bertscore(self, questions, responses, references):
        """BERTScore 기반 평가"""
        print("\n🔍 BERTScore 평가 중...")
        
        # BERTScore 계산
        P, R, F1 = score(responses, references, lang="ko", verbose=True)
        
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item()
        }
    
    def evaluate(self, sample_size=500):
        """종합 평가 실행"""
        print(f"\n🚀 고급 의료 모델 평가 시작")
        print("=" * 50)
        
        # 테스트 데이터 로딩
        test_data = self.load_test_data(sample_size)
        
        # 평가 결과 저장
        results = {
            "total_samples": len(test_data),
            "evaluations": [],
            "scores": {
                "medical_professionalism": [],
                "emergency_detection": [],
                "symptom_detection": [],
                "response_quality": [],
                "response_times": []
            }
        }
        
        print(f"\n📊 평가 진행 중...")
        
        for i, item in enumerate(tqdm(test_data, desc="평가 진행")):
            question = item.get('question', '')
            reference = item.get('answer', '')
            
            # 응답 생성
            start_time = time.time()
            response = self.generate_response(question)
            response_time = time.time() - start_time
            
            # 각 항목별 평가
            medical_score = self.evaluate_medical_professionalism(question, response)
            emergency_score = self.evaluate_emergency_detection(question, response)
            symptom_score = self.evaluate_symptom_detection(question, response)
            quality_score = self.evaluate_response_quality(response)
            
            # 결과 저장
            eval_result = {
                "question": question,
                "response": response,
                "reference": reference,
                "medical_professionalism": medical_score,
                "emergency_detection": emergency_score,
                "symptom_detection": symptom_score,
                "response_quality": quality_score,
                "response_time": response_time
            }
            
            results["evaluations"].append(eval_result)
            results["scores"]["medical_professionalism"].append(medical_score)
            results["scores"]["emergency_detection"].append(emergency_score)
            results["scores"]["symptom_detection"].append(symptom_score)
            results["scores"]["response_quality"].append(quality_score)
            results["scores"]["response_times"].append(response_time)
        
        # BERTScore 평가
        questions = [item.get('question', '') for item in test_data]
        responses = [eval_result["response"] for eval_result in results["evaluations"]]
        references = [item.get('answer', '') for item in test_data]
        
        bertscore_results = self.evaluate_with_bertscore(questions, responses, references)
        results["bertscore"] = bertscore_results
        
        # 통계 계산
        results["statistics"] = {
            "medical_professionalism": {
                "mean": sum(results["scores"]["medical_professionalism"]) / len(results["scores"]["medical_professionalism"]),
                "max": max(results["scores"]["medical_professionalism"]),
                "min": min(results["scores"]["medical_professionalism"])
            },
            "emergency_detection": {
                "mean": sum(results["scores"]["emergency_detection"]) / len(results["scores"]["emergency_detection"]),
                "max": max(results["scores"]["emergency_detection"]),
                "min": min(results["scores"]["emergency_detection"])
            },
            "symptom_detection": {
                "mean": sum(results["scores"]["symptom_detection"]) / len(results["scores"]["symptom_detection"]),
                "max": max(results["scores"]["symptom_detection"]),
                "min": min(results["scores"]["symptom_detection"])
            },
            "response_quality": {
                "mean": sum(results["scores"]["response_quality"]) / len(results["scores"]["response_quality"]),
                "max": max(results["scores"]["response_quality"]),
                "min": min(results["scores"]["response_quality"])
            },
            "response_time": {
                "mean": sum(results["scores"]["response_times"]) / len(results["scores"]["response_times"]),
                "max": max(results["scores"]["response_times"]),
                "min": min(results["scores"]["response_times"])
            }
        }
        
        # 전체 정확도 계산
        overall_accuracy = (
            results["statistics"]["medical_professionalism"]["mean"] +
            results["statistics"]["emergency_detection"]["mean"] +
            results["statistics"]["symptom_detection"]["mean"] +
            results["statistics"]["response_quality"]["mean"]
        ) / 4
        
        results["overall_accuracy"] = overall_accuracy
        
        # 결과 출력
        print(f"\n📊 고급 의료 모델 평가 결과")
        print("=" * 50)
        print(f"📈 전체 정확도: {overall_accuracy:.2f}%")
        print(f"🏥 의료 전문성: {results['statistics']['medical_professionalism']['mean']:.2f}%")
        print(f"🚨 응급 감지: {results['statistics']['emergency_detection']['mean']:.2f}%")
        print(f"🔍 증상 감지: {results['statistics']['symptom_detection']['mean']:.2f}%")
        print(f"💬 응답 품질: {results['statistics']['response_quality']['mean']:.2f}%")
        print(f"⏱️ 평균 응답 시간: {results['statistics']['response_time']['mean']:.2f}초")
        
        print(f"\n🔍 BERTScore 결과")
        print(f"📊 Precision: {bertscore_results['bertscore_precision']:.4f}")
        print(f"📊 Recall: {bertscore_results['bertscore_recall']:.4f}")
        print(f"📊 F1-Score: {bertscore_results['bertscore_f1']:.4f}")
        
        # 결과 저장
        output_path = "evaluation_results_advanced.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 평가 결과 저장: {output_path}")
        
        return results

def main():
    """메인 실행 함수"""
    print("🚀 고급 의료 모델 평가 시작")
    print("=" * 50)
    
    # 평가자 초기화
    evaluator = AdvancedMedicalEvaluator()
    
    # 평가 실행
    results = evaluator.evaluate(sample_size=500)
    
    print("\n🎉 고급 의료 모델 평가 완료!")

if __name__ == "__main__":
    main()
