#!/usr/bin/env python3
"""
개선된 하이브리드 의료 챗봇
- 응급/증상 감지 강화
- 올바른 답변 생성 개선
- 질문 유형별 맞춤 응답
"""

import os
import json
import time
import random
import re
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

class ImprovedHybridMedicalChatbot:
    def __init__(self, model_path: str = "models/medical_finetuned_improved"):
        """개선된 하이브리드 의료 챗봇 초기화"""
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 사용 디바이스: {self.device}")
        
        # 모델 로딩
        self._load_model()
        
        # 의료 지식베이스 로딩
        self._load_enhanced_medical_knowledge()
        
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
    
    def _load_enhanced_medical_knowledge(self):
        """강화된 의료 지식베이스 로딩"""
        
        # 1. 응급상황 키워드 (강화)
        self.emergency_keywords = {
            "심각한": ["심한", "심각한", "극심한", "매우 심한", "극도로"],
            "갑작스러운": ["갑자기", "급작스럽게", "순간적으로", "돌연히"],
            "생명위험": ["생명위험", "생명을 위협", "치명적", "위험한", "심각한 상태"],
            "응급증상": [
                "심한 가슴통증", "호흡곤란", "의식불명", "대량출혈", "심한 복통",
                "갑작스러운 두통", "시야장애", "언어장애", "마비", "경련",
                "고열", "호흡정지", "심장정지", "쇼크", "중독"
            ],
            "응급키워드": ["119", "응급실", "구급차", "응급", "긴급", "즉시", "당장"]
        }
        
        # 2. 증상-질병 매핑 (확장)
        self.symptom_disease_map = {
            "감기": {
                "증상": ["콧물", "기침", "목아픔", "발열", "몸살", "재채기", "코막힘", "인후통"],
                "심각도": "경미",
                "응급여부": False
            },
            "독감": {
                "증상": ["고열", "전신근육통", "두통", "피로감", "오한", "기침", "인후통"],
                "심각도": "중등도",
                "응급여부": False
            },
            "당뇨병": {
                "증상": ["다뇨", "다음", "다식", "체중감소", "피로", "시야흐림", "상처치유지연"],
                "심각도": "중등도",
                "응급여부": False
            },
            "고혈압": {
                "증상": ["두통", "어지러움", "가슴답답", "호흡곤란", "코피", "시야장애"],
                "심각도": "중등도",
                "응급여부": False
            },
            "심근경색": {
                "증상": ["심한 가슴통증", "호흡곤란", "식은땀", "메스꺼움", "어지러움", "팔통증"],
                "심각도": "심각",
                "응급여부": True
            },
            "뇌졸중": {
                "증상": ["갑작스러운 두통", "시야장애", "언어장애", "마비", "의식변화", "균형장애"],
                "심각도": "심각",
                "응급여부": True
            },
            "위염": {
                "증상": ["복통", "속쓰림", "메스꺼움", "구토", "소화불량", "식욕부진"],
                "심각도": "경미",
                "응급여부": False
            },
            "우울증": {
                "증상": ["우울감", "무기력", "수면장애", "식욕부진", "집중력저하", "자살생각"],
                "심각도": "중등도",
                "응급여부": False
            }
        }
        
        # 3. 질병-치료 매핑 (상세화)
        self.disease_treatment_map = {
            "감기": {
                "약물": ["아세트아미노펜", "이부프로펜", "진해제", "거담제", "해열제"],
                "자조": ["충분한 휴식", "수분 섭취", "비타민C", "온수 가글", "실내 가습"],
                "의사방문": ["고열(38.5°C 이상)", "3-4일 이상 지속", "호흡곤란", "가슴통증"],
                "응급상황": False
            },
            "독감": {
                "약물": ["타미플루", "리렌자", "아세트아미노펜", "이부프로펜"],
                "자조": ["충분한 휴식", "수분 섭취", "격리", "마스크 착용"],
                "의사방문": ["고열 지속", "호흡곤란", "가슴통증", "의식변화"],
                "응급상황": False
            },
            "당뇨병": {
                "약물": ["의사 처방 약물 (인슐린, 경구약)"],
                "자조": ["규칙적인 식사", "운동", "혈당 측정", "체중 관리", "발 관리"],
                "의사방문": ["혈당 조절 불량", "합병증 증상", "새로운 증상", "상처치유지연"],
                "응급상황": False
            },
            "고혈압": {
                "약물": ["의사 처방 약물 (ACE 억제제, 이뇨제 등)"],
                "자조": ["저염식", "규칙적인 운동", "금연", "금주", "스트레스 관리"],
                "의사방문": ["고혈압 위기", "합병증 증상", "약물 부작용"],
                "응급상황": False
            },
            "심근경색": {
                "약물": ["응급실에서 즉시 치료"],
                "자조": ["즉시 119 신고", "안정된 자세 유지", "니트로글리세린 복용"],
                "의사방문": ["즉시 응급실 방문"],
                "응급상황": True
            },
            "뇌졸중": {
                "약물": ["응급실에서 즉시 치료"],
                "자조": ["즉시 119 신고", "안정된 자세 유지", "구토 시 옆으로 눕히기"],
                "의사방문": ["즉시 응급실 방문"],
                "응급상황": True
            }
        }
        
        # 4. 질문 유형별 응답 템플릿
        self.question_templates = {
            "증상문의": "현재 증상에 대해 문의하신 것 같습니다.",
            "치료문의": "치료 방법에 대해 문의하신 것 같습니다.",
            "응급상황": "응급상황으로 보입니다.",
            "일반상담": "일반적인 의료 상담을 요청하신 것 같습니다.",
            "약물문의": "약물에 대해 문의하신 것 같습니다."
        }
    
    def _detect_emergency_enhanced(self, text: str) -> Tuple[bool, str]:
        """강화된 응급상황 감지"""
        text_lower = text.lower()
        emergency_reasons = []
        
        # 1. 응급 키워드 체크
        for category, keywords in self.emergency_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emergency_reasons.append(f"{category}: {keyword}")
        
        # 2. 응급 질병 체크
        for disease, info in self.symptom_disease_map.items():
            if info["응급여부"] and info["심각도"] == "심각":
                for symptom in info["증상"]:
                    if symptom in text_lower:
                        emergency_reasons.append(f"응급질병: {disease} ({symptom})")
        
        # 3. 응급 패턴 체크
        emergency_patterns = [
            r"심하게\s+아프", r"갑자기\s+아프", r"생명\s*위험", r"즉시\s*병원",
            r"119\s*신고", r"응급실\s*방문", r"구급차\s*호출"
        ]
        
        for pattern in emergency_patterns:
            if re.search(pattern, text_lower):
                emergency_reasons.append(f"응급패턴: {pattern}")
        
        is_emergency = len(emergency_reasons) > 0
        reason = "; ".join(emergency_reasons) if emergency_reasons else ""
        
        return is_emergency, reason
    
    def _extract_symptoms_enhanced(self, text: str) -> List[Dict[str, Any]]:
        """강화된 증상 추출"""
        text_lower = text.lower()
        detected_symptoms = []
        
        # 1. 질병명 직접 언급 체크
        for disease, info in self.symptom_disease_map.items():
            if disease in text_lower:
                detected_symptoms.append({
                    "disease": disease,
                    "symptoms": [disease],
                    "severity": info["심각도"],
                    "is_emergency": info["응급여부"]
                })
                continue
        
        # 2. 증상 키워드 매칭
        for disease, info in self.symptom_disease_map.items():
            matched_symptoms = []
            for symptom in info["증상"]:
                if symptom in text_lower:
                    matched_symptoms.append(symptom)
            
            if matched_symptoms:
                detected_symptoms.append({
                    "disease": disease,
                    "symptoms": matched_symptoms,
                    "severity": info["심각도"],
                    "is_emergency": info["응급여부"]
                })
        
        # 3. 일반적인 증상 표현 매칭
        general_symptoms = {
            "두통": ["머리", "두통", "머리가 아프", "두통이"],
            "복통": ["배", "복부", "위", "속", "배가 아프", "위가 아프", "속이"],
            "가슴통증": ["가슴", "흉부", "가슴이 아프"],
            "호흡곤란": ["숨", "호흡", "숨이 차", "호흡이"],
            "발열": ["열", "발열", "고열", "체온"],
            "피로": ["피로", "무기력", "힘들", "지치"]
        }
        
        for symptom_type, expressions in general_symptoms.items():
            for expression in expressions:
                if expression in text_lower:
                    # 가장 관련성 높은 질병 찾기
                    best_disease = None
                    for disease, info in self.symptom_disease_map.items():
                        if symptom_type in info["증상"] or any(s in info["증상"] for s in expressions):
                            best_disease = disease
                            break
                    
                    if best_disease and not any(s["disease"] == best_disease for s in detected_symptoms):
                        detected_symptoms.append({
                            "disease": best_disease,
                            "symptoms": [symptom_type],
                            "severity": self.symptom_disease_map[best_disease]["심각도"],
                            "is_emergency": self.symptom_disease_map[best_disease]["응급여부"]
                        })
        
        return detected_symptoms
    
    def _classify_question_type(self, text: str) -> str:
        """질문 유형 분류"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["증상", "아프", "아픔", "불편"]):
            return "증상문의"
        elif any(word in text_lower for word in ["치료", "약물", "복용", "처방"]):
            return "치료문의"
        elif any(word in text_lower for word in ["약", "약물", "복용", "처방전"]):
            return "약물문의"
        elif any(word in text_lower for word in ["상담", "문의", "질문", "도움"]):
            return "일반상담"
        else:
            return "일반상담"
    
    def _generate_emergency_response(self, text: str, emergency_reason: str) -> str:
        """응급상황 응답 생성"""
        response = "🚨 응급상황으로 보입니다!\n\n"
        response += "즉시 다음 조치를 취하세요:\n"
        response += "1. 119에 신고하세요\n"
        response += "2. 응급실을 방문하세요\n"
        response += "3. 안정된 자세를 유지하세요\n\n"
        response += f"감지된 응급상황: {emergency_reason}\n\n"
        response += "⚠️ 이 챗봇은 응급상황에 대한 정확한 진단을 제공할 수 없습니다. 반드시 의료진과 상담하세요."
        
        return response
    
    def _generate_disease_response(self, detected_symptoms: List[Dict[str, Any]]) -> str:
        """질병별 맞춤 응답 생성"""
        if not detected_symptoms:
            return "구체적인 증상을 알려주시면 더 정확한 상담을 도와드릴 수 있습니다."
        
        # 가장 심각한 질병 우선 처리
        emergency_diseases = [s for s in detected_symptoms if s["is_emergency"]]
        if emergency_diseases:
            disease_info = emergency_diseases[0]
            return self._generate_emergency_response("", f"응급질병: {disease_info['disease']}")
        
        # 일반 질병 처리
        disease_info = detected_symptoms[0]
        disease = disease_info["disease"]
        
        if disease in self.disease_treatment_map:
            treatment = self.disease_treatment_map[disease]
            response = f"🔍 {disease} 증상으로 보입니다.\n\n"
            
            # 자조 방법
            if "자조" in treatment:
                response += "📋 자조 방법:\n"
                for method in treatment["자조"]:
                    response += f"• {method}\n"
                response += "\n"
            
            # 약물 정보
            if "약물" in treatment:
                response += "💊 약물 정보:\n"
                for medicine in treatment["약물"]:
                    response += f"• {medicine}\n"
                response += "\n"
            
            # 의사 방문 시점
            if "의사방문" in treatment:
                response += "🏥 의사 방문이 필요한 경우:\n"
                for condition in treatment["의사방문"]:
                    response += f"• {condition}\n"
                response += "\n"
            
            response += "⚠️ 정확한 진단과 치료를 위해 의료진과 상담하세요."
            
            return response
        
        return "증상이 지속되거나 악화되면 의료진과 상담하세요."
    
    def _get_model_response_enhanced(self, question: str) -> str:
        """개선된 모델 기반 응답 생성 (빠른 버전)"""
        try:
            # 간단한 프롬프트로 속도 향상
            prompt = f"의료 상담: {question}\n의료진:"
            
            # 토크나이징
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # 빠른 생성 설정
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 50,  # 길이 단축
                    num_return_sequences=1,
                    temperature=0.8,  # 약간 높여서 다양성 증가
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs),
                    repetition_penalty=1.1,  # 반복 억제 강화
                    max_time=5.0  # 최대 5초 제한
                )
            
            # 디코딩
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            
            # 응답 정리
            if "의료진:" in response:
                response = response.split("의료진:")[-1].strip()
            
            # 반복 제거 및 품질 검증
            response = self._clean_model_response(response)
            
            # 응답이 너무 짧거나 유효하지 않으면 폴백 사용
            if len(response) < 10 or not self._is_valid_medical_response(response):
                return "구체적인 증상이나 질문을 알려주시면 더 정확한 상담을 도와드릴 수 있습니다."
            
            return response
                
        except Exception as e:
            print(f"⚠️ 모델 응답 생성 오류: {e}")
            return "구체적인 증상이나 질문을 알려주시면 더 정확한 상담을 도와드릴 수 있습니다."
    
    def _clean_model_response(self, response: str) -> str:
        """모델 응답 정리 및 품질 검증"""
        # 반복 제거
        lines = response.split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_lines:
                # 유효하지 않은 패턴 제거
                if not any(pattern in line for pattern in [
                    "의료진:", "간호사:", "진료:", "내원한 환자에게",
                    "가장 적절한 치료법은", "1 스테로이드", "2 모기약"
                ]):
                    cleaned_lines.append(line)
                    seen_lines.add(line)
        
        if cleaned_lines:
            return cleaned_lines[0]
        else:
            return "정확한 상담을 위해 의료진과 직접 상담하시기 바랍니다."
    
    def get_hybrid_response_enhanced(self, question: str) -> Tuple[str, str, bool, List[Dict], float]:
        """개선된 하이브리드 응답 생성"""
        start_time = time.time()
        
        # 1. 응급상황 우선 체크 (강화)
        is_emergency, emergency_reason = self._detect_emergency_enhanced(question)
        if is_emergency:
            response = self._generate_emergency_response(question, emergency_reason)
            response_time = time.time() - start_time
            return response, "emergency", is_emergency, [], response_time
        
        # 2. 증상 추출 및 질병별 응답 (강화)
        detected_symptoms = self._extract_symptoms_enhanced(question)
        if detected_symptoms:
            response = self._generate_disease_response(detected_symptoms)
            response_time = time.time() - start_time
            return response, "rule_based", is_emergency, detected_symptoms, response_time
        
        # 3. 모델 기반 응답 시도 (개선)
        model_response = self._get_model_response_enhanced(question)
        if self._is_valid_medical_response(model_response):
            response_time = time.time() - start_time
            return model_response, "model_based", is_emergency, [], response_time
        
        # 4. 폴백 응답
        fallback_response = "구체적인 증상이나 질문을 알려주시면 더 정확한 상담을 도와드릴 수 있습니다. 정확한 진단을 위해 의료진과 상담하시기 바랍니다."
        response_time = time.time() - start_time
        return fallback_response, "fallback", is_emergency, [], response_time
    
    def _is_valid_medical_response(self, response: str) -> bool:
        """의료 응답 유효성 검사"""
        if not response or len(response) < 10:
            return False
        
        invalid_patterns = [
            "내원한 환자에게", "가장 적절한 치료법은", "1 스테로이드", 
            "2 모기약", "3 피부트레이닝", "4 피부트레이닝"
        ]
        
        for pattern in invalid_patterns:
            if pattern in response:
                return False
        
        # 의료 관련 키워드가 있는지 확인
        medical_keywords = ["의료", "병원", "증상", "치료", "약물", "의사", "상담"]
        return any(keyword in response for keyword in medical_keywords)

def main():
    """메인 실행 함수"""
    print("🚀 개선된 하이브리드 의료 챗봇 테스트")
    print("=" * 50)
    
    # 챗봇 초기화
    chatbot = ImprovedHybridMedicalChatbot()
    
    # 테스트 케이스
    test_cases = [
        "감기 걸린 것 같아요. 콧물이 나고 목이 아파요.",
        "가슴이 심하게 아프고 숨이 차요.",
        "머리가 아픈데 어떻게 해야 할까요?",
        "당뇨병이 있는데 혈당 관리가 어려워요.",
        "갑자기 머리가 심하게 아프고 구토가 나요.",
        "심한 가슴통증과 함께 식은땀이 나요.",
        "갑작스러운 두통과 시야장애가 있어요.",
        "위가 아프고 속이 쓰려요."
    ]
    
    print(f"\n🧪 개선된 하이브리드 챗봇 테스트:")
    
    for i, question in enumerate(test_cases, 1):
        print(f"\n--- 테스트 {i} ---")
        print(f"입력: {question}")
        
        response, source, is_emergency, symptoms, response_time = chatbot.get_hybrid_response_enhanced(question)
        
        print(f"응답: {response}")
        print(f"응급상황: {is_emergency}")
        print(f"감지된 증상: {[s['disease'] for s in symptoms]}")
        print(f"응답 시간: {response_time:.2f}초")
        print(f"응답 소스: {source}")
    
    print(f"\n✅ 개선된 하이브리드 챗봇 테스트 완료!")

if __name__ == "__main__":
    main()
