#!/usr/bin/env python3
"""
의료 시스템 구축 스크립트
- 의료 용어 사전 구축
- 상담 시나리오 생성
- 응급상황 대응 시스템
"""

import os
import json
from typing import Dict, List, Any

class MedicalSystemBuilder:
    def __init__(self):
        """의료 시스템 구축자 초기화"""
        self.medical_terms = {}
        self.consultation_scenarios = []
        self.emergency_protocols = {}
        
    def build_medical_terms_dictionary(self):
        """의료 용어 사전 구축"""
        print("📚 의료 용어 사전 구축 중...")
        
        self.medical_terms = {
            "증상": {
                "두통": ["편두통", "긴장성 두통", "군발성 두통", "만성 두통"],
                "복통": ["위통", "장통", "복부 통증", "배 아픔"],
                "발열": ["열", "고열", "저열", "체온 상승"],
                "기침": ["마른기침", "가래기침", "지속성 기침", "밤기침"],
                "어지러움": ["현기증", "어지러움", "실신", "의식 잃음"],
                "구토": ["토함", "구역질", "메스꺼움", "역류"],
                "설사": ["변비", "설사", "변색", "변의 이상"]
            },
            "질병": {
                "감기": ["상기도 감염", "코감기", "인후통", "기침"],
                "독감": ["인플루엔자", "고열", "전신 통증", "피로감"],
                "고혈압": ["혈압 상승", "두통", "어지러움", "가슴 통증"],
                "당뇨병": ["혈당 상승", "다뇨", "다음", "체중 감소"],
                "심장병": ["가슴 통증", "호흡 곤란", "부종", "피로감"],
                "뇌졸중": ["마비", "언어 장애", "의식 장애", "두통"]
            },
            "치료": {
                "약물": ["처방약", "경구약", "주사약", "연고"],
                "수술": ["수술", "내시경", "복강경", "개복술"],
                "물리치료": ["재활", "운동치료", "마사지", "전기치료"],
                "식이요법": ["식단 조절", "금식", "제한식", "영양 관리"]
            },
            "응급": {
                "119": ["구급차", "응급실", "응급처치", "생명 구조"],
                "심폐소생술": ["CPR", "인공호흡", "심장마사지", "응급처치"],
                "출혈": ["지혈", "압박", "응급처치", "병원 이송"],
                "의식불명": ["의식 확인", "호흡 확인", "맥박 확인", "응급실"]
            }
        }
        
        print(f"✅ 의료 용어 사전 구축 완료: {len(self.medical_terms)}개 카테고리")
        return self.medical_terms
    
    def build_consultation_scenarios(self):
        """상담 시나리오 구축"""
        print("💬 상담 시나리오 구축 중...")
        
        self.consultation_scenarios = [
            {
                "scenario": "일반 상담",
                "pattern": "환자: {symptom}이 있어요.\n의료진: {symptom}이 있으시군요. 언제부터 시작되었나요?",
                "keywords": ["증상", "통증", "불편함"],
                "response_template": "네, {symptom}이 있으시군요. 추가로 어떤 증상이 있으신지, 언제부터 시작되었는지 알려주시면 더 정확한 상담을 도와드릴 수 있습니다."
            },
            {
                "scenario": "응급 상담",
                "pattern": "환자: 응급상황이에요! {emergency_symptom}\n의료진: 즉시 119에 연락하세요!",
                "keywords": ["응급", "119", "구급차", "생명", "위험"],
                "response_template": "응급상황이시군요! 즉시 119에 연락하여 구급차를 호출하시고, 응급실로 이송받으시기 바랍니다. 생명이 위험할 수 있으니 빠른 조치가 필요합니다."
            }
        ]
        
        print(f"✅ 상담 시나리오 구축 완료: {len(self.consultation_scenarios)}개 시나리오")
        return self.consultation_scenarios
    
    def build_emergency_protocols(self):
        """응급상황 대응 프로토콜 구축"""
        print("🚨 응급상황 대응 프로토콜 구축 중...")
        
        self.emergency_protocols = {
            "심장마비": {
                "symptoms": ["가슴 통증", "호흡 곤란", "의식 잃음", "팔 통증"],
                "action": "즉시 119에 연락하고 심폐소생술(CPR)을 시작하세요.",
                "priority": "최고"
            },
            "뇌졸중": {
                "symptoms": ["마비", "언어 장애", "의식 장애", "두통"],
                "action": "즉시 119에 연락하고 응급실로 이송하세요. 시간이 생명입니다.",
                "priority": "최고"
            },
            "출혈": {
                "symptoms": ["과다 출혈", "지속적 출혈", "피가 멈추지 않음"],
                "action": "출혈 부위를 압박하여 지혈하고 119에 연락하세요.",
                "priority": "높음"
            }
        }
        
        print(f"✅ 응급상황 대응 프로토콜 구축 완료: {len(self.emergency_protocols)}개 프로토콜")
        return self.emergency_protocols
    
    def build_complete_system(self):
        """완전한 의료 시스템 구축"""
        print("🏥 완전한 의료 시스템 구축 시작")
        print("=" * 50)
        
        # 각 구성 요소 구축
        medical_terms = self.build_medical_terms_dictionary()
        scenarios = self.build_consultation_scenarios()
        protocols = self.build_emergency_protocols()
        
        # 통합 시스템 구성
        complete_system = {
            "medical_terms": medical_terms,
            "consultation_scenarios": scenarios,
            "emergency_protocols": protocols,
            "metadata": {
                "version": "1.0",
                "created_date": "2024-01-01",
                "description": "한국어 의료 상담 챗봇을 위한 완전한 의료 시스템"
            }
        }
        
        # 시스템 저장
        output_path = "medical_system.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(complete_system, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 완전한 의료 시스템 구축 완료!")
        print(f"📁 시스템 저장 위치: {output_path}")
        
        return complete_system

def main():
    """메인 실행 함수"""
    print("🏥 의료 시스템 구축 시작")
    print("=" * 50)
    
    # 시스템 구축자 초기화
    builder = MedicalSystemBuilder()
    
    # 완전한 시스템 구축
    system = builder.build_complete_system()
    
    print("\n🎉 의료 시스템 구축 완료!")

if __name__ == "__main__":
    main()