#!/usr/bin/env python3
"""
객관식 데이터를 제외하고 주관식 데이터만 필터링하는 스크립트
- 객관식: 1) 2) 3) 4) 5) 형태의 선택지가 있는 질문
- 주관식: 긴 설명형 답변을 가진 질문
"""

import json
import os
import re
from typing import List, Dict, Any

def is_objective_question(question: str) -> bool:
    """객관식 질문인지 판단"""
    # 객관식 패턴: 1) 2) 3) 4) 5) 형태의 선택지가 있는지 확인
    objective_patterns = [
        r'\d+\)\s+',  # 1) 2) 3) 4) 5) 패턴
        r'\(\d+\)\s+',  # (1) (2) (3) (4) (5) 패턴
        r'[A-E]\)\s+',  # A) B) C) D) E) 패턴
        r'[가-힣]\)\s+',  # 가) 나) 다) 라) 마) 패턴
    ]
    
    for pattern in objective_patterns:
        if re.search(pattern, question):
            return True
    return False

def is_subjective_answer(answer: str) -> bool:
    """주관식 답변인지 판단"""
    # 답변이 너무 짧으면 객관식으로 간주
    if len(answer.strip()) < 50:
        return False
    
    # 객관식 답변 패턴 확인
    objective_answer_patterns = [
        r'^\d+\)\s+',  # 1) 2) 3) 4) 5) 로 시작
        r'^[A-E]\)\s+',  # A) B) C) D) E) 로 시작
        r'^[가-힣]\)\s+',  # 가) 나) 다) 라) 마) 로 시작
    ]
    
    for pattern in objective_answer_patterns:
        if re.match(pattern, answer.strip()):
            return False
    
    return True

def filter_subjective_data(input_file: str, output_file: str) -> Dict[str, int]:
    """주관식 데이터만 필터링"""
    print(f"📖 데이터 로딩: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 전체 데이터 수: {len(data)}")
    
    # 필터링된 데이터
    filtered_data = []
    stats = {
        'total': len(data),
        'objective_removed': 0,
        'subjective_kept': 0,
        'short_answer_removed': 0
    }
    
    for item in data:
        question = item.get('question', '')
        answer = item.get('answer', '')
        
        # 객관식 질문인지 확인
        if is_objective_question(question):
            stats['objective_removed'] += 1
            continue
        
        # 주관식 답변인지 확인
        if not is_subjective_answer(answer):
            stats['short_answer_removed'] += 1
            continue
        
        # 주관식 데이터로 판단
        filtered_data.append(item)
        stats['subjective_kept'] += 1
    
    # 필터링된 데이터 저장
    print(f"💾 필터링된 데이터 저장: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    return stats

def main():
    """메인 실행 함수"""
    print("🔍 주관식 데이터 필터링 시작")
    print("=" * 50)
    
    # 데이터 파일 경로 (Essential + Professional)
    data_files = [
        {
            'input': 'data/processed/splits/essential_medical_train.json',
            'output': 'data/processed/splits/essential_medical_train_subjective.json'
        },
        {
            'input': 'data/processed/splits/essential_medical_val.json',
            'output': 'data/processed/splits/essential_medical_val_subjective.json'
        },
        {
            'input': 'data/processed/splits/essential_medical_test.json',
            'output': 'data/processed/splits/essential_medical_test_subjective.json'
        },
        {
            'input': 'data/processed/splits/professional_medical_train.json',
            'output': 'data/processed/splits/professional_medical_train_subjective.json'
        },
        {
            'input': 'data/processed/splits/professional_medical_val.json',
            'output': 'data/processed/splits/professional_medical_val_subjective.json'
        },
        {
            'input': 'data/processed/splits/professional_medical_test.json',
            'output': 'data/processed/splits/professional_medical_test_subjective.json'
        }
    ]
    
    total_stats = {
        'total': 0,
        'objective_removed': 0,
        'subjective_kept': 0,
        'short_answer_removed': 0
    }
    
    # 각 파일 처리
    for file_info in data_files:
        if os.path.exists(file_info['input']):
            print(f"\n📁 처리 중: {file_info['input']}")
            stats = filter_subjective_data(file_info['input'], file_info['output'])
            
            # 통계 누적
            for key in total_stats:
                total_stats[key] += stats[key]
            
            print(f"  ✅ 객관식 제거: {stats['objective_removed']}개")
            print(f"  ✅ 주관식 유지: {stats['subjective_kept']}개")
            print(f"  ✅ 짧은 답변 제거: {stats['short_answer_removed']}개")
        else:
            print(f"⚠️  파일 없음: {file_info['input']}")
    
    # 전체 통계 출력
    print("\n" + "=" * 50)
    print("📊 전체 필터링 결과")
    print("=" * 50)
    print(f"📈 전체 데이터: {total_stats['total']}개")
    print(f"🗑️  객관식 제거: {total_stats['objective_removed']}개")
    print(f"🗑️  짧은 답변 제거: {total_stats['short_answer_removed']}개")
    print(f"✅ 주관식 유지: {total_stats['subjective_kept']}개")
    print(f"📉 데이터 감소율: {((total_stats['total'] - total_stats['subjective_kept']) / total_stats['total'] * 100):.1f}%")
    
    # 메모리 절약 효과 추정
    original_size = total_stats['total']
    filtered_size = total_stats['subjective_kept']
    memory_reduction = (original_size - filtered_size) / original_size * 100
    
    print(f"\n💾 메모리 절약 효과:")
    print(f"  - 데이터 크기 감소: {memory_reduction:.1f}%")
    print(f"  - 학습 시간 단축: 약 {memory_reduction:.1f}%")
    print(f"  - 메모리 사용량 감소: 약 {memory_reduction:.1f}%")

if __name__ == "__main__":
    main()
