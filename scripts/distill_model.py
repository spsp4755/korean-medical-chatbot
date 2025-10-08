#!/usr/bin/env python3
"""
Knowledge Distillation 실행 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimization.distillation import KnowledgeDistiller
import yaml


def main():
    """Knowledge Distillation 실험 실행"""
    print("🚀 Knowledge Distillation 실험 시작")
    print("=" * 60)
    
    # 설정 로드
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Teacher 모델: 20% Pruned 모델 사용
    teacher_model = "models/pruned/model_20_percent_pruned"
    
    # Student 모델: 더 작은 모델 선택
    student_model = "skt/kogpt2-base-v2"  # 124M 파라미터
    
    print(f"📊 Teacher 모델: {teacher_model}")
    print(f"📊 Student 모델: {student_model}")
    print(f"🎯 목표: Teacher 지식을 Student로 전달하여 경량화")
    
    # Distiller 초기화
    distiller = KnowledgeDistiller(
        teacher_model_name=teacher_model,
        student_model_name=student_model,
        output_dir="models/distilled"
    )
    
    # Distillation 실행
    results = distiller.run_distillation(
        temperature=3.0,      # Soft targets의 부드러움 조절
        alpha=0.7,           # Distillation loss weight
        num_epochs=2,        # 학습 에포크 (시간 절약)
        batch_size=2,        # 배치 크기 (메모리 절약)
        learning_rate=5e-5   # 학습률
    )
    
    # 결과 분석
    distiller.analyze_results(results)
    
    print("\n✅ Knowledge Distillation 실험 완료!")


if __name__ == "__main__":
    main()
