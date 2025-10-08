#!/usr/bin/env python3
"""
Structured Pruning 실행 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimization.pruning import ModelPruner
import yaml


def main():
    """Pruning 실험 실행"""
    print("🚀 모델 Pruning 실험 시작")
    print("=" * 60)
    
    # 설정 로드
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_name = config['model']['base_model_name']
    print(f"📊 대상 모델: {model_name}")
    print(f"🎯 목표: 가중치 20-30% 제거로 추가 경량화")
    
    # Pruner 초기화
    pruner = ModelPruner(
        model_name=model_name,
        output_dir="models/pruned"
    )
    
    # Pruning 실험 실행
    results = pruner.run_pruning_experiment(
        sparsity_levels=[0.1, 0.2, 0.3]  # 10%, 20%, 30% 제거
    )
    
    # 결과 분석
    pruner.analyze_results(results)
    
    print("\n✅ Pruning 실험 완료!")


if __name__ == "__main__":
    main()










