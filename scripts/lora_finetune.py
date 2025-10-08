#!/usr/bin/env python3
"""
LoRA 파인튜닝 실행 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimization.lora import LoRAFineTuner
import yaml


def main():
    """LoRA 파인튜닝 실험 실행"""
    print("🚀 LoRA 파인튜닝 실험 시작")
    print("=" * 60)
    
    # 설정 로드
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 베이스 모델: 원본 Student 모델 사용 (토크나이저 호환성 문제 해결)
    base_model = "skt/kogpt2-base-v2"
    
    print(f"📊 베이스 모델: {base_model}")
    print(f"🎯 목표: LoRA 파인튜닝으로 속도 향상 100% 달성")
    
    # LoRA FineTuner 초기화
    lora_tuner = LoRAFineTuner(
        model_name=base_model,
        output_dir="models/lora"
    )
    
    # LoRA 파인튜닝 실행
    results = lora_tuner.run_lora_finetuning(
        r=16,                    # rank
        lora_alpha=32,          # LoRA alpha
        lora_dropout=0.1,       # LoRA dropout
        num_epochs=3,           # 학습 에포크
        batch_size=2,           # 배치 크기
        learning_rate=2e-4      # 학습률
    )
    
    # 결과 분석
    lora_tuner.analyze_results(results)
    
    print("\n✅ LoRA 파인튜닝 실험 완료!")


if __name__ == "__main__":
    main()
