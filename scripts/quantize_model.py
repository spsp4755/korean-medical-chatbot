#!/usr/bin/env python3
"""
모델 양자화 실행 스크립트
42dot 모델을 INT8로 양자화하여 크기와 속도를 개선합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.optimization.quantization import ModelQuantizer
import yaml
import json

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    """메인 함수"""
    print("🚀 모델 양자화 실험 시작")
    print("=" * 60)
    
    # 설정 로드
    config = load_config()
    model_name = config["model"]["base_model_name"]
    
    print(f"📊 대상 모델: {model_name}")
    print(f"🎯 목표: 모델 크기 50% 감소, 추론 속도 2배 향상")
    
    # 양자화기 초기화
    quantizer = ModelQuantizer(
        model_name=model_name,
        output_dir="models/quantized"
    )
    
    try:
        # 양자화 실험 실행 (모델 압축)
        results = quantizer.run_quantization_experiment(["compression"])
        
        # 결과 분석
        print("\n📊 양자화 실험 결과 분석:")
        print("=" * 60)
        
        if "original" in results:
            orig = results["original"]
            print(f"🔹 원본 모델:")
            print(f"  - 크기: {orig['model_size_mb']:.2f}MB")
            print(f"  - 추론 속도: {orig['inference_speed_seconds']:.2f}초")
            print(f"  - 메모리 사용량: {orig['memory_usage_mb']:.2f}MB")
        
        if "int8" in results:
            if "error" in results["int8"]:
                print(f"❌ INT8 양자화 실패: {results['int8']['error']}")
            else:
                int8 = results["int8"]
                print(f"🔹 INT8 양자화 모델:")
                print(f"  - 크기: {int8['quantized_size_mb']:.2f}MB")
                print(f"  - 추론 속도: {int8['quantized_speed_seconds']:.2f}초")
                print(f"  - 메모리 사용량: {int8['quantized_memory_mb']:.2f}MB")
                print(f"  - 크기 감소: {int8['size_reduction_percent']:.1f}%")
                print(f"  - 속도 향상: {int8['speed_improvement_percent']:.1f}%")
                
                # 목표 달성 여부 확인
                print(f"\n🎯 목표 달성 여부:")
                size_goal = int8['size_reduction_percent'] >= 50
                speed_goal = int8['speed_improvement_percent'] >= 100
                
                print(f"  - 크기 50% 감소: {'✅' if size_goal else '❌'} ({int8['size_reduction_percent']:.1f}%)")
                print(f"  - 속도 2배 향상: {'✅' if speed_goal else '❌'} ({int8['speed_improvement_percent']:.1f}%)")
                
                if size_goal and speed_goal:
                    print(f"\n🎉 모든 목표 달성!")
                else:
                    print(f"\n⚠️ 일부 목표 미달성 - 추가 최적화 필요")
        
        # 다음 단계 제안
        print(f"\n🚀 다음 단계:")
        print(f"1. Pruning 기법 적용 (가중치 20-30% 제거)")
        print(f"2. Knowledge Distillation 구현")
        print(f"3. LoRA 파인튜닝 적용")
        
    except Exception as e:
        print(f"❌ 양자화 실험 실패: {str(e)}")
        print(f"오류 상세: {type(e).__name__}")

if __name__ == "__main__":
    main()
