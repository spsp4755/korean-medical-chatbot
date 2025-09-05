#!/usr/bin/env python3
"""
환경 확인 스크립트
GPU 사용 가능 여부와 시스템 리소스를 확인합니다.
"""

import torch
import psutil
import platform

def check_environment():
    """시스템 환경을 확인합니다."""
    print("🖥️ 시스템 환경 확인")
    print("=" * 50)
    
    # 기본 시스템 정보
    print(f"운영체제: {platform.system()} {platform.release()}")
    print(f"Python 버전: {platform.python_version()}")
    print(f"PyTorch 버전: {torch.__version__}")
    
    # CPU 정보
    cpu_count = psutil.cpu_count()
    print(f"CPU 코어 수: {cpu_count}")
    
    # 메모리 정보
    memory = psutil.virtual_memory()
    print(f"총 RAM: {memory.total / 1024**3:.1f}GB")
    print(f"사용 가능 RAM: {memory.available / 1024**3:.1f}GB")
    print(f"RAM 사용률: {memory.percent}%")
    
    # GPU 정보
    print("\n🎮 GPU 정보:")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ GPU 사용 가능: {gpu_count}개")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # 현재 GPU 메모리 사용량
        current_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"현재 GPU 메모리 사용량: {current_memory:.2f}GB")
        
    else:
        print("❌ GPU 사용 불가 (CPU 모드만 가능)")
    
    # 모델 크기별 권장사항
    print("\n📋 모델 크기별 권장사항:")
    print("=" * 50)
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory >= 16:
            print("✅ 16GB+ GPU: KoLlama2-7b (13.5GB) 실행 가능")
        elif gpu_memory >= 8:
            print("⚠️ 8GB GPU: KoLlama2-7b는 메모리 부족 가능성")
            print("   권장: 더 작은 모델 사용")
        else:
            print("❌ 8GB 미만 GPU: KoLlama2-7b 실행 불가")
            print("   권장: BERT 계열 모델 사용")
    else:
        print("❌ CPU 모드: KoLlama2-7b 실행 불가 (너무 느림)")
        print("   권장: BERT 계열 모델 사용")
    
    # 추천 모델
    print("\n🎯 추천 모델:")
    print("=" * 50)
    
    if torch.cuda.is_available() and gpu_memory >= 16:
        print("1. psymon/KoLlama2-7b (한국어 LLM, 13.5GB)")
        print("2. beomi/KcBERT-base (한국어 BERT, 1.1GB)")
    else:
        print("1. beomi/KcBERT-base (한국어 BERT, 1.1GB)")
        print("2. klue/roberta-base (KLUE RoBERTa, 1.1GB)")
        print("3. beomi/KoAlpaca-Polyglot-12.8B-v1.1 (더 큰 모델, GPU 필요)")

def main():
    """메인 함수"""
    check_environment()

if __name__ == "__main__":
    main()
