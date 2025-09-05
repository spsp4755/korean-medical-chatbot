#!/usr/bin/env python3
"""
모델 선택 실행 스크립트
한국어 LLM 모델을 비교하고 최적의 모델을 선택합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.model_selector import ModelSelector

def main():
    """메인 함수"""
    print("🚀 한국어 LLM 모델 선택 프로세스 시작")
    print("=" * 60)
    
    # 1. 모델 비교 데이터 확인
    comparison_results_path = "outputs/model_comparison_results.json"
    
    if not Path(comparison_results_path).exists():
        print("❌ 모델 비교 데이터가 없습니다.")
        print("먼저 다음 명령어를 실행해주세요:")
        print("python scripts/model_comparison.py")
        return
    
    # 2. 모델 선택기 초기화
    selector = ModelSelector()
    
    try:
        # 3. 모델 비교 수행
        print("📊 모델 비교 분석 중...")
        comparison = selector.compare_models(
            "psymon/KoLlama2-7b",
            "beomi/KcBERT-base"
        )
        
        # 4. 결과 출력
        print("\n🏆 모델 선택 결과")
        print("=" * 60)
        print(f"추천 모델: {comparison.winner}")
        print(f"선택 이유: {comparison.reasoning}")
        
        print("\n📋 상세 비교:")
        print(f"  {comparison.model1.name}:")
        print(f"    - 크기: {comparison.model1.size_gb:.1f}GB")
        print(f"    - 속도: {comparison.model1.inference_speed:.1f} 토큰/초")
        print(f"    - 한국어 점수: {comparison.model1.korean_score:.2f}")
        
        print(f"  {comparison.model2.name}:")
        print(f"    - 크기: {comparison.model2.size_gb:.1f}GB")
        print(f"    - 속도: {comparison.model2.inference_speed:.1f} 토큰/초")
        print(f"    - 한국어 점수: {comparison.model2.korean_score:.2f}")
        
        # 5. 보고서 저장
        print("\n📄 보고서 생성 중...")
        selector.save_comparison_report(comparison)
        
        # 6. 설정 파일 업데이트 제안
        print("\n🔧 다음 단계:")
        print("1. 선택된 모델로 configs/config.yaml 업데이트")
        print("2. 베이스라인 모델 성능 측정 시작")
        print("3. 의료 도메인 파인튜닝 준비")
        
        # 7. 설정 파일 업데이트 코드 생성
        selected_model = comparison.winner
        config_update_code = f"""
# configs/config.yaml에서 다음 부분을 업데이트하세요:
model:
  base_model_name: "{selected_model}"
"""
        
        print("\n📝 설정 파일 업데이트:")
        print(config_update_code)
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        print("모델 비교 데이터를 확인해주세요.")

if __name__ == "__main__":
    main()
