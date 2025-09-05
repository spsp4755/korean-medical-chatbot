"""
모델 선택 및 평가를 위한 클래스
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM

@dataclass
class ModelSpecs:
    """모델 사양 정보"""
    name: str
    size_gb: float
    parameter_count: int
    vocab_size: int
    max_length: int
    inference_speed: float  # tokens per second
    korean_score: float  # 0-1
    memory_usage: float  # GB
    load_time: float  # seconds

@dataclass
class ModelComparison:
    """모델 비교 결과"""
    model1: ModelSpecs
    model2: ModelSpecs
    winner: str
    reasoning: str
    recommendation: str

class ModelSelector:
    """모델 선택 및 평가 클래스"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.available_models = [
            "psymon/KoLlama2-7b",
            "beomi/KcBERT-base"
        ]
    
    def load_model_specs(self, results_path: str = "outputs/model_comparison_results.json") -> Dict[str, ModelSpecs]:
        """모델 비교 결과를 로드하여 ModelSpecs 객체로 변환"""
        results_path = Path(results_path)
        
        if not results_path.exists():
            raise FileNotFoundError(f"모델 비교 결과 파일을 찾을 수 없습니다: {results_path}")
        
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        model_specs = {}
        
        for model_name, result in results.items():
            if "error" in result.get("model_info", {}):
                continue
            
            info = result["model_info"]
            speed = result.get("speed_benchmark", {})
            korean = result.get("korean_understanding", {})
            
            model_specs[model_name] = ModelSpecs(
                name=model_name,
                size_gb=info.get("model_size_gb", 0),
                parameter_count=info.get("parameter_count", 0),
                vocab_size=info.get("vocab_size", 0),
                max_length=info.get("max_length", 0),
                inference_speed=speed.get("tokens_per_second", 0),
                korean_score=korean.get("avg_keyword_score", 0),
                memory_usage=0,  # 별도 측정 필요
                load_time=0  # 별도 측정 필요
            )
        
        return model_specs
    
    def compare_models(self, model1_name: str, model2_name: str) -> ModelComparison:
        """두 모델을 비교하여 추천 모델을 결정"""
        try:
            model_specs = self.load_model_specs()
            
            if model1_name not in model_specs or model2_name not in model_specs:
                raise ValueError("모델 사양 정보를 찾을 수 없습니다.")
            
            model1 = model_specs[model1_name]
            model2 = model_specs[model2_name]
            
            # 비교 기준별 점수 계산
            scores = {
                model1_name: 0,
                model2_name: 0
            }
            
            # 1. 모델 크기 (작을수록 좋음) - 25% 가중치
            if model1.size_gb < model2.size_gb:
                scores[model1_name] += 25
            elif model2.size_gb < model1.size_gb:
                scores[model2_name] += 25
            else:
                scores[model1_name] += 12.5
                scores[model2_name] += 12.5
            
            # 2. 추론 속도 (빠를수록 좋음) - 25% 가중치
            if model1.inference_speed > model2.inference_speed:
                scores[model1_name] += 25
            elif model2.inference_speed > model1.inference_speed:
                scores[model2_name] += 25
            else:
                scores[model1_name] += 12.5
                scores[model2_name] += 12.5
            
            # 3. 한국어 이해 능력 (높을수록 좋음) - 30% 가중치
            if model1.korean_score > model2.korean_score:
                scores[model1_name] += 30
            elif model2.korean_score > model1.korean_score:
                scores[model2_name] += 30
            else:
                scores[model1_name] += 15
                scores[model2_name] += 15
            
            # 4. 파라미터 수 (적당한 수준이 좋음) - 20% 가중치
            # 7B-12B 범위에서 7B가 더 효율적
            if 6e9 <= model1.parameter_count <= 8e9 and not (6e9 <= model2.parameter_count <= 8e9):
                scores[model1_name] += 20
            elif 6e9 <= model2.parameter_count <= 8e9 and not (6e9 <= model1.parameter_count <= 8e9):
                scores[model2_name] += 20
            else:
                scores[model1_name] += 10
                scores[model2_name] += 10
            
            # 승자 결정
            winner = model1_name if scores[model1_name] > scores[model2_name] else model2_name
            
            # 추천 이유 생성
            reasoning = self._generate_reasoning(model1, model2, scores)
            
            # 최종 추천사항
            recommendation = self._generate_recommendation(winner, model1, model2)
            
            return ModelComparison(
                model1=model1,
                model2=model2,
                winner=winner,
                reasoning=reasoning,
                recommendation=recommendation
            )
            
        except Exception as e:
            raise Exception(f"모델 비교 중 오류 발생: {str(e)}")
    
    def _generate_reasoning(self, model1: ModelSpecs, model2: ModelSpecs, scores: Dict[str, float]) -> str:
        """비교 결과에 대한 상세한 이유 생성"""
        reasoning_parts = []
        
        # 크기 비교
        if model1.size_gb < model2.size_gb:
            reasoning_parts.append(f"{model1.name}이 {model2.size_gb - model1.size_gb:.1f}GB 더 작습니다.")
        elif model2.size_gb < model1.size_gb:
            reasoning_parts.append(f"{model2.name}이 {model1.size_gb - model2.size_gb:.1f}GB 더 작습니다.")
        
        # 속도 비교
        if model1.inference_speed > model2.inference_speed:
            reasoning_parts.append(f"{model1.name}이 {model1.inference_speed - model2.inference_speed:.1f} 토큰/초 더 빠릅니다.")
        elif model2.inference_speed > model1.inference_speed:
            reasoning_parts.append(f"{model2.name}이 {model2.inference_speed - model1.inference_speed:.1f} 토큰/초 더 빠릅니다.")
        
        # 한국어 능력 비교
        if model1.korean_score > model2.korean_score:
            reasoning_parts.append(f"{model1.name}의 한국어 이해 점수가 {model1.korean_score - model2.korean_score:.2f} 더 높습니다.")
        elif model2.korean_score > model1.korean_score:
            reasoning_parts.append(f"{model2.name}의 한국어 이해 점수가 {model2.korean_score - model1.korean_score:.2f} 더 높습니다.")
        
        return " ".join(reasoning_parts)
    
    def _generate_recommendation(self, winner: str, model1: ModelSpecs, model2: ModelSpecs) -> str:
        """최종 추천사항 생성"""
        winner_model = model1 if winner == model1.name else model2
        
        recommendations = [
            f"추천 모델: {winner}",
            f"모델 크기: {winner_model.size_gb:.1f}GB",
            f"추론 속도: {winner_model.inference_speed:.1f} 토큰/초",
            f"한국어 점수: {winner_model.korean_score:.2f}",
            "",
            "다음 단계:",
            "1. 선택된 모델로 베이스라인 성능 측정",
            "2. 의료 도메인 데이터로 파인튜닝 준비",
            "3. 경량화 기법 적용 계획 수립"
        ]
        
        return "\n".join(recommendations)
    
    def save_comparison_report(self, comparison: ModelComparison, output_path: str = "outputs/model_selection_report.md"):
        """모델 비교 보고서를 Markdown 형식으로 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = f"""# 모델 선택 보고서

## 📊 모델 비교 결과

### {comparison.model1.name}
- **크기**: {comparison.model1.size_gb:.1f}GB
- **파라미터 수**: {comparison.model1.parameter_count:,}
- **추론 속도**: {comparison.model1.inference_speed:.1f} 토큰/초
- **한국어 점수**: {comparison.model1.korean_score:.2f}

### {comparison.model2.name}
- **크기**: {comparison.model2.size_gb:.1f}GB
- **파라미터 수**: {comparison.model2.parameter_count:,}
- **추론 속도**: {comparison.model2.inference_speed:.1f} 토큰/초
- **한국어 점수**: {comparison.model2.korean_score:.2f}

## 🏆 추천 모델: {comparison.winner}

### 📋 선택 이유
{comparison.reasoning}

### 🎯 추천사항
{comparison.recommendation}

## 📅 다음 단계
1. **베이스라인 모델 구축**: 선택된 모델로 기본 성능 측정
2. **의료 도메인 파인튜닝**: AI Hub 의료 데이터 활용
3. **경량화 기법 적용**: Quantization, Pruning, Knowledge Distillation
4. **성능 최적화**: 추론 속도 및 메모리 사용량 개선

---
*생성일시: {Path().cwd()}*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📋 모델 선택 보고서가 {output_path}에 저장되었습니다.")

def main():
    """메인 함수"""
    selector = ModelSelector()
    
    try:
        # 모델 비교 수행
        comparison = selector.compare_models(
            "beomi/KULLM-2-7B",
            "EleutherAI/polyglot-ko-12.8b"
        )
        
        # 결과 출력
        print("🏆 모델 선택 결과")
        print("=" * 50)
        print(f"추천 모델: {comparison.winner}")
        print(f"선택 이유: {comparison.reasoning}")
        print("\n추천사항:")
        print(comparison.recommendation)
        
        # 보고서 저장
        selector.save_comparison_report(comparison)
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        print("먼저 scripts/model_comparison.py를 실행하여 모델 비교 데이터를 생성해주세요.")

if __name__ == "__main__":
    main()
