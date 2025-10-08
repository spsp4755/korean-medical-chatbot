import json
import os
from datetime import datetime

def load_evaluation_results():
    """3개 프롬프트 평가 결과 로드"""
    results = {}
    
    # Basic 프롬프트 결과
    if os.path.exists("results/peft_model_evaluation.json"):
        with open("results/peft_model_evaluation.json", "r", encoding="utf-8") as f:
            results["basic"] = json.load(f)
    
    # Improved 프롬프트 결과
    if os.path.exists("results/improved_prompt_evaluation.json"):
        with open("results/improved_prompt_evaluation.json", "r", encoding="utf-8") as f:
            results["improved"] = json.load(f)
    
    # Optimized 프롬프트 결과
    if os.path.exists("results/optimized_prompt_evaluation.json"):
        with open("results/optimized_prompt_evaluation.json", "r", encoding="utf-8") as f:
            results["optimized"] = json.load(f)
    
    return results

def compare_three_prompts(results):
    """3개 프롬프트 성능 비교"""
    comparison = {
        "comparison_timestamp": datetime.now().isoformat(),
        "prompt_types": {
            "basic": "간단한 프롬프트 (질문: {question}\\n답변:)",
            "improved": "복잡한 프롬프트 (응급상황 판단, 실용적 조치 등)",
            "optimized": "최적화된 프롬프트 (Basic + 최소한의 개선)"
        },
        "performance_comparison": {}
    }
    
    # 각 프롬프트별 메트릭 추출
    basic_bertscore = results["basic"]["bertscore"]
    basic_medical = results["basic"]["medical_metrics"]
    
    improved_bertscore = results["improved"]["bertscore"]
    improved_medical = results["improved"]["medical_metrics"]
    
    optimized_bertscore = results["optimized"]["bertscore"]
    optimized_medical = results["optimized"]["medical_metrics"]
    
    # BERTScore F1 비교
    comparison["performance_comparison"]["bertscore_f1"] = {
        "basic": {
            "value": basic_bertscore["bertscore_f1"],
            "formatted": f"{basic_bertscore['bertscore_f1']:.4f}"
        },
        "improved": {
            "value": improved_bertscore["bertscore_f1"],
            "formatted": f"{improved_bertscore['bertscore_f1']:.4f}",
            "vs_basic": f"{((improved_bertscore['bertscore_f1'] - basic_bertscore['bertscore_f1']) / basic_bertscore['bertscore_f1'] * 100):+.1f}%"
        },
        "optimized": {
            "value": optimized_bertscore["bertscore_f1"],
            "formatted": f"{optimized_bertscore['bertscore_f1']:.4f}",
            "vs_basic": f"{((optimized_bertscore['bertscore_f1'] - basic_bertscore['bertscore_f1']) / basic_bertscore['bertscore_f1'] * 100):+.1f}%"
        },
        "best": "basic" if basic_bertscore["bertscore_f1"] >= max(improved_bertscore["bertscore_f1"], optimized_bertscore["bertscore_f1"]) 
                else "improved" if improved_bertscore["bertscore_f1"] >= optimized_bertscore["bertscore_f1"] else "optimized"
    }
    
    # 응답 길이 비교
    comparison["performance_comparison"]["response_length"] = {
        "basic": {
            "value": basic_medical["avg_response_length"],
            "formatted": f"{basic_medical['avg_response_length']:.1f}자"
        },
        "improved": {
            "value": improved_medical["avg_response_length"],
            "formatted": f"{improved_medical['avg_response_length']:.1f}자",
            "vs_basic": f"{((improved_medical['avg_response_length'] - basic_medical['avg_response_length']) / basic_medical['avg_response_length'] * 100):+.1f}%"
        },
        "optimized": {
            "value": optimized_medical["avg_response_length"],
            "formatted": f"{optimized_medical['avg_response_length']:.1f}자",
            "vs_basic": f"{((optimized_medical['avg_response_length'] - basic_medical['avg_response_length']) / basic_medical['avg_response_length'] * 100):+.1f}%"
        },
        "best": "basic" if basic_medical["avg_response_length"] <= min(improved_medical["avg_response_length"], optimized_medical["avg_response_length"])
                else "improved" if improved_medical["avg_response_length"] <= optimized_medical["avg_response_length"] else "optimized"
    }
    
    # 의료 용어 포함도 비교
    comparison["performance_comparison"]["medical_coverage"] = {
        "basic": {
            "value": basic_medical["medical_coverage"],
            "formatted": f"{basic_medical['medical_coverage']:.4f}"
        },
        "improved": {
            "value": improved_medical["medical_coverage"],
            "formatted": f"{improved_medical['medical_coverage']:.4f}",
            "vs_basic": f"{((improved_medical['medical_coverage'] - basic_medical['medical_coverage']) / basic_medical['medical_coverage'] * 100):+.1f}%"
        },
        "optimized": {
            "value": optimized_medical["medical_coverage"],
            "formatted": f"{optimized_medical['medical_coverage']:.4f}",
            "vs_basic": f"{((optimized_medical['medical_coverage'] - basic_medical['medical_coverage']) / basic_medical['medical_coverage'] * 100):+.1f}%"
        },
        "best": "basic" if basic_medical["medical_coverage"] >= max(improved_medical["medical_coverage"], optimized_medical["medical_coverage"])
                else "improved" if improved_medical["medical_coverage"] >= optimized_medical["medical_coverage"] else "optimized"
    }
    
    # 안전성 점수 비교
    comparison["performance_comparison"]["safety_score"] = {
        "basic": {
            "value": basic_medical["safety_score"],
            "formatted": f"{basic_medical['safety_score']:.4f}"
        },
        "improved": {
            "value": improved_medical["safety_score"],
            "formatted": f"{improved_medical['safety_score']:.4f}",
            "vs_basic": f"{((improved_medical['safety_score'] - basic_medical['safety_score']) / basic_medical['safety_score'] * 100):+.1f}%"
        },
        "optimized": {
            "value": optimized_medical["safety_score"],
            "formatted": f"{optimized_medical['safety_score']:.4f}",
            "vs_basic": f"{((optimized_medical['safety_score'] - basic_medical['safety_score']) / basic_medical['safety_score'] * 100):+.1f}%"
        },
        "best": "basic" if basic_medical["safety_score"] >= max(improved_medical["safety_score"], optimized_medical["safety_score"])
                else "improved" if improved_medical["safety_score"] >= optimized_medical["safety_score"] else "optimized"
    }
    
    # 응답 품질 비교
    comparison["performance_comparison"]["response_quality"] = {
        "basic": {
            "value": basic_medical["response_quality"],
            "formatted": f"{basic_medical['response_quality']:.4f}"
        },
        "improved": {
            "value": improved_medical["response_quality"],
            "formatted": f"{improved_medical['response_quality']:.4f}",
            "vs_basic": f"{((improved_medical['response_quality'] - basic_medical['response_quality']) / basic_medical['response_quality'] * 100):+.1f}%"
        },
        "optimized": {
            "value": optimized_medical["response_quality"],
            "formatted": f"{optimized_medical['response_quality']:.4f}",
            "vs_basic": f"{((optimized_medical['response_quality'] - basic_medical['response_quality']) / basic_medical['response_quality'] * 100):+.1f}%"
        },
        "best": "basic" if basic_medical["response_quality"] >= max(improved_medical["response_quality"], optimized_medical["response_quality"])
                else "improved" if improved_medical["response_quality"] >= optimized_medical["response_quality"] else "optimized"
    }
    
    # 전체 성능 점수 계산 (각 메트릭의 상대적 성능)
    def calculate_overall_score(bertscore_f1, response_length, medical_coverage, safety_score, response_quality):
        # BERTScore F1 (높을수록 좋음)
        f1_score = bertscore_f1
        
        # 응답 길이 (짧을수록 좋음, 200자 기준으로 정규화)
        length_score = max(0, 1 - abs(response_length - 200) / 200)
        
        # 의료 용어 포함도 (높을수록 좋음)
        coverage_score = medical_coverage
        
        # 안전성 점수 (높을수록 좋음)
        safety_score_norm = safety_score
        
        # 응답 품질 (높을수록 좋음)
        quality_score = response_quality
        
        # 가중 평균 (BERTScore와 안전성이 가장 중요)
        overall = (f1_score * 0.3 + length_score * 0.1 + coverage_score * 0.1 + 
                  safety_score_norm * 0.3 + quality_score * 0.2)
        
        return overall
    
    basic_overall = calculate_overall_score(
        basic_bertscore["bertscore_f1"],
        basic_medical["avg_response_length"],
        basic_medical["medical_coverage"],
        basic_medical["safety_score"],
        basic_medical["response_quality"]
    )
    
    improved_overall = calculate_overall_score(
        improved_bertscore["bertscore_f1"],
        improved_medical["avg_response_length"],
        improved_medical["medical_coverage"],
        improved_medical["safety_score"],
        improved_medical["response_quality"]
    )
    
    optimized_overall = calculate_overall_score(
        optimized_bertscore["bertscore_f1"],
        optimized_medical["avg_response_length"],
        optimized_medical["medical_coverage"],
        optimized_medical["safety_score"],
        optimized_medical["response_quality"]
    )
    
    comparison["performance_comparison"]["overall_score"] = {
        "basic": {
            "value": basic_overall,
            "formatted": f"{basic_overall:.4f}"
        },
        "improved": {
            "value": improved_overall,
            "formatted": f"{improved_overall:.4f}",
            "vs_basic": f"{((improved_overall - basic_overall) / basic_overall * 100):+.1f}%"
        },
        "optimized": {
            "value": optimized_overall,
            "formatted": f"{optimized_overall:.4f}",
            "vs_basic": f"{((optimized_overall - basic_overall) / basic_overall * 100):+.1f}%"
        },
        "best": "basic" if basic_overall >= max(improved_overall, optimized_overall)
                else "improved" if improved_overall >= optimized_overall else "optimized"
    }
    
    return comparison

def print_triple_comparison(comparison):
    """3개 프롬프트 비교 결과 출력"""
    print("🏆 3개 프롬프트 성능 비교 결과")
    print("="*80)
    
    perf = comparison["performance_comparison"]
    
    # BERTScore F1
    print(f"\n📊 BERTScore F1 (정확도)")
    print(f"  Basic:    {perf['bertscore_f1']['basic']['formatted']}")
    print(f"  Improved: {perf['bertscore_f1']['improved']['formatted']} ({perf['bertscore_f1']['improved']['vs_basic']})")
    print(f"  Optimized: {perf['bertscore_f1']['optimized']['formatted']} ({perf['bertscore_f1']['optimized']['vs_basic']})")
    print(f"  🥇 최고: {perf['bertscore_f1']['best'].upper()}")
    
    # 응답 길이
    print(f"\n📏 응답 길이")
    print(f"  Basic:    {perf['response_length']['basic']['formatted']}")
    print(f"  Improved: {perf['response_length']['improved']['formatted']} ({perf['response_length']['improved']['vs_basic']})")
    print(f"  Optimized: {perf['response_length']['optimized']['formatted']} ({perf['response_length']['optimized']['vs_basic']})")
    print(f"  🥇 최고: {perf['response_length']['best'].upper()}")
    
    # 의료 용어 포함도
    print(f"\n🏥 의료 용어 포함도")
    print(f"  Basic:    {perf['medical_coverage']['basic']['formatted']}")
    print(f"  Improved: {perf['medical_coverage']['improved']['formatted']} ({perf['medical_coverage']['improved']['vs_basic']})")
    print(f"  Optimized: {perf['medical_coverage']['optimized']['formatted']} ({perf['medical_coverage']['optimized']['vs_basic']})")
    print(f"  🥇 최고: {perf['medical_coverage']['best'].upper()}")
    
    # 안전성 점수
    print(f"\n🛡️ 안전성 점수")
    print(f"  Basic:    {perf['safety_score']['basic']['formatted']}")
    print(f"  Improved: {perf['safety_score']['improved']['formatted']} ({perf['safety_score']['improved']['vs_basic']})")
    print(f"  Optimized: {perf['safety_score']['optimized']['formatted']} ({perf['safety_score']['optimized']['vs_basic']})")
    print(f"  🥇 최고: {perf['safety_score']['best'].upper()}")
    
    # 응답 품질
    print(f"\n⭐ 응답 품질")
    print(f"  Basic:    {perf['response_quality']['basic']['formatted']}")
    print(f"  Improved: {perf['response_quality']['improved']['formatted']} ({perf['response_quality']['improved']['vs_basic']})")
    print(f"  Optimized: {perf['response_quality']['optimized']['formatted']} ({perf['response_quality']['optimized']['vs_basic']})")
    print(f"  🥇 최고: {perf['response_quality']['best'].upper()}")
    
    # 전체 성능 점수
    print(f"\n🏆 전체 성능 점수 (가중 평균)")
    print(f"  Basic:    {perf['overall_score']['basic']['formatted']}")
    print(f"  Improved: {perf['overall_score']['improved']['formatted']} ({perf['overall_score']['improved']['vs_basic']})")
    print(f"  Optimized: {perf['overall_score']['optimized']['formatted']} ({perf['overall_score']['optimized']['vs_basic']})")
    print(f"  🥇 최고: {perf['overall_score']['best'].upper()}")
    
    # 최종 권장사항
    best_prompt = perf['overall_score']['best']
    print(f"\n💡 최종 권장사항")
    print(f"  🎯 추천 프롬프트: {best_prompt.upper()}")
    
    if best_prompt == "basic":
        print("  📝 이유: 가장 안정적이고 균형잡힌 성능")
    elif best_prompt == "improved":
        print("  📝 이유: 복잡한 프롬프트가 오히려 성능 향상")
    else:
        print("  📝 이유: Basic의 장점을 유지하면서 개선된 성능")

def main():
    print("3개 프롬프트 성능 비교 분석 시작...")
    
    # 평가 결과 로드
    results = load_evaluation_results()
    
    if len(results) != 3:
        print(f"❌ 3개 프롬프트 결과가 모두 필요합니다. 현재: {len(results)}개")
        print("필요한 파일:")
        print("- results/peft_model_evaluation.json (Basic)")
        print("- results/improved_prompt_evaluation.json (Improved)")
        print("- results/optimized_prompt_evaluation.json (Optimized)")
        return
    
    # 비교 분석
    comparison = compare_three_prompts(results)
    
    # 결과 출력
    print_triple_comparison(comparison)
    
    # 결과 저장
    with open("results/triple_prompt_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 비교 결과가 results/triple_prompt_comparison.json에 저장되었습니다!")

if __name__ == "__main__":
    main()

