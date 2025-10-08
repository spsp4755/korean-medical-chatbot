import json
import os
from datetime import datetime

def load_basic_metrics():
    """기본 프롬프트 성능 메트릭 로드 (peft_medical_evaluation.py 결과)"""
    with open("results/peft_model_evaluation.json", "r", encoding="utf-8") as f:
        return json.load(f)

def load_improved_metrics():
    """개선된 프롬프트 성능 메트릭 로드"""
    with open("results/improved_prompt_evaluation.json", "r", encoding="utf-8") as f:
        return json.load(f)

def compare_performance():
    """성능 비교 분석"""
    print("🔍 성능 비교 분석 시작...")
    
    # 메트릭 로드
    basic = load_basic_metrics()
    improved = load_improved_metrics()
    
    # 비교 결과 생성
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "basic_metrics": basic,
        "improved_metrics": improved,
        "comparison_analysis": {}
    }
    
    # 메트릭 추출
    basic_bertscore = basic["bertscore"]
    basic_medical = basic["medical_metrics"]
    improved_bertscore = improved["bertscore"]
    improved_medical = improved["medical_metrics"]
    
    # 비교 분석
    analysis = comparison["comparison_analysis"]
    
    # BERTScore 비교
    basic_f1 = basic_bertscore["bertscore_f1"]
    improved_f1 = improved_bertscore["bertscore_f1"]
    analysis["bertscore_f1"] = {
        "basic": f"{basic_f1:.4f}",
        "improved": f"{improved_f1:.4f}",
        "improvement": f"{((improved_f1 - basic_f1) / basic_f1 * 100):+.1f}%",
        "status": "✅ 개선됨" if improved_f1 > basic_f1 else "❌ 악화됨"
    }
    
    # 응답 길이 비교
    basic_length = basic_medical["avg_response_length"]
    improved_length = improved_medical["avg_response_length"]
    analysis["response_length"] = {
        "basic": f"{basic_length:.1f}자",
        "improved": f"{improved_length:.1f}자",
        "improvement": f"{((improved_length - basic_length) / basic_length * 100):+.1f}%",
        "status": "✅ 개선됨" if improved_length < basic_length else "❌ 악화됨"
    }
    
    # 의료 용어 포함도 비교
    basic_medical_coverage = basic_medical["medical_coverage"]
    improved_medical_coverage = improved_medical["medical_coverage"]
    analysis["medical_coverage"] = {
        "basic": f"{basic_medical_coverage:.4f}",
        "improved": f"{improved_medical_coverage:.4f}",
        "improvement": f"{((improved_medical_coverage - basic_medical_coverage) / basic_medical_coverage * 100):+.1f}%",
        "status": "✅ 개선됨" if improved_medical_coverage > basic_medical_coverage else "❌ 악화됨"
    }
    
    # 안전성 점수 비교
    basic_safety = basic_medical["safety_score"]
    improved_safety = improved_medical["safety_score"]
    analysis["safety_score"] = {
        "basic": f"{basic_safety:.4f}",
        "improved": f"{improved_safety:.4f}",
        "improvement": f"{((improved_safety - basic_safety) / basic_safety * 100):+.1f}%",
        "status": "✅ 개선됨" if improved_safety > basic_safety else "❌ 악화됨"
    }
    
    # 응답 품질 비교
    basic_quality = basic_medical["response_quality"]
    improved_quality = improved_medical["response_quality"]
    analysis["response_quality"] = {
        "basic": f"{basic_quality:.4f}",
        "improved": f"{improved_quality:.4f}",
        "improvement": f"{((improved_quality - basic_quality) / basic_quality * 100):+.1f}%",
        "status": "✅ 개선됨" if improved_quality > basic_quality else "❌ 악화됨"
    }
    
    # 전체 개선도 계산
    improvements = [
        improved_f1 > basic_f1,  # BERTScore F1 개선
        improved_length < basic_length,  # 응답 길이 단축
        improved_medical_coverage > basic_medical_coverage,  # 의료 용어 포함도 개선
        improved_safety > basic_safety,  # 안전성 점수 개선
        improved_quality > basic_quality  # 응답 품질 개선
    ]
    
    improvement_count = sum(improvements)
    total_criteria = len(improvements)
    
    analysis["overall_improvement"] = {
        "improved_criteria": improvement_count,
        "total_criteria": total_criteria,
        "improvement_rate": f"{(improvement_count / total_criteria * 100):.1f}%",
        "status": "✅ 대폭 개선됨" if improvement_count >= 4 else "⚠️ 부분 개선됨" if improvement_count >= 2 else "❌ 개선 필요"
    }
    
    return comparison

def print_comparison_results(comparison):
    """비교 결과 출력"""
    print("\n" + "="*80)
    print("📊 프롬프트 개선 성능 비교 결과")
    print("="*80)
    
    analysis = comparison["comparison_analysis"]
    
    print(f"\n🔍 응답 길이:")
    print(f"  기본: {analysis['response_length']['basic']}")
    print(f"  개선: {analysis['response_length']['improved']}")
    print(f"  개선도: {analysis['response_length']['improvement']} {analysis['response_length']['status']}")
    
    print(f"\n🏥 의료 용어 포함도:")
    print(f"  기본: {analysis['medical_coverage']['basic']}")
    print(f"  개선: {analysis['medical_coverage']['improved']}")
    print(f"  개선도: {analysis['medical_coverage']['improvement']} {analysis['medical_coverage']['status']}")
    
    print(f"\n🛡️ 안전성 점수:")
    print(f"  기본: {analysis['safety_score']['basic']}")
    print(f"  개선: {analysis['safety_score']['improved']}")
    print(f"  개선도: {analysis['safety_score']['improvement']} {analysis['safety_score']['status']}")
    
    print(f"\n⭐ 응답 품질:")
    print(f"  기본: {analysis['response_quality']['basic']}")
    print(f"  개선: {analysis['response_quality']['improved']}")
    print(f"  개선도: {analysis['response_quality']['improvement']} {analysis['response_quality']['status']}")
    
    print(f"\n🎯 전체 개선도:")
    overall = analysis['overall_improvement']
    print(f"  개선된 기준: {overall['improved_criteria']}/{overall['total_criteria']}")
    print(f"  개선률: {overall['improvement_rate']}")
    print(f"  상태: {overall['status']}")
    
    print("\n" + "="*80)

def main():
    print("🚀 성능 비교 분석 시작")
    
    # 성능 비교
    comparison = compare_performance()
    
    # 결과 출력
    print_comparison_results(comparison)
    
    # 결과 저장
    os.makedirs("results", exist_ok=True)
    with open("results/performance_comparison_analysis.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 비교 분석 결과가 results/performance_comparison_analysis.json에 저장되었습니다!")

if __name__ == "__main__":
    main()
