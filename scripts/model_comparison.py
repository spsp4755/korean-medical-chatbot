#!/usr/bin/env python3
"""
한국어 LLM 모델 비교 분석 스크립트
KULLM-2 vs Polyglot-Ko 모델의 성능, 크기, 속도를 비교합니다.
"""

import json
import time
import torch
from pathlib import Path
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil
import yaml

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """설정 파일을 로드합니다."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_model_info(model_name: str) -> Dict[str, Any]:
    """모델 정보를 가져옵니다."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        # 모델 크기 계산
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        return {
            "model_name": model_name,
            "vocab_size": tokenizer.vocab_size,
            "max_length": tokenizer.model_max_length,
            "parameter_count": param_count,
            "model_size_mb": model_size_mb,
            "model_size_gb": model_size_mb / 1024,
            "tokenizer_type": type(tokenizer).__name__,
            "model_type": type(model).__name__
        }
    except Exception as e:
        return {
            "model_name": model_name,
            "error": str(e)
        }

def benchmark_inference_speed(model_name: str, test_prompts: List[str], num_runs: int = 3) -> Dict[str, Any]:
    """추론 속도를 벤치마크합니다."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        times = []
        tokens_generated = []
        
        for _ in range(num_runs):
            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                end_time = time.time()
                
                generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                tokens_generated.append(len(generated_tokens))
                times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        tokens_per_second = avg_tokens / avg_time if avg_time > 0 else 0
        
        return {
            "avg_inference_time": avg_time,
            "avg_tokens_generated": avg_tokens,
            "tokens_per_second": tokens_per_second,
            "total_runs": len(times)
        }
    except Exception as e:
        return {
            "error": str(e)
        }

def test_korean_understanding(model_name: str, test_cases: List[Dict[str, str]]) -> Dict[str, Any]:
    """한국어 이해 능력을 테스트합니다."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        results = []
        
        for test_case in test_cases:
            prompt = test_case["prompt"]
            expected_keywords = test_case.get("expected_keywords", [])
            
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            # 키워드 매칭 점수 계산
            keyword_score = 0
            if expected_keywords:
                for keyword in expected_keywords:
                    if keyword.lower() in response.lower():
                        keyword_score += 1
                keyword_score = keyword_score / len(expected_keywords)
            
            results.append({
                "prompt": prompt,
                "response": response,
                "expected_keywords": expected_keywords,
                "keyword_score": keyword_score
            })
        
        avg_keyword_score = sum(r["keyword_score"] for r in results) / len(results)
        
        return {
            "test_results": results,
            "avg_keyword_score": avg_keyword_score,
            "total_tests": len(results)
        }
    except Exception as e:
        return {
            "error": str(e)
        }

def main():
    """메인 함수"""
    print("🚀 한국어 LLM 모델 비교 분석 시작")
    
    # 설정 로드
    config = load_config()
    
    # 비교할 모델들 (실제 존재하는 한국어 LLM 모델)
    models = [
        "psymon/KoLlama2-7b",  # 한국어 최적화 Llama2 모델
        "beomi/KcBERT-base"    # 한국어 BERT 모델 (비교용)
    ]
    
    # 테스트 프롬프트
    test_prompts = [
        "안녕하세요. 오늘 기분이 어떠신가요?",
        "의료 상담을 받고 싶습니다.",
        "감기 증상이 있는데 어떻게 해야 할까요?",
        "한국의 전통 음식에 대해 설명해주세요."
    ]
    
    # 한국어 이해 테스트 케이스
    korean_test_cases = [
        {
            "prompt": "의료 상담: 두통이 심하고 메스꺼움이 있습니다. 어떤 질병일 가능성이 있나요?",
            "expected_keywords": ["두통", "메스꺼움", "의사", "진료", "병원"]
        },
        {
            "prompt": "한국어 문법 질문: '나는 학교에 간다'와 '나는 학교에 가고 있다'의 차이점은 무엇인가요?",
            "expected_keywords": ["현재", "진행", "시제", "문법"]
        },
        {
            "prompt": "수학 문제: 2x + 5 = 13일 때 x의 값은?",
            "expected_keywords": ["4", "방정식", "해", "계산"]
        }
    ]
    
    results = {}
    
    for model_name in models:
        print(f"\n📊 {model_name} 분석 중...")
        
        # 1. 모델 정보 수집
        print("  - 모델 정보 수집 중...")
        model_info = get_model_info(model_name)
        results[model_name] = {"model_info": model_info}
        
        if "error" in model_info:
            print(f"  ❌ 모델 로드 실패: {model_info['error']}")
            continue
        
        print(f"  ✅ 모델 크기: {model_info['model_size_gb']:.2f}GB")
        print(f"  ✅ 파라미터 수: {model_info['parameter_count']:,}")
        
        # 2. 추론 속도 벤치마크
        print("  - 추론 속도 벤치마크 중...")
        speed_results = benchmark_inference_speed(model_name, test_prompts)
        results[model_name]["speed_benchmark"] = speed_results
        
        if "error" not in speed_results:
            print(f"  ✅ 평균 추론 시간: {speed_results['avg_inference_time']:.2f}초")
            print(f"  ✅ 토큰/초: {speed_results['tokens_per_second']:.2f}")
        
        # 3. 한국어 이해 테스트
        print("  - 한국어 이해 테스트 중...")
        korean_results = test_korean_understanding(model_name, korean_test_cases)
        results[model_name]["korean_understanding"] = korean_results
        
        if "error" not in korean_results:
            print(f"  ✅ 한국어 이해 점수: {korean_results['avg_keyword_score']:.2f}")
    
    # 결과 저장
    output_path = Path("outputs/model_comparison_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📋 결과가 {output_path}에 저장되었습니다.")
    
    # 간단한 비교 요약
    print("\n📊 모델 비교 요약:")
    for model_name, result in results.items():
        if "error" not in result.get("model_info", {}):
            info = result["model_info"]
            speed = result.get("speed_benchmark", {})
            korean = result.get("korean_understanding", {})
            
            print(f"\n🔹 {model_name}:")
            print(f"  - 크기: {info['model_size_gb']:.2f}GB")
            print(f"  - 파라미터: {info['parameter_count']:,}")
            if "tokens_per_second" in speed:
                print(f"  - 속도: {speed['tokens_per_second']:.2f} 토큰/초")
            if "avg_keyword_score" in korean:
                print(f"  - 한국어 점수: {korean['avg_keyword_score']:.2f}")

if __name__ == "__main__":
    main()
