#!/usr/bin/env python3
"""
빠른 모델 테스트 스크립트
모델 로드 및 기본 추론 테스트를 수행합니다.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def test_model_loading(model_name: str) -> dict:
    """모델 로딩 테스트"""
    print(f"🔄 {model_name} 로딩 테스트 중...")
    
    try:
        start_time = time.time()
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        load_time = time.time() - start_time
        
        # 메모리 사용량 확인
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
        else:
            memory_used = 0
        
        return {
            "success": True,
            "load_time": load_time,
            "memory_used_gb": memory_used,
            "device": str(next(model.parameters()).device)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def test_simple_inference(model_name: str) -> dict:
    """간단한 추론 테스트"""
    print(f"🧠 {model_name} 추론 테스트 중...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 테스트 프롬프트
        test_prompts = [
            "안녕하세요.",
            "의료 상담을 받고 싶습니다.",
            "감기 증상이 있습니다."
        ]
        
        results = []
        
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # polyglot-ko 모델의 경우 token_type_ids 제거
            if "polyglot" in model_name.lower():
                inputs.pop("token_type_ids", None)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            inference_time = time.time() - start_time
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            results.append({
                "prompt": prompt,
                "response": response,
                "inference_time": inference_time
            })
        
        return {
            "success": True,
            "results": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """메인 함수"""
    print("🚀 빠른 모델 테스트 시작")
    
    # 테스트할 모델들 (베이스라인 후보 모델들)
    models = [
        "skt/kogpt2-base-v2",           # GPT-2 기반 (124M, ~500MB) ✅
        "42dot/42dot_LLM-SFT-1.3B"      # SFT 적용 (1.3B, ~2.6GB) ⭐
    ]
    
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"📊 {model_name} 테스트")
        print(f"{'='*50}")
        
        # 1. 모델 로딩 테스트
        load_result = test_model_loading(model_name)
        
        if load_result["success"]:
            print(f"✅ 로딩 성공: {load_result['load_time']:.2f}초")
            print(f"✅ 메모리 사용량: {load_result['memory_used_gb']:.2f}GB")
            print(f"✅ 디바이스: {load_result['device']}")
            
            # 2. 추론 테스트
            inference_result = test_simple_inference(model_name)
            
            if inference_result["success"]:
                print(f"✅ 추론 테스트 성공")
                for result in inference_result["results"]:
                    print(f"  Q: {result['prompt']}")
                    print(f"  A: {result['response']}")
                    print(f"  시간: {result['inference_time']:.2f}초")
                    print()
            else:
                print(f"❌ 추론 테스트 실패: {inference_result['error']}")
        else:
            print(f"❌ 모델 로딩 실패: {load_result['error']}")
        
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
