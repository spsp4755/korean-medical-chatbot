"""
INT8 양자화 모듈
모델을 INT8로 양자화하여 크기와 추론 속도를 개선합니다.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
import time
from typing import Dict, Any, Optional
import psutil
import os

class ModelQuantizer:
    """모델 양자화 클래스"""
    
    def __init__(self, model_name: str, output_dir: str = "models/quantized"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 원본 모델과 토크나이저
        self.original_model = None
        self.original_tokenizer = None
        self.quantized_model = None
        
        # 성능 측정 결과
        self.quantization_results = {}
    
    def load_original_model(self) -> Dict[str, Any]:
        """원본 모델 로드 및 성능 측정"""
        print(f"🔄 원본 모델 로딩 중: {self.model_name}")
        
        start_time = time.time()
        
        # 토크나이저 로드
        self.original_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.original_tokenizer.pad_token is None:
            self.original_tokenizer.pad_token = self.original_tokenizer.eos_token
        
        # 모델 로드 (float32로 로드하여 압축 효과 확인)
        self.original_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # float32로 로드
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        load_time = time.time() - start_time
        
        # 모델 크기 측정
        model_size = self._get_model_size(self.original_model)
        
        # 메모리 사용량 측정
        memory_usage = self._get_memory_usage()
        
        # 추론 속도 측정
        inference_speed = self._benchmark_inference(self.original_model, self.original_tokenizer)
        
        results = {
            "model_name": self.model_name,
            "load_time": load_time,
            "model_size_mb": model_size,
            "memory_usage_mb": memory_usage,
            "inference_speed_seconds": inference_speed,
            "device": str(next(self.original_model.parameters()).device)
        }
        
        print(f"✅ 원본 모델 로딩 완료: {load_time:.2f}초")
        print(f"📊 모델 크기: {model_size:.2f}MB")
        print(f"🧠 추론 속도: {inference_speed:.2f}초")
        
        return results
    
    def quantize_model(self, method: str = "int8") -> Dict[str, Any]:
        """모델 양자화"""
        print(f"🔧 {method.upper()} 양자화 시작...")
        
        if method == "int8":
            return self._quantize_int8()
        elif method == "int8_linear":
            return self._quantize_int8_linear_only()
        elif method == "int4":
            return self._quantize_int4()
        elif method == "compression":
            return self._compress_model()
        else:
            raise ValueError(f"지원하지 않는 양자화 방법: {method}")
    
    def _quantize_int8(self) -> Dict[str, Any]:
        """INT8 양자화 구현"""
        try:
            # PyTorch의 동적 양자화 사용
            print("  - 동적 양자화 적용 중...")
            
            # 모델을 평가 모드로 설정
            self.original_model.eval()
            
            # 동적 양자화 적용 (Embedding 제외)
            quantized_model = torch.quantization.quantize_dynamic(
                self.original_model,
                {nn.Linear, nn.LayerNorm},  # Embedding 제외
                dtype=torch.qint8
            )
            
            self.quantized_model = quantized_model
            
            # 양자화된 모델 성능 측정
            quantized_size = self._get_model_size(quantized_model)
            quantized_speed = self._benchmark_inference(quantized_model, self.original_tokenizer)
            quantized_memory = self._get_memory_usage()
            
            results = {
                "quantization_method": "int8_dynamic",
                "quantized_size_mb": quantized_size,
                "quantized_speed_seconds": quantized_speed,
                "quantized_memory_mb": quantized_memory,
                "size_reduction_percent": (self.quantization_results["original"]["model_size_mb"] - quantized_size) / self.quantization_results["original"]["model_size_mb"] * 100,
                "speed_improvement_percent": (self.quantization_results["original"]["inference_speed_seconds"] - quantized_speed) / self.quantization_results["original"]["inference_speed_seconds"] * 100
            }
            
            print(f"✅ INT8 양자화 완료")
            print(f"📊 양자화된 크기: {quantized_size:.2f}MB")
            print(f"🧠 양자화된 속도: {quantized_speed:.2f}초")
            print(f"📉 크기 감소: {results['size_reduction_percent']:.1f}%")
            print(f"⚡ 속도 향상: {results['speed_improvement_percent']:.1f}%")
            
            return results
            
        except Exception as e:
            print(f"❌ INT8 양자화 실패: {str(e)}")
            return {"error": str(e)}
    
    def _quantize_int8_linear_only(self) -> Dict[str, Any]:
        """INT8 양자화 구현 (Linear 레이어만)"""
        try:
            print("  - Linear 레이어만 양자화 적용 중...")
            
            # 모델을 평가 모드로 설정
            self.original_model.eval()
            
            # CPU에서 양자화 엔진 설정
            torch.backends.quantized.engine = 'qnnpack'
            
            # Linear 레이어만 양자화
            quantized_model = torch.quantization.quantize_dynamic(
                self.original_model,
                {nn.Linear},  # Linear 레이어만
                dtype=torch.qint8
            )
            
            self.quantized_model = quantized_model
            
            # 양자화된 모델 성능 측정
            quantized_size = self._get_model_size(quantized_model)
            quantized_speed = self._benchmark_inference(quantized_model, self.original_tokenizer)
            quantized_memory = self._get_memory_usage()
            
            results = {
                "quantization_method": "int8_linear_only",
                "quantized_size_mb": quantized_size,
                "quantized_speed_seconds": quantized_speed,
                "quantized_memory_mb": quantized_memory,
                "size_reduction_percent": (self.quantization_results["original"]["model_size_mb"] - quantized_size) / self.quantization_results["original"]["model_size_mb"] * 100,
                "speed_improvement_percent": (self.quantization_results["original"]["inference_speed_seconds"] - quantized_speed) / self.quantization_results["original"]["inference_speed_seconds"] * 100
            }
            
            print(f"✅ INT8 Linear 양자화 완료")
            print(f"📊 양자화된 크기: {quantized_size:.2f}MB")
            print(f"🧠 양자화된 속도: {quantized_speed:.2f}초")
            print(f"📉 크기 감소: {results['size_reduction_percent']:.1f}%")
            print(f"⚡ 속도 향상: {results['speed_improvement_percent']:.1f}%")
            
            return results
            
        except Exception as e:
            print(f"❌ INT8 Linear 양자화 실패: {str(e)}")
            return {"error": str(e)}
    
    def _compress_model(self) -> Dict[str, Any]:
        """모델 압축 (가중치 정밀도 감소)"""
        try:
            print("  - 모델 압축 적용 중...")
            
            # 모델을 평가 모드로 설정
            self.original_model.eval()
            
            # 원본 모델의 데이터 타입 확인
            original_dtype = next(self.original_model.parameters()).dtype
            print(f"  - 원본 모델 데이터 타입: {original_dtype}")
            
            # 모델을 float16으로 변환 (압축)
            compressed_model = self.original_model.half()
            
            # 압축된 모델의 데이터 타입 확인
            compressed_dtype = next(compressed_model.parameters()).dtype
            print(f"  - 압축된 모델 데이터 타입: {compressed_dtype}")
            
            self.quantized_model = compressed_model
            
            # 압축된 모델 성능 측정
            compressed_speed = self._benchmark_inference(compressed_model, self.original_tokenizer)
            compressed_memory = self._get_memory_usage()
            
            # 이론적 크기 계산
            original_size = self._get_theoretical_size(self.original_model, torch.float32)
            compressed_size = self._get_theoretical_size(compressed_model, torch.float16)
            
            results = {
                "quantization_method": "float16_compression",
                "original_dtype": str(original_dtype),
                "compressed_dtype": str(compressed_dtype),
                "original_size_mb": original_size,
                "quantized_size_mb": compressed_size,
                "quantized_speed_seconds": compressed_speed,
                "quantized_memory_mb": compressed_memory,
                "size_reduction_percent": (original_size - compressed_size) / original_size * 100,
                "speed_improvement_percent": (self.quantization_results["original"]["inference_speed_seconds"] - compressed_speed) / self.quantization_results["original"]["inference_speed_seconds"] * 100
            }
            
            print(f"✅ 모델 압축 완료")
            print(f"📊 원본 크기: {original_size:.2f}MB")
            print(f"📊 압축된 크기: {compressed_size:.2f}MB")
            print(f"🧠 압축된 속도: {compressed_speed:.2f}초")
            print(f"📉 크기 감소: {results['size_reduction_percent']:.1f}%")
            print(f"⚡ 속도 향상: {results['speed_improvement_percent']:.1f}%")
            
            return results
            
        except Exception as e:
            print(f"❌ 모델 압축 실패: {str(e)}")
            return {"error": str(e)}
    
    def _quantize_int4(self) -> Dict[str, Any]:
        """INT4 양자화 구현 (bitsandbytes 사용)"""
        try:
            print("  - INT4 양자화 적용 중...")
            
            # bitsandbytes를 사용한 INT4 양자화
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            quantized_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            
            self.quantized_model = quantized_model
            
            # 성능 측정
            quantized_size = self._get_model_size(quantized_model)
            quantized_speed = self._benchmark_inference(quantized_model, self.original_tokenizer)
            quantized_memory = self._get_memory_usage()
            
            results = {
                "quantization_method": "int4_bitsandbytes",
                "quantized_size_mb": quantized_size,
                "quantized_speed_seconds": quantized_speed,
                "quantized_memory_mb": quantized_memory,
                "size_reduction_percent": (self.quantization_results["original"]["model_size_mb"] - quantized_size) / self.quantization_results["original"]["model_size_mb"] * 100,
                "speed_improvement_percent": (self.quantization_results["original"]["inference_speed_seconds"] - quantized_speed) / self.quantization_results["original"]["inference_speed_seconds"] * 100
            }
            
            print(f"✅ INT4 양자화 완료")
            print(f"📊 양자화된 크기: {quantized_size:.2f}MB")
            print(f"🧠 양자화된 속도: {quantized_speed:.2f}초")
            print(f"📉 크기 감소: {results['size_reduction_percent']:.1f}%")
            print(f"⚡ 속도 향상: {results['speed_improvement_percent']:.1f}%")
            
            return results
            
        except Exception as e:
            print(f"❌ INT4 양자화 실패: {str(e)}")
            return {"error": str(e)}
    
    def _get_model_size(self, model) -> float:
        """모델 크기 측정 (MB)"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    def _get_model_size_by_dtype(self, model) -> float:
        """데이터 타입별 모델 크기 측정 (MB) - 이론적 계산"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            # 파라미터 수 * 데이터 타입 크기 (이론적)
            if param.dtype == torch.float32:
                param_size += param.nelement() * 4  # float32 = 4 bytes
            elif param.dtype == torch.float16:
                param_size += param.nelement() * 2  # float16 = 2 bytes
            elif param.dtype == torch.int8:
                param_size += param.nelement() * 1  # int8 = 1 byte
            else:
                param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            if buffer.dtype == torch.float32:
                buffer_size += buffer.nelement() * 4
            elif buffer.dtype == torch.float16:
                buffer_size += buffer.nelement() * 2
            else:
                buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    def _get_theoretical_size(self, model, target_dtype) -> float:
        """특정 데이터 타입으로 변환했을 때의 이론적 크기 (MB)"""
        param_size = 0
        buffer_size = 0
        
        # 타입별 바이트 크기
        dtype_bytes = {
            torch.float32: 4,
            torch.float16: 2,
            torch.int8: 1,
            torch.int32: 4
        }
        
        target_bytes = dtype_bytes.get(target_dtype, 4)
        
        for param in model.parameters():
            param_size += param.nelement() * target_bytes
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * target_bytes
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    def _get_memory_usage(self) -> float:
        """메모리 사용량 측정 (MB)"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        else:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024**2
    
    def _benchmark_inference(self, model, tokenizer, num_runs: int = 3) -> float:
        """추론 속도 벤치마크"""
        test_prompts = [
            "안녕하세요.",
            "의료 상담을 받고 싶습니다.",
            "감기 증상이 있습니다."
        ]
        
        times = []
        
        for _ in range(num_runs):
            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                end_time = time.time()
                
                times.append(end_time - start_time)
        
        return sum(times) / len(times)
    
    def save_quantized_model(self, model_name: str = "quantized_model"):
        """양자화된 모델 저장"""
        if self.quantized_model is None:
            print("❌ 양자화된 모델이 없습니다.")
            return
        
        save_path = self.output_dir / model_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 양자화된 모델 저장 중: {save_path}")
        
        # 모델과 토크나이저 저장
        self.quantized_model.save_pretrained(save_path)
        self.original_tokenizer.save_pretrained(save_path)
        
        # 양자화 결과 저장
        results_path = save_path / "quantization_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.quantization_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 모델 저장 완료: {save_path}")
    
    def run_quantization_experiment(self, methods: list = ["int8"]) -> Dict[str, Any]:
        """양자화 실험 실행"""
        print("🚀 양자화 실험 시작")
        print("=" * 60)
        
        # 원본 모델 성능 측정
        original_results = self.load_original_model()
        self.quantization_results["original"] = original_results
        
        # 각 양자화 방법별 실험
        for method in methods:
            print(f"\n{'='*50}")
            print(f"🔧 {method.upper()} 양자화 실험")
            print(f"{'='*50}")
            
            quantized_results = self.quantize_model(method)
            self.quantization_results[method] = quantized_results
            
            # 양자화된 모델 저장
            if "error" not in quantized_results:
                self.save_quantized_model(f"model_{method}")
        
        # 전체 결과 저장
        results_path = self.output_dir / "quantization_experiment_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.quantization_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📋 실험 결과가 {results_path}에 저장되었습니다.")
        
        return self.quantization_results

def main():
    """메인 함수"""
    # 42dot 모델로 양자화 실험
    quantizer = ModelQuantizer("42dot/42dot_LLM-SFT-1.3B")
    
    # INT8 양자화 실험 실행
    results = quantizer.run_quantization_experiment(["int8"])
    
    # 결과 요약 출력
    print("\n📊 양자화 실험 결과 요약:")
    print("=" * 60)
    
    if "original" in results:
        orig = results["original"]
        print(f"원본 모델: {orig['model_size_mb']:.2f}MB, {orig['inference_speed_seconds']:.2f}초")
    
    if "int8" in results and "error" not in results["int8"]:
        int8 = results["int8"]
        print(f"INT8 모델: {int8['quantized_size_mb']:.2f}MB, {int8['quantized_speed_seconds']:.2f}초")
        print(f"크기 감소: {int8['size_reduction_percent']:.1f}%")
        print(f"속도 향상: {int8['speed_improvement_percent']:.1f}%")

if __name__ == "__main__":
    main()
