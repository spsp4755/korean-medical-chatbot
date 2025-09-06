"""
Structured Pruning 모듈
가중치 20-30% 제거를 통한 모델 경량화
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, Any, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import psutil
import os
from pathlib import Path
import json


class ModelPruner:
    """Structured Pruning을 통한 모델 경량화"""
    
    def __init__(self, model_name: str, output_dir: str = "models/pruned"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델과 토크나이저 로드
        print(f"🔄 모델 로딩 중: {model_name}")
        self.original_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        self.original_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 패딩 토큰 설정
        if self.original_tokenizer.pad_token is None:
            self.original_tokenizer.pad_token = self.original_tokenizer.eos_token
        
        self.pruned_model = None
        self.pruning_results = {}
        
        print(f"✅ 모델 로딩 완료")
    
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
    
    def _benchmark_inference(self, model, tokenizer, num_runs: int = 3) -> float:
        """추론 속도 벤치마크 (초)"""
        model.eval()
        test_text = "안녕하세요. 의료 상담을 받고 싶습니다."
        
        # 토크나이징
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
        
        # 워밍업
        with torch.no_grad():
            for _ in range(2):
                _ = model.generate(
                    inputs.input_ids,
                    max_length=inputs.input_ids.shape[1] + 20,
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
        
        # 실제 측정
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model.generate(
                    inputs.input_ids,
                    max_length=inputs.input_ids.shape[1] + 20,
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        return avg_time
    
    def _get_memory_usage(self) -> float:
        """메모리 사용량 측정 (MB)"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        else:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024**2
    
    def _apply_structured_pruning(self, sparsity: float) -> nn.Module:
        """Structured Pruning 적용"""
        print(f"  - {sparsity*100:.1f}% 가중치 제거 중...")
        
        # 모델 복사
        pruned_model = self.original_model
        
        # Pruning할 레이어 식별
        layers_to_prune = []
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                layers_to_prune.append((module, 'weight'))
        
        print(f"  - {len(layers_to_prune)}개 Linear 레이어 발견")
        
        # Structured Pruning 적용
        for module, param_name in layers_to_prune:
            # Magnitude-based pruning으로 중요하지 않은 뉴런 식별
            prune.ln_structured(
                module, 
                name=param_name, 
                amount=sparsity, 
                n=2,  # L2 norm
                dim=0  # 출력 차원에서 제거
            )
        
        # Pruning 마스크 적용
        for module, param_name in layers_to_prune:
            prune.remove(module, param_name)
        
        return pruned_model
    
    def _evaluate_pruning_quality(self, original_model, pruned_model, tokenizer) -> Dict[str, float]:
        """Pruning 품질 평가"""
        print("  - Pruning 품질 평가 중...")
        
        # 테스트 텍스트들
        test_texts = [
            "안녕하세요. 의료 상담을 받고 싶습니다.",
            "머리가 아픈데 어떻게 해야 할까요?",
            "감기 증상이 있는데 병원에 가야 할까요?",
            "복통이 심한데 응급실에 가야 할까요?",
            "피부에 발진이 생겼는데 원인이 뭘까요?"
        ]
        
        quality_scores = []
        
        for text in test_texts:
            # 원본 모델 추론
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                original_output = original_model.generate(
                    inputs.input_ids,
                    max_length=inputs.input_ids.shape[1] + 30,
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                pruned_output = pruned_model.generate(
                    inputs.input_ids,
                    max_length=inputs.input_ids.shape[1] + 30,
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # 출력 비교 (간단한 유사도 측정)
            original_text = tokenizer.decode(original_output[0], skip_special_tokens=True)
            pruned_text = tokenizer.decode(pruned_output[0], skip_special_tokens=True)
            
            # 단어 수 기반 유사도 (간단한 메트릭)
            original_words = set(original_text.split())
            pruned_words = set(pruned_text.split())
            
            if len(original_words) > 0:
                similarity = len(original_words.intersection(pruned_words)) / len(original_words)
                quality_scores.append(similarity)
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        return {"quality_score": avg_quality, "num_tests": len(test_texts)}
    
    def run_pruning_experiment(self, sparsity_levels: List[float] = [0.1, 0.2, 0.3]) -> Dict[str, Any]:
        """Pruning 실험 실행"""
        print("🚀 Structured Pruning 실험 시작")
        print("=" * 50)
        
        # 원본 모델 성능 측정
        print("📊 원본 모델 성능 측정 중...")
        original_size = self._get_model_size(self.original_model)
        original_speed = self._benchmark_inference(self.original_model, self.original_tokenizer)
        original_memory = self._get_memory_usage()
        
        print(f"📊 원본 크기: {original_size:.2f}MB")
        print(f"🧠 원본 속도: {original_speed:.2f}초")
        print(f"💾 원본 메모리: {original_memory:.2f}MB")
        
        results = {
            "original_model": {
                "size_mb": original_size,
                "inference_speed_sec": original_speed,
                "memory_usage_mb": original_memory
            },
            "pruning_experiments": []
        }
        
        # 각 sparsity level에 대해 실험
        for sparsity in sparsity_levels:
            print(f"\n{'='*50}")
            print(f"🔧 {sparsity*100:.1f}% Pruning 실험")
            print(f"{'='*50}")
            
            try:
                # Pruning 적용
                pruned_model = self._apply_structured_pruning(sparsity)
                
                # 성능 측정
                pruned_size = self._get_model_size(pruned_model)
                pruned_speed = self._benchmark_inference(pruned_model, self.original_tokenizer)
                pruned_memory = self._get_memory_usage()
                
                # 품질 평가
                quality_results = self._evaluate_pruning_quality(
                    self.original_model, pruned_model, self.original_tokenizer
                )
                
                # 결과 계산
                size_reduction = ((original_size - pruned_size) / original_size) * 100
                speed_improvement = ((original_speed - pruned_speed) / original_speed) * 100
                memory_reduction = ((original_memory - pruned_memory) / original_memory) * 100
                
                experiment_result = {
                    "sparsity": sparsity,
                    "size_mb": pruned_size,
                    "inference_speed_sec": pruned_speed,
                    "memory_usage_mb": pruned_memory,
                    "size_reduction_percent": size_reduction,
                    "speed_improvement_percent": speed_improvement,
                    "memory_reduction_percent": memory_reduction,
                    "quality_score": quality_results["quality_score"],
                    "num_quality_tests": quality_results["num_tests"]
                }
                
                results["pruning_experiments"].append(experiment_result)
                
                print(f"✅ {sparsity*100:.1f}% Pruning 완료")
                print(f"📊 크기: {pruned_size:.2f}MB ({size_reduction:.1f}% 감소)")
                print(f"🧠 속도: {pruned_speed:.2f}초 ({speed_improvement:.1f}% 향상)")
                print(f"💾 메모리: {pruned_memory:.2f}MB ({memory_reduction:.1f}% 감소)")
                print(f"🎯 품질 점수: {quality_results['quality_score']:.3f}")
                
                # 최적 모델 저장 (20% pruning)
                if sparsity == 0.2:
                    self.pruned_model = pruned_model
                    model_path = self.output_dir / "model_20_percent_pruned"
                    print(f"💾 최적 모델 저장 중: {model_path}")
                    pruned_model.save_pretrained(model_path)
                    self.original_tokenizer.save_pretrained(model_path)
                    print(f"✅ 모델 저장 완료: {model_path}")
                
            except Exception as e:
                print(f"❌ {sparsity*100:.1f}% Pruning 실패: {str(e)}")
                results["pruning_experiments"].append({
                    "sparsity": sparsity,
                    "error": str(e)
                })
        
        # 결과 저장
        results_path = self.output_dir / "pruning_experiment_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 실험 결과가 {results_path}에 저장되었습니다.")
        
        return results
    
    def analyze_results(self, results: Dict[str, Any]) -> None:
        """실험 결과 분석 및 권장사항 제시"""
        print("\n📊 Pruning 실험 결과 분석:")
        print("=" * 60)
        
        original = results["original_model"]
        print(f"🔹 원본 모델:")
        print(f"  - 크기: {original['size_mb']:.2f}MB")
        print(f"  - 추론 속도: {original['inference_speed_sec']:.2f}초")
        print(f"  - 메모리 사용량: {original['memory_usage_mb']:.2f}MB")
        
        print(f"\n🔹 Pruning 실험 결과:")
        for exp in results["pruning_experiments"]:
            if "error" not in exp:
                print(f"  - {exp['sparsity']*100:.1f}% 제거:")
                print(f"    크기: {exp['size_mb']:.2f}MB ({exp['size_reduction_percent']:.1f}% 감소)")
                print(f"    속도: {exp['inference_speed_sec']:.2f}초 ({exp['speed_improvement_percent']:.1f}% 향상)")
                print(f"    메모리: {exp['memory_usage_mb']:.2f}MB ({exp['memory_reduction_percent']:.1f}% 감소)")
                print(f"    품질: {exp['quality_score']:.3f}")
        
        # 최적 sparsity 추천
        best_experiment = None
        best_score = -1
        
        for exp in results["pruning_experiments"]:
            if "error" not in exp:
                # 종합 점수 계산 (크기 감소 + 속도 향상 + 품질 보존)
                score = (exp['size_reduction_percent'] * 0.3 + 
                        exp['speed_improvement_percent'] * 0.3 + 
                        exp['quality_score'] * 100 * 0.4)
                
                if score > best_score:
                    best_score = score
                    best_experiment = exp
        
        if best_experiment:
            print(f"\n🏆 최적 Pruning 설정: {best_experiment['sparsity']*100:.1f}% 제거")
            print(f"   종합 점수: {best_score:.2f}")
        
        print(f"\n🚀 다음 단계:")
        print("1. Knowledge Distillation 구현")
        print("2. LoRA 파인튜닝 적용")
        print("3. 의료 도메인 파인튜닝")
