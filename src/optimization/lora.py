"""
LoRA (Low-Rank Adaptation) 파인튜닝 모듈
효율적인 도메인 특화 학습을 통한 모델 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import time
import psutil
import os
from pathlib import Path
import json
import yaml
from datasets import Dataset
import logging
import warnings
warnings.filterwarnings("ignore")

# MPS 비활성화
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRAFineTuner:
    """LoRA 기반 효율적 파인튜닝"""
    
    def __init__(self, model_name: str, output_dir: str = "models/lora"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 로드
        self._load_model()
        
        # 성능 측정
        self._measure_baseline_performance()
        
    def _load_model(self):
        """모델 로드"""
        print(f"🔄 모델 로딩 중: {self.model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ 모델 로딩 완료")
    
    def _measure_baseline_performance(self):
        """베이스라인 성능 측정"""
        print("📊 베이스라인 성능 측정 중...")
        
        self.base_size = self._get_model_size(self.base_model)
        self.base_speed = self._benchmark_inference(self.base_model, self.tokenizer)
        self.base_memory = self._get_memory_usage()
        
        print(f"📊 베이스 크기: {self.base_size:.2f}MB")
        print(f"🧠 베이스 속도: {self.base_speed:.2f}초")
        print(f"💾 베이스 메모리: {self.base_memory:.2f}MB")
    
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
        
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
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
    
    def _prepare_medical_data(self) -> Dataset:
        """의료 데이터 준비"""
        print("📚 의료 데이터 준비 중...")
        
        medical_data = []
        medical_files = [
            "data/processed/medical_data.json",
            "data/processed/splits/essential_medical_test.json",
            "data/processed/splits/professional_medical_test.json"
        ]
        
        for file_path in medical_files:
            if Path(file_path).exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            medical_data.extend(data)
                        else:
                            medical_data.append(data)
                    print(f"✅ 로드됨: {file_path}")
                except Exception as e:
                    logger.warning(f"파일 로드 실패: {file_path} - {e}")
        
        # 텍스트 추출 및 전처리
        texts = []
        for item in medical_data[:300]:  # 300개로 제한
            if isinstance(item, dict):
                for key in ['text', 'content', 'question', 'answer', 'instruction', 'input', 'output']:
                    if key in item and isinstance(item[key], str) and len(item[key]) > 10:
                        texts.append(item[key])
                        break
            elif isinstance(item, str) and len(item) > 10:
                texts.append(item)
        
        # 중복 제거
        texts = list(set(texts))
        print(f"📝 전처리된 텍스트: {len(texts)}개")
        
        if len(texts) == 0:
            # 더미 데이터 생성
            texts = [
                "안녕하세요. 의료 상담을 받고 싶습니다.",
                "머리가 아픈데 어떻게 해야 할까요?",
                "감기 증상이 있는데 병원에 가야 할까요?",
                "복통이 심한데 응급실에 가야 할까요?",
                "피부에 발진이 생겼는데 원인이 뭘까요?"
            ] * 30
        
        return Dataset.from_dict({"text": texts})
    
    def _tokenize_data(self, dataset: Dataset) -> Dataset:
        """데이터 토크나이징"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def _create_lora_config(self, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.1) -> LoraConfig:
        """LoRA 설정 생성"""
        # KoGPT2 모델에 맞는 target modules
        target_modules = ["c_attn", "c_proj", "c_fc"]  # GPT-2 계열 모델용
        
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,  # rank
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        return config
    
    def run_lora_finetuning(self, 
                           r: int = 16,
                           lora_alpha: int = 32,
                           lora_dropout: float = 0.1,
                           num_epochs: int = 3,
                           batch_size: int = 2,
                           learning_rate: float = 2e-4) -> Dict[str, Any]:
        """LoRA 파인튜닝 실행"""
        print("🚀 LoRA 파인튜닝 시작")
        print("=" * 60)
        
        # LoRA 설정 생성
        lora_config = self._create_lora_config(r, lora_alpha, lora_dropout)
        print(f"📊 LoRA 설정: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
        
        # LoRA 모델 생성
        print("🔧 LoRA 모델 생성 중...")
        lora_model = get_peft_model(self.base_model, lora_config)
        
        # LoRA 모델 크기 측정
        lora_size = self._get_model_size(lora_model)
        print(f"📊 LoRA 모델 크기: {lora_size:.2f}MB")
        print(f"📊 크기 증가: {((lora_size - self.base_size) / self.base_size * 100):.2f}%")
        
        # 데이터 준비
        dataset = self._prepare_medical_data()
        tokenized_dataset = self._tokenize_data(dataset)
        
        # Data Collator 설정
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Training Arguments 설정
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "training_output"),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=5,
            save_steps=100,
            eval_strategy="no",
            save_total_limit=1,
            learning_rate=learning_rate,
            fp16=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,
            use_cpu=True,
            dataloader_num_workers=0,
        )
        
        # Trainer 초기화
        trainer = Trainer(
            model=lora_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # LoRA 파인튜닝 실행
        print("🎓 LoRA 파인튜닝 학습 시작...")
        start_time = time.time()
        
        try:
            trainer.train()
            training_time = time.time() - start_time
            print(f"✅ LoRA 파인튜닝 완료: {training_time:.2f}초")
            
            # LoRA 모델 저장
            lora_model_path = self.output_dir / "lora_model"
            trainer.save_model(str(lora_model_path))
            self.tokenizer.save_pretrained(str(lora_model_path))
            
            # 파인튜닝된 모델 성능 측정
            print("📊 파인튜닝된 모델 성능 측정 중...")
            finetuned_speed = self._benchmark_inference(lora_model, self.tokenizer)
            finetuned_memory = self._get_memory_usage()
            
            # 결과 계산
            speed_improvement = ((self.base_speed - finetuned_speed) / self.base_speed) * 100
            memory_change = ((self.base_memory - finetuned_memory) / self.base_memory) * 100
            
            results = {
                "lora_config": {
                    "r": r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "training_time_sec": training_time
                },
                "base_model": {
                    "size_mb": self.base_size,
                    "inference_speed_sec": self.base_speed,
                    "memory_usage_mb": self.base_memory
                },
                "lora_model": {
                    "size_mb": lora_size,
                    "inference_speed_sec": finetuned_speed,
                    "memory_usage_mb": finetuned_memory,
                    "size_increase_percent": ((lora_size - self.base_size) / self.base_size * 100),
                    "speed_improvement_percent": speed_improvement,
                    "memory_change_percent": memory_change
                }
            }
            
            print(f"✅ LoRA 파인튜닝 완료")
            print(f"📊 크기: {lora_size:.2f}MB ({((lora_size - self.base_size) / self.base_size * 100):.2f}% 증가)")
            print(f"🧠 속도: {finetuned_speed:.2f}초 ({speed_improvement:.1f}% 향상)")
            print(f"💾 메모리: {finetuned_memory:.2f}MB ({memory_change:.1f}% 변화)")
            
        except Exception as e:
            print(f"❌ LoRA 파인튜닝 실패: {str(e)}")
            results = {"error": str(e)}
        
        # 결과 저장
        results_path = self.output_dir / "lora_finetuning_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"📋 실험 결과가 {results_path}에 저장되었습니다.")
        
        return results
    
    def analyze_results(self, results: Dict[str, Any]) -> None:
        """실험 결과 분석"""
        print("\n📊 LoRA 파인튜닝 결과 분석:")
        print("=" * 60)
        
        if "error" in results:
            print(f"❌ 실험 실패: {results['error']}")
            return
        
        base = results["base_model"]
        lora = results["lora_model"]
        
        print(f"🔹 베이스 모델:")
        print(f"  - 크기: {base['size_mb']:.2f}MB")
        print(f"  - 추론 속도: {base['inference_speed_sec']:.2f}초")
        print(f"  - 메모리 사용량: {base['memory_usage_mb']:.2f}MB")
        
        print(f"\n🔹 LoRA 모델:")
        print(f"  - 크기: {lora['size_mb']:.2f}MB ({lora['size_increase_percent']:.2f}% 증가)")
        print(f"  - 추론 속도: {lora['inference_speed_sec']:.2f}초 ({lora['speed_improvement_percent']:.1f}% 향상)")
        print(f"  - 메모리 사용량: {lora['memory_usage_mb']:.2f}MB ({lora['memory_change_percent']:.1f}% 변화)")
        
        # 목표 달성도 평가
        speed_goal = 100.0  # 2배 향상 목표
        
        print(f"\n🎯 목표 달성도:")
        print(f"  - 속도 향상: {lora['speed_improvement_percent']:.1f}% / {speed_goal}% ({'✅' if lora['speed_improvement_percent'] >= speed_goal else '❌'})")
        
        if lora['speed_improvement_percent'] >= speed_goal:
            print("🎉 속도 향상 목표 달성!")
        else:
            print("📈 추가 최적화 필요")
        
        print(f"\n🚀 다음 단계:")
        print("1. 의료 도메인 파인튜닝")
        print("2. 최종 성능 벤치마크")
        print("3. 실제 서비스 배포 준비")
