#!/usr/bin/env python3
"""
포트폴리오용 강화된 Knowledge Distillation
안정성과 정확성을 모두 고려한 구현
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

# MPS 비활성화 (Apple Silicon Mac 호환성)
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustDistillationTrainer(Trainer):
    """강화된 Knowledge Distillation Trainer"""
    
    def __init__(self, teacher_model, temperature=3.0, alpha=0.7, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Teacher 모델을 evaluation 모드로 설정
        self.teacher_model.eval()
        
        # 버전 호환성을 위한 메서드 오버라이드
        self._setup_compute_loss()
        
    def _setup_compute_loss(self):
        """compute_loss 메서드 동적 설정"""
        import inspect
        
        # 현재 Trainer의 compute_loss 시그니처 확인
        try:
            sig = inspect.signature(super().compute_loss)
            params = list(sig.parameters.keys())
            
            if 'num_items_in_batch' in params:
                # 최신 버전
                self._compute_loss_method = self._compute_loss_v2
            else:
                # 이전 버전
                self._compute_loss_method = self._compute_loss_v1
                
        except Exception as e:
            logger.warning(f"시그니처 확인 실패, 기본 방법 사용: {e}")
            self._compute_loss_method = self._compute_loss_v1
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """버전 호환 compute_loss"""
        return self._compute_loss_method(model, inputs, return_outputs, **kwargs)
    
    def _compute_loss_v1(self, model, inputs, return_outputs=False):
        """이전 버전 호환"""
        return self._compute_distillation_loss(model, inputs, return_outputs)
    
    def _compute_loss_v2(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """최신 버전 호환"""
        return self._compute_distillation_loss(model, inputs, return_outputs)
    
    def _compute_distillation_loss(self, model, inputs, return_outputs=False):
        """실제 Knowledge Distillation Loss 계산"""
        # Student 모델 출력
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # Teacher 모델 출력 (gradient 계산 없이)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Hard targets (ground truth)
        labels = inputs.get("labels")
        if labels is not None:
            # Cross entropy loss for hard targets
            hard_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        else:
            hard_loss = 0.0
        
        # Soft targets (teacher knowledge)
        # Temperature scaling 적용
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence loss for soft targets
        soft_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return (total_loss, student_outputs) if return_outputs else total_loss


class RobustKnowledgeDistiller:
    """포트폴리오용 강화된 Knowledge Distiller"""
    
    def __init__(self, teacher_model_name: str, student_model_name: str, output_dir: str = "models/distilled"):
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 로드
        self._load_models()
        
        # 성능 측정
        self._measure_baseline_performance()
        
    def _load_models(self):
        """모델 로드"""
        print(f"🔄 Teacher 모델 로딩 중: {self.teacher_model_name}")
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            self.teacher_model_name,
            torch_dtype=torch.float32,
            device_map="cpu"  # CPU 강제 사용 (MPS 문제 해결)
        )
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_name)
        
        print(f"🔄 Student 모델 로딩 중: {self.student_model_name}")
        self.student_model = AutoModelForCausalLM.from_pretrained(
            self.student_model_name,
            torch_dtype=torch.float32,
            device_map="cpu"  # CPU 강제 사용 (MPS 문제 해결)
        )
        self.student_tokenizer = AutoTokenizer.from_pretrained(self.student_model_name)
        
        # 패딩 토큰 설정 및 호환성 확인
        for tokenizer in [self.teacher_tokenizer, self.student_tokenizer]:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        # 토크나이저 호환성 확인
        print(f"📊 Teacher vocab size: {len(self.teacher_tokenizer)}")
        print(f"📊 Student vocab size: {len(self.student_tokenizer)}")
        
        # Student 모델의 vocabulary 크기 확인
        student_vocab_size = self.student_model.config.vocab_size
        teacher_vocab_size = self.teacher_model.config.vocab_size
        print(f"📊 Teacher model vocab: {teacher_vocab_size}")
        print(f"📊 Student model vocab: {student_vocab_size}")
        
        # Student 토크나이저를 Teacher와 맞춤 (필요시)
        if len(self.student_tokenizer) != student_vocab_size:
            print("⚠️ Student 토크나이저와 모델 vocabulary 불일치 감지")
            # Student 토크나이저를 모델에 맞춤
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
        
        print(f"✅ 모델 로딩 완료")
    
    def _measure_baseline_performance(self):
        """베이스라인 성능 측정"""
        print("📊 베이스라인 성능 측정 중...")
        
        self.teacher_size = self._get_model_size(self.teacher_model)
        self.student_size = self._get_model_size(self.student_model)
        self.teacher_speed = self._benchmark_inference(self.teacher_model, self.teacher_tokenizer)
        self.student_speed = self._benchmark_inference(self.student_model, self.student_tokenizer)
        self.teacher_memory = self._get_memory_usage()
        self.student_memory = self._get_memory_usage()
        
        print(f"📊 Teacher 크기: {self.teacher_size:.2f}MB")
        print(f"📊 Student 크기: {self.student_size:.2f}MB")
        print(f"📊 크기 감소: {((self.teacher_size - self.student_size) / self.teacher_size * 100):.1f}%")
        print(f"🧠 Teacher 속도: {self.teacher_speed:.2f}초")
        print(f"🧠 Student 속도: {self.student_speed:.2f}초")
    
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
        
        # 텍스트 추출
        texts = []
        for item in medical_data[:500]:  # 500개로 제한 (안정성)
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
            ] * 20
        
        return Dataset.from_dict({"text": texts})
    
    def _tokenize_data(self, dataset: Dataset) -> Dataset:
        """데이터 토크나이징"""
        def tokenize_function(examples):
            return self.student_tokenizer(
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
    
    def run_robust_distillation(self, 
                               temperature: float = 3.0,
                               alpha: float = 0.7,
                               num_epochs: int = 2,
                               batch_size: int = 2,
                               learning_rate: float = 5e-5) -> Dict[str, Any]:
        """강화된 Knowledge Distillation 실행"""
        print("🚀 강화된 Knowledge Distillation 시작")
        print("=" * 60)
        
        # 데이터 준비
        dataset = self._prepare_medical_data()
        tokenized_dataset = self._tokenize_data(dataset)
        
        # Data Collator 설정
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.student_tokenizer,
            mlm=False
        )
        
        # Training Arguments 설정 (CPU 최적화)
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
            fp16=False,  # CPU 환경에서 안정성 우선
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,  # wandb 등 비활성화
            use_cpu=True,  # CPU 강제 사용
            dataloader_num_workers=0,  # 멀티프로세싱 비활성화
        )
        
        # Robust Trainer 초기화
        trainer = RobustDistillationTrainer(
            teacher_model=self.teacher_model,
            temperature=temperature,
            alpha=alpha,
            model=self.student_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.student_tokenizer,
        )
        
        # Distillation 실행
        print("🎓 Knowledge Distillation 학습 시작...")
        start_time = time.time()
        
        try:
            trainer.train()
            training_time = time.time() - start_time
            print(f"✅ Distillation 완료: {training_time:.2f}초")
            
            # Distilled 모델 저장
            distilled_model_path = self.output_dir / "robust_distilled_model"
            trainer.save_model(str(distilled_model_path))
            self.student_tokenizer.save_pretrained(str(distilled_model_path))
            
            # Distilled 모델 성능 측정
            print("📊 Distilled 모델 성능 측정 중...")
            distilled_size = self._get_model_size(self.student_model)
            distilled_speed = self._benchmark_inference(self.student_model, self.student_tokenizer)
            distilled_memory = self._get_memory_usage()
            
            # 결과 계산
            size_reduction = ((self.teacher_size - distilled_size) / self.teacher_size) * 100
            speed_improvement = ((self.teacher_speed - distilled_speed) / self.teacher_speed) * 100
            memory_reduction = ((self.teacher_memory - distilled_memory) / self.teacher_memory) * 100
            
            results = {
                "distillation_config": {
                    "temperature": temperature,
                    "alpha": alpha,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "training_time_sec": training_time
                },
                "teacher_model": {
                    "size_mb": self.teacher_size,
                    "inference_speed_sec": self.teacher_speed,
                    "memory_usage_mb": self.teacher_memory
                },
                "student_model": {
                    "size_mb": self.student_size,
                    "inference_speed_sec": self.student_speed,
                    "memory_usage_mb": self.student_memory
                },
                "distilled_model": {
                    "size_mb": distilled_size,
                    "inference_speed_sec": distilled_speed,
                    "memory_usage_mb": distilled_memory,
                    "size_reduction_percent": size_reduction,
                    "speed_improvement_percent": speed_improvement,
                    "memory_reduction_percent": memory_reduction
                }
            }
            
            print(f"✅ 강화된 Knowledge Distillation 완료")
            print(f"📊 크기: {distilled_size:.2f}MB ({size_reduction:.1f}% 감소)")
            print(f"🧠 속도: {distilled_speed:.2f}초 ({speed_improvement:.1f}% 향상)")
            print(f"💾 메모리: {distilled_memory:.2f}MB ({memory_reduction:.1f}% 감소)")
            
        except Exception as e:
            print(f"❌ Knowledge Distillation 실패: {str(e)}")
            results = {"error": str(e)}
        
        # 결과 저장
        results_path = self.output_dir / "robust_distillation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"📋 실험 결과가 {results_path}에 저장되었습니다.")
        
        return results


def main():
    """강화된 Knowledge Distillation 실행"""
    print("🚀 포트폴리오용 강화된 Knowledge Distillation")
    print("=" * 60)
    
    # Teacher: 20% Pruned 모델, Student: KoGPT2-base
    distiller = RobustKnowledgeDistiller(
        teacher_model_name="models/pruned/model_20_percent_pruned",
        student_model_name="skt/kogpt2-base-v2",
        output_dir="models/distilled"
    )
    
    # Distillation 실행
    results = distiller.run_robust_distillation(
        temperature=3.0,
        alpha=0.7,
        num_epochs=2,
        batch_size=2,
        learning_rate=5e-5
    )
    
    print("\n✅ 포트폴리오용 Knowledge Distillation 완료!")


if __name__ == "__main__":
    main()
