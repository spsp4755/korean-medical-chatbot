"""
Knowledge Distillation 모듈
Teacher-Student 구조로 지식 전달을 통한 모델 경량화
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

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistillationTrainer(Trainer):
    """Knowledge Distillation을 위한 커스텀 Trainer"""
    
    def __init__(self, teacher_model, temperature=3.0, alpha=0.7, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha  # distillation loss weight
        
        # Teacher 모델을 evaluation 모드로 설정
        self.teacher_model.eval()
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Knowledge Distillation Loss 계산"""
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


class KnowledgeDistiller:
    """Knowledge Distillation을 통한 모델 경량화"""
    
    def __init__(self, teacher_model_name: str, student_model_name: str, output_dir: str = "models/distilled"):
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Teacher 모델 로드
        print(f"🔄 Teacher 모델 로딩 중: {teacher_model_name}")
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        
        # Student 모델 로드 (더 작은 모델)
        print(f"🔄 Student 모델 로딩 중: {student_model_name}")
        self.student_model = AutoModelForCausalLM.from_pretrained(
            student_model_name,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        
        # 패딩 토큰 설정
        for tokenizer in [self.teacher_tokenizer, self.student_tokenizer]:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        self.distilled_model = None
        self.distillation_results = {}
        
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
    
    def _prepare_training_data(self, data_path: str = "data/processed") -> Dataset:
        """학습 데이터 준비"""
        print("📚 학습 데이터 준비 중...")
        
        # 의료 데이터 로드
        medical_data = []
        data_dir = Path(data_path)
        
        # 의료 관련 JSON 파일들 로드
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
            else:
                print(f"⚠️ 파일 없음: {file_path}")
        
        print(f"📊 로드된 의료 데이터: {len(medical_data)}개 샘플")
        
        # 텍스트 데이터 추출 및 전처리
        texts = []
        for item in medical_data[:1000]:  # 처음 1000개만 사용 (메모리 절약)
            if isinstance(item, dict):
                # 다양한 키에서 텍스트 추출
                for key in ['text', 'content', 'question', 'answer', 'instruction', 'input', 'output']:
                    if key in item and isinstance(item[key], str):
                        texts.append(item[key])
                        break
            elif isinstance(item, str):
                texts.append(item)
        
        # 중복 제거 및 길이 제한
        texts = list(set(texts))
        texts = [text for text in texts if len(text) > 10 and len(text) < 500]
        
        print(f"📝 전처리된 텍스트: {len(texts)}개")
        
        # Dataset 생성
        dataset = Dataset.from_dict({"text": texts})
        return dataset
    
    def _tokenize_data(self, dataset: Dataset, tokenizer) -> Dataset:
        """데이터 토크나이징"""
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def run_distillation(self, 
                        temperature: float = 3.0,
                        alpha: float = 0.7,
                        num_epochs: int = 3,
                        batch_size: int = 4,
                        learning_rate: float = 5e-5) -> Dict[str, Any]:
        """Knowledge Distillation 실행"""
        print("🚀 Knowledge Distillation 시작")
        print("=" * 60)
        
        # 원본 모델 성능 측정
        print("📊 Teacher 모델 성능 측정 중...")
        teacher_size = self._get_model_size(self.teacher_model)
        teacher_speed = self._benchmark_inference(self.teacher_model, self.teacher_tokenizer)
        teacher_memory = self._get_memory_usage()
        
        print("📊 Student 모델 성능 측정 중...")
        student_size = self._get_model_size(self.student_model)
        student_speed = self._benchmark_inference(self.student_model, self.student_tokenizer)
        student_memory = self._get_memory_usage()
        
        print(f"📊 Teacher 크기: {teacher_size:.2f}MB")
        print(f"📊 Student 크기: {student_size:.2f}MB")
        print(f"📊 크기 감소: {((teacher_size - student_size) / teacher_size * 100):.1f}%")
        
        # 학습 데이터 준비
        try:
            dataset = self._prepare_training_data()
            tokenized_dataset = self._tokenize_data(dataset, self.student_tokenizer)
        except Exception as e:
            print(f"⚠️ 데이터 로드 실패, 더미 데이터 사용: {e}")
            # 더미 데이터 생성
            dummy_texts = [
                "안녕하세요. 의료 상담을 받고 싶습니다.",
                "머리가 아픈데 어떻게 해야 할까요?",
                "감기 증상이 있는데 병원에 가야 할까요?",
                "복통이 심한데 응급실에 가야 할까요?",
                "피부에 발진이 생겼는데 원인이 뭘까요?"
            ] * 20  # 100개 샘플
            dataset = Dataset.from_dict({"text": dummy_texts})
            tokenized_dataset = self._tokenize_data(dataset, self.student_tokenizer)
        
        # Data Collator 설정
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.student_tokenizer,
            mlm=False
        )
        
        # Training Arguments 설정
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "training_output"),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            save_steps=500,
            eval_strategy="no",  # evaluation_strategy -> eval_strategy
            save_total_limit=2,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        # Distillation Trainer 초기화
        trainer = DistillationTrainer(
            teacher_model=self.teacher_model,
            temperature=temperature,
            alpha=alpha,
            model=self.student_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.student_tokenizer,  # 토크나이저 추가
        )
        
        # Distillation 실행
        print("🎓 Knowledge Distillation 학습 시작...")
        start_time = time.time()
        
        try:
            trainer.train()
            training_time = time.time() - start_time
            print(f"✅ Distillation 완료: {training_time:.2f}초")
            
            # Distilled 모델 저장
            distilled_model_path = self.output_dir / "distilled_model"
            trainer.save_model(str(distilled_model_path))
            self.student_tokenizer.save_pretrained(str(distilled_model_path))
            self.distilled_model = self.student_model
            
            # Distilled 모델 성능 측정
            print("📊 Distilled 모델 성능 측정 중...")
            distilled_size = self._get_model_size(self.distilled_model)
            distilled_speed = self._benchmark_inference(self.distilled_model, self.student_tokenizer)
            distilled_memory = self._get_memory_usage()
            
            # 결과 계산
            size_reduction = ((teacher_size - distilled_size) / teacher_size) * 100
            speed_improvement = ((teacher_speed - distilled_speed) / teacher_speed) * 100
            memory_reduction = ((teacher_memory - distilled_memory) / teacher_memory) * 100
            
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
                    "size_mb": teacher_size,
                    "inference_speed_sec": teacher_speed,
                    "memory_usage_mb": teacher_memory
                },
                "student_model": {
                    "size_mb": student_size,
                    "inference_speed_sec": student_speed,
                    "memory_usage_mb": student_memory
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
            
            print(f"✅ Knowledge Distillation 완료")
            print(f"📊 크기: {distilled_size:.2f}MB ({size_reduction:.1f}% 감소)")
            print(f"🧠 속도: {distilled_speed:.2f}초 ({speed_improvement:.1f}% 향상)")
            print(f"💾 메모리: {distilled_memory:.2f}MB ({memory_reduction:.1f}% 감소)")
            
        except Exception as e:
            print(f"❌ Knowledge Distillation 실패: {str(e)}")
            results = {"error": str(e)}
        
        # 결과 저장
        results_path = self.output_dir / "distillation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"📋 실험 결과가 {results_path}에 저장되었습니다.")
        
        return results
    
    def analyze_results(self, results: Dict[str, Any]) -> None:
        """실험 결과 분석"""
        print("\n📊 Knowledge Distillation 결과 분석:")
        print("=" * 60)
        
        if "error" in results:
            print(f"❌ 실험 실패: {results['error']}")
            return
        
        teacher = results["teacher_model"]
        student = results["student_model"]
        distilled = results["distilled_model"]
        
        print(f"🔹 Teacher 모델:")
        print(f"  - 크기: {teacher['size_mb']:.2f}MB")
        print(f"  - 추론 속도: {teacher['inference_speed_sec']:.2f}초")
        print(f"  - 메모리 사용량: {teacher['memory_usage_mb']:.2f}MB")
        
        print(f"\n🔹 Student 모델:")
        print(f"  - 크기: {student['size_mb']:.2f}MB")
        print(f"  - 추론 속도: {student['inference_speed_sec']:.2f}초")
        print(f"  - 메모리 사용량: {student['memory_usage_mb']:.2f}MB")
        
        print(f"\n🔹 Distilled 모델:")
        print(f"  - 크기: {distilled['size_mb']:.2f}MB ({distilled['size_reduction_percent']:.1f}% 감소)")
        print(f"  - 추론 속도: {distilled['inference_speed_sec']:.2f}초 ({distilled['speed_improvement_percent']:.1f}% 향상)")
        print(f"  - 메모리 사용량: {distilled['memory_usage_mb']:.2f}MB ({distilled['memory_reduction_percent']:.1f}% 감소)")
        
        # 목표 달성도 평가
        size_goal = 50.0  # 50% 감소 목표
        speed_goal = 100.0  # 2배 향상 목표
        memory_goal = 70.0  # 70% 감소 목표
        
        print(f"\n🎯 목표 달성도:")
        print(f"  - 크기 감소: {distilled['size_reduction_percent']:.1f}% / {size_goal}% ({'✅' if distilled['size_reduction_percent'] >= size_goal else '❌'})")
        print(f"  - 속도 향상: {distilled['speed_improvement_percent']:.1f}% / {speed_goal}% ({'✅' if distilled['speed_improvement_percent'] >= speed_goal else '❌'})")
        print(f"  - 메모리 감소: {distilled['memory_reduction_percent']:.1f}% / {memory_goal}% ({'✅' if distilled['memory_reduction_percent'] >= memory_goal else '❌'})")
        
        print(f"\n🚀 다음 단계:")
        print("1. LoRA 파인튜닝 적용")
        print("2. 의료 도메인 파인튜닝")
        print("3. 최종 성능 벤치마크")
