#!/usr/bin/env python3
"""
진짜 LLM 챗봇 학습 스크립트
- train/valid/test 데이터 분리 사용
- data leakage 방지
- 실제 대화형 챗봇 학습
"""

import os
import json
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
# from datasets import Dataset  # datasets 라이브러리 없이도 작동
import warnings
warnings.filterwarnings("ignore")

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"  # wandb 완전 비활성화

class SimpleDataset(TorchDataset):
    """간단한 PyTorch Dataset"""
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class TrueLLMTrainer:
    def __init__(self, model_name="skt/kogpt2-base-v2"):
        """진짜 LLM 챗봇 학습자 초기화"""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 사용 디바이스: {self.device}")
        
        # 모델과 토크나이저 로딩
        self._load_model()
        
    def _load_model(self):
        """모델과 토크나이저 로딩"""
        print(f"\n📥 모델 로딩 중: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
            
        print("✅ 모델 로드 완료")
    
    def load_data(self, split="train"):
        """데이터 로딩 (data leakage 방지)"""
        print(f"\n📊 {split} 데이터 로딩 중...")
        
        # 오직 해당 split만 로딩
        data_files = {
            "train": "data/processed/splits/professional_medical_train.json",
            "validation": "data/processed/splits/professional_medical_val.json",
            "test": "data/processed/splits/professional_medical_test.json"
        }
        
        file_path = data_files[split]
        if not os.path.exists(file_path):
            print(f"❌ {file_path} 파일이 없습니다.")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ {split} 데이터 로드 완료: {len(data)}개 샘플")
        return data
    
    def create_conversation_format(self, data):
        """대화 형식으로 데이터 변환"""
        print(f"\n🔄 대화 형식으로 데이터 변환 중...")
        
        formatted_data = []
        for item in data:
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            
            if question and answer:
                # 대화 형식으로 변환
                conversation = f"사용자: {question}\n의료진: {answer}"
                formatted_data.append({"text": conversation})
        
        print(f"✅ 변환 완료: {len(formatted_data)}개 대화")
        return formatted_data
    
    def tokenize_function(self, examples):
        """토크나이징 함수"""
        # 배치 처리용으로 수정
        if isinstance(examples["text"], list):
            # 배치 처리
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
        else:
            # 단일 텍스트 처리
            tokenized = self.tokenizer(
                [examples["text"]],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
        
        # 라벨 설정 (입력과 동일)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def prepare_dataset(self, data):
        """데이터셋 준비"""
        print(f"\n📋 데이터셋 준비 중...")
        
        # 대화 형식으로 변환
        formatted_data = self.create_conversation_format(data)
        
        # 간단한 토크나이징 (배치 처리 없이)
        tokenized_data = []
        for item in formatted_data:
            # 개별 토크나이징
            tokens = self.tokenizer(
                item["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            
            tokenized_data.append({
                "input_ids": tokens["input_ids"].squeeze(0),  # 배치 차원 제거
                "attention_mask": tokens["attention_mask"].squeeze(0),
                "labels": tokens["input_ids"].squeeze(0)  # 라벨은 입력과 동일
            })
        
        print(f"✅ 데이터셋 준비 완료: {len(tokenized_data)}개 샘플")
        return tokenized_data
    
    def train(self, train_data, validation_data, output_dir="models/true_llm_chatbot"):
        """모델 학습"""
        print(f"\n🚀 진짜 LLM 챗봇 학습 시작")
        print("=" * 50)
        
        # 데이터셋 준비
        train_data_processed = self.prepare_dataset(train_data)
        val_data_processed = self.prepare_dataset(validation_data)
        
        # PyTorch Dataset으로 변환
        train_dataset = SimpleDataset(train_data_processed)
        val_dataset = SimpleDataset(val_data_processed)
        
        # 데이터 콜레이터 (패딩 문제 해결)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM이므로 MLM 비활성화
            pad_to_multiple_of=8  # 메모리 효율성을 위해 8의 배수로 패딩
        )
        
        # 학습 인수 설정
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,  # KoGPT2는 작은 모델이므로 2로 증가
            per_device_eval_batch_size=2,   # KoGPT2는 작은 모델이므로 2로 증가
            warmup_steps=50,                # warmup 단계 복원
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            eval_strategy="steps",  # evaluation_strategy -> eval_strategy
            eval_steps=100,
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],  # wandb 완전 비활성화
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            gradient_accumulation_steps=4,  # KoGPT2는 작은 모델이므로 4로 감소
            fp16=False,                     # MPS에서는 FP16 지원 안됨
            max_steps=500                   # 학습 단계 감소 (메모리 절약)
        )
        
        # 트레이너 생성
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # 학습 시작
        print(f"\n📚 학습 시작...")
        trainer.train()
        
        # 모델 저장
        print(f"\n💾 모델 저장 중...")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ 학습 완료! 모델 저장 위치: {output_dir}")
        return trainer
    
    def evaluate(self, test_data):
        """테스트 데이터로 평가"""
        print(f"\n📊 테스트 데이터로 평가 중...")
        
        test_dataset = self.prepare_dataset(test_data)
        
        # 데이터 콜레이터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 트레이너 생성
        trainer = Trainer(
            model=self.model,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # 평가 실행
        eval_results = trainer.evaluate(test_dataset)
        
        print(f"📈 평가 결과:")
        for key, value in eval_results.items():
            print(f"   {key}: {value:.4f}")
        
        return eval_results

def main():
    """메인 실행 함수"""
    print("🚀 진짜 LLM 챗봇 학습 시작")
    print("=" * 50)
    
    # 학습자 초기화
    trainer = TrueLLMTrainer()
    
    # 데이터 로딩 (data leakage 방지)
    print(f"\n📊 데이터 로딩 (data leakage 방지)")
    train_data = trainer.load_data("train")
    val_data = trainer.load_data("validation")
    test_data = trainer.load_data("test")
    
    if not train_data or not val_data:
        print("❌ 학습 데이터가 부족합니다.")
        return
    
    # 학습 실행
    trainer.train(train_data, val_data)
    
    # 테스트 평가
    if test_data:
        trainer.evaluate(test_data)
    
    print(f"\n✅ 진짜 LLM 챗봇 학습 완료!")

if __name__ == "__main__":
    main()
