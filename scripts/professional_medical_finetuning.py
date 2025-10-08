#!/usr/bin/env python3
"""
Professional 의료 파인튜닝 스크립트
- Essential 학습 후 Professional 데이터로 추가 학습
- 42dot LLM 사용
- 메모리 최적화
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

class MedicalDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 대화 형식으로 변환
        if 'question' in item and 'answer' in item:
            text = f"환자: {item['question']}\n의료진: {item['answer']}"
        else:
            text = str(item)
        
        # 토크나이징
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

class ProfessionalMedicalFineTuner:
    def __init__(self, model_path="models/medical_finetuned_advanced"):
        """Professional 의료 파인튜너 초기화 (Essential 학습된 모델 사용)"""
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 사용 디바이스: {self.device}")
        
        # 모델 로딩
        self._load_model()
        
    def _load_model(self):
        """Essential 학습된 모델과 토크나이저 로딩"""
        print(f"\n📥 Essential 학습된 모델 로딩 중: {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True,
                use_cache=False
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print("✅ Essential 학습된 모델 로드 완료")
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            raise
    
    def load_data(self, data_paths):
        """데이터 로딩"""
        print(f"\n📊 Professional 데이터 로딩 중...")
        
        all_data = []
        for path in data_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.extend(data)
                    print(f"✅ {path}: {len(data)}개 샘플")
            else:
                print(f"⚠️ {path}: 파일 없음")
        
        print(f"📊 총 Professional 데이터: {len(all_data)}개 샘플")
        return all_data
    
    def fine_tune(self, train_data, val_data, output_dir="./models/medical_finetuned_professional"):
        """Professional 데이터로 추가 파인튜닝"""
        print(f"\n🚀 Professional 의료 파인튜닝 시작")
        print(f"📊 Train: {len(train_data)}개, Val: {len(val_data)}개")
        
        # 데이터셋 생성
        train_dataset = MedicalDataset(train_data, self.tokenizer)
        val_dataset = MedicalDataset(val_data, self.tokenizer)
        
        # 데이터 콜레이터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Professional 학습용 파라미터 (더 작은 학습률)
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=2,             # Professional은 더 적은 에포크
            per_device_train_batch_size=1,  # 극한 메모리 절약
            per_device_eval_batch_size=1,   # 극한 메모리 절약
            warmup_steps=25,                # 최소화
            weight_decay=0.0,               # 42dot 공식: 0
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,               # 로깅 빈도 감소
            eval_strategy="steps",
            eval_steps=200,                 # 평가 빈도 감소
            save_steps=200,                 # 저장 빈도 감소
            save_total_limit=2,             # 체크포인트 수 최소화
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            gradient_accumulation_steps=64, # 극한 그래디언트 누적
            fp16=False,                     # MPS 호환성
            max_steps=300,                  # Professional은 더 적은 스텝
            learning_rate=1e-5,             # 더 작은 학습률 (추가 학습)
            warmup_ratio=0.03,              # 42dot 공식: 0.03
            max_grad_norm=0.5,              # 그래디언트 클리핑 강화
            lr_scheduler_type="cosine",     # 42dot 공식 스케줄러
            dataloader_num_workers=0,       # 멀티프로세싱 비활성화
            dataloader_drop_last=True       # 마지막 배치 드롭
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
        
        # 학습 실행
        print(f"\n📚 Professional 학습 시작...")
        trainer.train()
        
        # 모델 저장
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"\n✅ Professional 파인튜닝 완료!")
        print(f"📁 모델 저장 위치: {output_dir}")
        
        return trainer

def main():
    """메인 실행 함수"""
    print("🚀 Professional 의료 파인튜닝 시작")
    print("=" * 50)
    
    # 파인튜너 초기화 (Essential 학습된 모델 사용)
    fine_tuner = ProfessionalMedicalFineTuner()
    
    # Professional 데이터 로딩
    train_paths = [
        "data/processed/splits/professional_medical_train.json"
    ]
    
    val_paths = [
        "data/processed/splits/professional_medical_val.json"
    ]
    
    train_data = fine_tuner.load_data(train_paths)
    val_data = fine_tuner.load_data(val_paths)
    
    # 파인튜닝 실행
    trainer = fine_tuner.fine_tune(train_data, val_data)
    
    print("\n🎉 Professional 의료 파인튜닝 완료!")

if __name__ == "__main__":
    main()
