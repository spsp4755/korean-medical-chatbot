#!/usr/bin/env python3
"""
PEFT 학습된 의료 모델을 활용한 챗봇
- LoRA 어댑터 사용
- 메모리 효율적
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PEFTMedicalChatbot:
    def __init__(self, base_model_name="42dot/42dot_LLM-SFT-1.3B", peft_model_path="models/medical_finetuned_peft"):
        """PEFT 의료 챗봇 초기화"""
        self.base_model_name = base_model_name
        self.peft_model_path = peft_model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 사용 디바이스: {self.device}")
        
        # 모델 로딩
        self._load_model()
        
    def _load_model(self):
        """PEFT 모델과 토크나이저 로딩"""
        print(f"\n📥 PEFT 모델 로딩 중...")
        
        try:
            # 기본 모델 로딩
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            # PEFT 모델 로딩
            self.model = PeftModel.from_pretrained(
                self.base_model,
                self.peft_model_path,
                torch_dtype=torch.float32
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print("✅ PEFT 모델 로드 완료")
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            raise
    
    def generate_response(self, question, max_length=200):
        """질문에 대한 응답 생성"""
        prompt = f"환자: {question}\n의료진:"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        return response
    
    def chat(self):
        """대화형 챗봇 실행"""
        print("\n🏥 PEFT 의료 상담 챗봇이 시작되었습니다!")
        print("💡 질문을 입력하세요. 'quit'을 입력하면 종료됩니다.")
        print("=" * 50)
        
        while True:
            try:
                # 사용자 입력
                question = input("\n👤 환자: ").strip()
                
                if question.lower() in ['quit', 'exit', '종료']:
                    print("\n👋 의료 상담을 종료합니다. 감사합니다!")
                    break
                
                if not question:
                    continue
                
                # 응답 생성
                print("🤖 의료진: ", end="", flush=True)
                response = self.generate_response(question)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 의료 상담을 종료합니다. 감사합니다!")
                break
            except Exception as e:
                print(f"\n❌ 오류가 발생했습니다: {e}")

def main():
    """메인 실행 함수"""
    print("🚀 PEFT 의료 챗봇 시작")
    print("=" * 50)
    
    # 챗봇 초기화
    chatbot = PEFTMedicalChatbot()
    
    # 대화 시작
    chatbot.chat()

if __name__ == "__main__":
    main()
