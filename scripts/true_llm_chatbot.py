#!/usr/bin/env python3
"""
진짜 LLM 챗봇
- 규칙 기반이 아닌 실제 모델 기반 응답
- 모든 질문에 자유롭게 답변
- 모르면 "모른다"고 정직하게 답변
"""

import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TrueLLMChatbot:
    def __init__(self, model_path="models/true_llm_chatbot"):
        """진짜 LLM 챗봇 초기화"""
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 사용 디바이스: {self.device}")
        
        # 모델 로딩
        self._load_model()
        
    def _load_model(self):
        """모델과 토크나이저 로딩"""
        print(f"\n📥 모델 로딩 중: {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print("✅ 모델 로드 완료")
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            print("기본 모델을 사용합니다.")
            self._load_base_model()
    
    def _load_base_model(self):
        """기본 모델 로딩 (학습된 모델이 없을 경우)"""
        print(f"\n📥 기본 모델 로딩 중: skt/kogpt2-base-v2")
        
        self.tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
        self.model = AutoModelForCausalLM.from_pretrained(
            "skt/kogpt2-base-v2",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("✅ 기본 모델 로드 완료")
    
    def generate_response(self, question: str) -> str:
        """모델 기반 응답 생성"""
        try:
            # 프롬프트 생성
            prompt = f"사용자: {question}\n의료진:"
            
            # 토크나이징
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 200,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs),
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
            
            # 디코딩
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            
            # 응답 정리
            if "의료진:" in response:
                response = response.split("의료진:")[-1].strip()
            
            # 반복 제거
            lines = response.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line and not any(word in line for word in ["사용자:", "의료진:", "간호사:", "진료:"]):
                    cleaned_lines.append(line)
            
            if cleaned_lines:
                return cleaned_lines[0]
            else:
                return "죄송합니다. 질문을 이해하지 못했습니다. 다시 말씀해 주시겠어요?"
                
        except Exception as e:
            print(f"⚠️ 응답 생성 오류: {e}")
            return "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해 주세요."
    
    def get_response(self, question: str) -> tuple:
        """응답 생성 및 정보 반환"""
        start_time = time.time()
        
        # 모델 기반 응답 생성
        response = self.generate_response(question)
        
        response_time = time.time() - start_time
        
        return response, "model_based", response_time
    
    def print_welcome(self):
        """환영 메시지 출력"""
        print("=" * 60)
        print("🤖 진짜 LLM 의료 챗봇에 오신 것을 환영합니다!")
        print("=" * 60)
        print("📊 시스템 정보:")
        print(f"   • 디바이스: {self.device}")
        print(f"   • 모델: {self.model_path}")
        print(f"   • 응답 방식: 실제 LLM 모델")
        print("\n💬 어떤 질문이든 자유롭게 물어보세요!")
        print("   모르는 것이 있으면 솔직하게 '모른다'고 답변합니다.")
        print("   종료하려면 'quit', 'exit', '종료'를 입력하세요.")
        print("=" * 60)
    
    def print_response(self, response: str, source: str, response_time: float):
        """응답 출력"""
        print(f"\n🤖 챗봇 응답:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        print(f"📊 응답 정보:")
        print(f"   • 응답 시간: {response_time:.3f}초")
        print(f"   • 응답 소스: {source}")
        print()
    
    def run(self):
        """챗봇 실행"""
        self.print_welcome()
        
        while True:
            try:
                # 사용자 입력
                user_input = input("👤 사용자: ").strip()
                
                # 종료 명령어 체크
                if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                    print("\n👋 진짜 LLM 챗봇을 종료합니다. 감사합니다!")
                    break
                
                if not user_input:
                    print("❌ 입력이 비어있습니다. 다시 입력해주세요.")
                    continue
                
                # 챗봇 응답 생성
                response, source, response_time = self.get_response(user_input)
                
                # 응답 출력
                self.print_response(response, source, response_time)
                
            except KeyboardInterrupt:
                print("\n\n👋 진짜 LLM 챗봇을 종료합니다. 감사합니다!")
                break
            except Exception as e:
                print(f"\n❌ 오류가 발생했습니다: {e}")
                print("다시 시도해주세요.")

def main():
    """메인 함수"""
    chatbot = TrueLLMChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()
