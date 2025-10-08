import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time
import json
from datetime import datetime

class ImprovedPromptMedicalChatbot:
    def __init__(self, base_model_path="42dot/42dot_LLM-SFT-1.3B", 
                 peft_model_path="models/medical_finetuned_peft"):
        self.base_model_path = base_model_path
        self.peft_model_path = peft_model_path
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 사용 디바이스: {self.device}")
        self._load_model()
        
        # 대화 기록 저장
        self.conversation_log = []
        
    def _load_model(self):
        """PEFT 모델 로드"""
        print("PEFT 모델 로딩 중...")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 베이스 모델 로드 (CPU 모드로 MPS 오류 해결)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        # PEFT 모델 로드
        self.model = PeftModel.from_pretrained(base_model, self.peft_model_path)
        self.model.eval()
        
        print("PEFT 모델 로딩 완료!")
    
    def create_improved_prompt(self, question):
        """개선된 프롬프트 생성"""
        prompt = f"""당신은 친근하고 실용적인 의료 상담 챗봇입니다. 환자의 질문에 대해 다음과 같이 답변해주세요:

1. 먼저 응급상황인지 판단하세요
2. 일반인이 이해하기 쉬운 언어로 설명하세요
3. 즉시 실행 가능한 조치를 제시하세요
4. 답변은 200자 이내로 간결하게 하세요
5. 필요시 의사 상담을 권하세요

환자 질문: {question}

의료진 답변:"""
        return prompt
    
    def generate_response(self, question, max_length=200, temperature=0.3):
        """질문에 대한 응답 생성 (개선된 프롬프트 사용)"""
        start_time = time.time()
        
        # 개선된 프롬프트 사용
        prompt = self.create_improved_prompt(question)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs.input_ids.shape[1] + max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                top_p=0.9,
                top_k=50
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # 대화 기록 저장
        self.conversation_log.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "response": response,
            "response_time": response_time,
            "prompt_type": "improved"
        })
        
        return response, response_time
    
    def save_conversation_log(self, filename="results/improved_prompt_conversation.json"):
        """대화 기록 저장"""
        os.makedirs("results", exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.conversation_log, f, ensure_ascii=False, indent=2)
        print(f"대화 기록이 {filename}에 저장되었습니다.")

def main():
    chatbot = ImprovedPromptMedicalChatbot()
    
    print("\n" + "="*60)
    print("🏥 개선된 프롬프트 의료 상담 챗봇")
    print("💡 더 친근하고 실용적인 답변을 제공합니다!")
    print("💡 질문을 입력하세요. 'quit'을 입력하면 종료됩니다.")
    print("="*60)
    
    # 테스트 질문들
    test_questions = [
        "장염에 걸린거 같은데 어떻게 해야하나요?",
        "머리가 아픈데 진통제 먹어도 될까요?",
        "열이 나는데 병원 가야 할까요?",
        "복통이 심한데 응급실 가야 하나요?"
    ]
    
    print("\n🧪 테스트 질문들:")
    for i, q in enumerate(test_questions, 1):
        print(f"{i}. {q}")
    
    print("\n" + "="*60)
    
    while True:
        user_input = input("\n👤 환자: ")
        if user_input.lower() in ["quit", "exit", "종료", "q"]:
            print("챗봇을 종료합니다.")
            break
        
        response, response_time = chatbot.generate_response(user_input)
        print(f"🤖 의료진: {response}")
        print(f"📊 응답 시간: {response_time:.3f}초")
    
    # 대화 기록 저장
    chatbot.save_conversation_log()

if __name__ == "__main__":
    main()


