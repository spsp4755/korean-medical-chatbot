import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time
import json
from datetime import datetime

class OptimizedFastChatbot:
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
        """PEFT 모델 로드 (최적화)"""
        print("PEFT 모델 로딩 중...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 로드 최적화
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            use_cache=True,  # 캐시 사용으로 속도 향상
            use_safetensors=False  # SafeTensors 비활성화로 속도 향상
        )
        
        self.model = PeftModel.from_pretrained(base_model, self.peft_model_path)
        self.model.eval()
        
        # 모델 최적화
        self.model = torch.compile(self.model, mode="reduce-overhead")  # PyTorch 2.0 컴파일
        
        print("PEFT 모델 로딩 완료!")
    
    def create_optimized_prompt(self, question):
        """최적화된 프롬프트 (더 간결하고 명확)"""
        prompt = f"""의료 상담 챗봇입니다. 간결하고 정확하게 답변하세요:

1. 응급상황이면 "즉시 병원" 안내
2. 일반인이 이해하기 쉽게 설명
3. 구체적인 조치 제시
4. 150자 이내로 간결하게

질문: {question}
답변:"""
        return prompt
    
    def generate_response(self, question, max_length=150, temperature=0.1):
        """최적화된 응답 생성"""
        start_time = time.time()
        
        prompt = self.create_optimized_prompt(question)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256  # 입력 길이 단축
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs.input_ids.shape[1] + max_length,
                num_return_sequences=1,
                temperature=temperature,  # 더 낮은 temperature로 일관성 향상
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.05,  # 반복 감소
                top_p=0.8,  # 더 집중된 생성
                top_k=30,   # 더 제한된 선택
                early_stopping=True,  # 조기 종료로 속도 향상
                use_cache=True  # 캐시 사용
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
            "prompt_type": "optimized_fast"
        })
        
        return response, response_time
    
    def save_conversation_log(self, filename="results/optimized_fast_conversation.json"):
        """대화 기록 저장"""
        os.makedirs("results", exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.conversation_log, f, ensure_ascii=False, indent=2)
        print(f"대화 기록이 {filename}에 저장되었습니다.")

def main():
    chatbot = OptimizedFastChatbot()
    
    print("\n" + "="*60)
    print("🚀 최적화된 고속 의료 상담 챗봇")
    print("💡 빠르고 정확한 답변을 제공합니다!")
    print("💡 질문을 입력하세요. 'quit'을 입력하면 종료됩니다.")
    print("="*60)
    
    # 테스트 질문들
    test_questions = [
        "장염에 걸린거 같은데 어떻게 해야하나요?",
        "머리가 아픈데 진통제 먹어도 될까요?",
        "열이 나는데 병원 가야 할까요?",
        "복통이 심한데 응급실 가야 하나요?",
        "저체온증이면 어떻게 해야 하나요?"  # 이전에 잘못 답변한 질문
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
