#!/usr/bin/env python3
"""
데이터 전처리 스크립트
AI Hub에서 다운로드한 데이터를 학습에 적합한 형태로 전처리합니다.
"""

import os
import json
import zipfile
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import argparse
from tqdm import tqdm
import yaml

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """설정 파일을 로드합니다."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def extract_zip_files(data_dir: str, output_dir: str) -> None:
    """압축 파일들을 압축 해제합니다."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"압축 파일 압축 해제 중... {data_path} -> {output_path}")
    
    for zip_file in tqdm(data_path.rglob("*.zip.part0")):
        # .part0 파일을 .zip으로 복사하여 압축 해제
        temp_zip = zip_file.with_suffix('.zip')
        temp_zip.write_bytes(zip_file.read_bytes())
        
        try:
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                # 압축 해제할 디렉토리 생성
                extract_dir = output_path / zip_file.parent.name
                extract_dir.mkdir(parents=True, exist_ok=True)
                
                # 압축 해제
                zip_ref.extractall(extract_dir)
                print(f"압축 해제 완료: {zip_file.name}")
                
        except Exception as e:
            print(f"압축 해제 실패: {zip_file.name} - {e}")
        finally:
            # 임시 파일 삭제
            if temp_zip.exists():
                temp_zip.unlink()

def process_medical_data(data_dir: str, output_dir: str) -> None:
    """의료 데이터를 처리합니다."""
    print("의료 데이터 처리 중...")
    
    processed_data = []
    
    # 라벨링 데이터에서 의료 관련 파일 찾기
    label_path = Path(data_dir) / "02.라벨링데이터"
    if label_path.exists():
        for file_path in label_path.rglob("*.json"):
            if "필수" in file_path.name or "전문" in file_path.name:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 의료 Q&A 형태로 변환
                        if 'question' in data and 'answer' in data:
                            processed_data.append({
                                'question': data['question'],
                                'answer': data['answer'],
                                'domain': data.get('domain', 'medical'),
                                'type': 'medical_qa'
                            })
                except Exception as e:
                    print(f"파일 처리 실패: {file_path} - {e}")
    
    # 처리된 데이터 저장
    output_path = Path(output_dir) / "medical_data.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"의료 데이터 처리 완료: {len(processed_data)}개 샘플")

def process_conversation_data(data_dir: str, output_dir: str) -> None:
    """대화 데이터를 처리합니다."""
    print("대화 데이터 처리 중...")
    
    processed_data = []
    
    # 라벨링 데이터에서 국어 교과 문제 찾기
    label_path = Path(data_dir) / "02.라벨링데이터"
    if label_path.exists():
        for file_path in label_path.rglob("*.json"):
            if "S2_" in file_path.name:  # 국어 교과 지문형 문제
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 국어 문제를 Q&A 형태로 변환
                        if 'learning_data_info' in data:
                            question = ""
                            answer = ""
                            explanation = ""
                            
                            for item in data['learning_data_info']:
                                if item['class_name'] == '문항':
                                    for info in item['class_info_list']:
                                        if 'text_description' in info:
                                            question = info['text_description']
                                elif item['class_name'] == '정답':
                                    for info in item['class_info_list']:
                                        if 'text_description' in info:
                                            answer = info['text_description']
                                elif item['class_name'] == '해설':
                                    for info in item['class_info_list']:
                                        if 'text_description' in info:
                                            explanation = info['text_description']
                            
                            if question and answer:
                                processed_data.append({
                                    'question': question,
                                    'answer': answer,
                                    'explanation': explanation,
                                    'subject': data.get('raw_data_info', {}).get('subject', '국어'),
                                    'type': 'korean_qa'
                                })
                except Exception as e:
                    print(f"파일 처리 실패: {file_path} - {e}")
    
    # 처리된 데이터 저장
    output_path = Path(output_dir) / "conversation_data.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"대화 데이터 처리 완료: {len(processed_data)}개 샘플")

def process_math_data(data_dir: str, output_dir: str) -> None:
    """수학 문제 데이터를 처리합니다."""
    print("수학 문제 데이터 처리 중...")
    
    processed_data = []
    
    # 라벨링 데이터에서 수학 교과 문제 찾기
    label_path = Path(data_dir) / "02.라벨링데이터"
    if label_path.exists():
        for file_path in label_path.rglob("*.json"):
            if "S3_" in file_path.name:  # 수학 교과 문제
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 수학 문제를 Q&A + 해설 형태로 변환
                        if 'learning_data_info' in data:
                            question = ""
                            answer = ""
                            explanation = ""
                            
                            for item in data['learning_data_info']:
                                if item['class_name'] == '문항(텍스트)':
                                    for info in item['class_info_list']:
                                        if 'text_description' in info:
                                            question = info['text_description']
                                elif item['class_name'] == '정답(텍스트)':
                                    for info in item['class_info_list']:
                                        if 'text_description' in info:
                                            answer = info['text_description']
                                elif item['class_name'] == '해설(텍스트)':
                                    for info in item['class_info_list']:
                                        if 'text_description' in info:
                                            explanation = info['text_description']
                            
                            if question and answer:
                                processed_data.append({
                                    'question': question,
                                    'answer': answer,
                                    'explanation': explanation,
                                    'subject': data.get('raw_data_info', {}).get('subject', '수학'),
                                    'type': 'math_qa'
                                })
                except Exception as e:
                    print(f"파일 처리 실패: {file_path} - {e}")
    
    # 처리된 데이터 저장
    output_path = Path(output_dir) / "math_data.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"수학 데이터 처리 완료: {len(processed_data)}개 샘플")

def process_interview_data(data_dir: str, output_dir: str) -> None:
    """인터뷰 멀티턴 데이터를 처리합니다."""
    print("인터뷰 멀티턴 데이터 처리 중...")
    
    processed_data = []
    
    # 인터뷰 데이터 처리
    interview_path = Path(data_dir) / "interview" / "02.라벨링데이터"
    if interview_path.exists():
        for file_path in interview_path.rglob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 인터뷰 대화를 멀티턴 형태로 변환
                    if 'script' in data:
                        conversation = []
                        for turn in data['script']:
                            if 'question_info' in turn and 'answer_info' in turn:
                                question = turn['question_info']['question_context']
                                # 여러 답변 중 첫 번째 답변 사용
                                if turn['answer_info']:
                                    answer = turn['answer_info'][0]['answer_context']
                                    conversation.append({
                                        'question': question,
                                        'answer': answer
                                    })
                        
                        if conversation:
                            processed_data.append({
                                'conversation': conversation,
                                'topic': data.get('interview_info', {}).get('topic', {}).get('topic', ''),
                                'type': 'interview_multiturn'
                            })
            except Exception as e:
                print(f"파일 처리 실패: {file_path} - {e}")
    
    # 처리된 데이터 저장
    output_path = Path(output_dir) / "interview_data.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"인터뷰 데이터 처리 완료: {len(processed_data)}개 샘플")

def create_data_splits(data_dir: str, output_dir: str, config: Dict[str, Any]) -> None:
    """데이터를 학습/검증/테스트로 분할합니다."""
    print("데이터 분할 중...")
    
    splits_dir = Path(output_dir) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # 각 데이터셋별로 분할
    data_files = [
        "medical_data.json",
        "conversation_data.json", 
        "math_data.json",
        "interview_data.json"
    ]
    
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    test_ratio = config['data']['test_ratio']
    
    for data_file in data_files:
        file_path = Path(data_dir) / data_file
        if not file_path.exists():
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 데이터 셔플
        import random
        random.shuffle(data)
        
        # 분할
        total_size = len(data)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        # 저장
        base_name = data_file.replace('.json', '')
        
        with open(splits_dir / f"{base_name}_train.json", 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(splits_dir / f"{base_name}_val.json", 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        with open(splits_dir / f"{base_name}_test.json", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        print(f"{base_name}: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

def main():
    parser = argparse.ArgumentParser(description="데이터 전처리 스크립트")
    parser.add_argument("--config", default="configs/config.yaml", help="설정 파일 경로")
    parser.add_argument("--data-dir", default=".", help="원본 데이터 디렉토리")
    parser.add_argument("--output-dir", default="data/processed", help="출력 디렉토리")
    parser.add_argument("--extract-only", action="store_true", help="압축 해제만 수행")
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 압축 해제
    extract_zip_files(args.data_dir, args.output_dir)
    
    if args.extract_only:
        print("압축 해제만 완료했습니다.")
        return
    
    # 데이터 처리
    process_medical_data(args.output_dir, args.output_dir)
    process_conversation_data(args.output_dir, args.output_dir)
    process_math_data(args.output_dir, args.output_dir)
    process_interview_data(args.output_dir, args.output_dir)
    
    # 데이터 분할
    create_data_splits(args.output_dir, args.output_dir, config)
    
    print("데이터 전처리가 완료되었습니다!")

if __name__ == "__main__":
    main()
