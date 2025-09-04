#!/usr/bin/env python3
"""
의료 데이터 BOM 문제 해결 스크립트
UTF-8 BOM을 제거하고 JSON 파일을 정상적으로 처리합니다.
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
import argparse
from tqdm import tqdm

def remove_bom_from_file(file_path: str) -> bool:
    """파일에서 BOM을 제거합니다."""
    try:
        # BOM이 있는 파일 읽기
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        
        # BOM 없이 다시 쓰기
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"BOM 제거 실패: {file_path} - {e}")
        return False

def process_medical_files(data_dir: str, output_dir: str) -> None:
    """의료 데이터 파일들을 처리합니다."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 의료 관련 파일들 찾기
    medical_files = []
    for pattern in ["*필수*", "*전문*"]:
        medical_files.extend(data_path.glob(pattern))
    
    print(f"발견된 의료 파일 수: {len(medical_files)}")
    
    processed_data = []
    failed_files = []
    
    for file_path in tqdm(medical_files, desc="의료 데이터 처리 중"):
        try:
            # BOM 제거
            if not remove_bom_from_file(str(file_path)):
                failed_files.append(file_path)
                continue
            
            # JSON 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 데이터가 리스트인 경우
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'question' in item and 'answer' in item:
                        processed_data.append({
                            'question': item['question'],
                            'answer': item['answer'],
                            'source': 'medical',
                            'type': '필수' if '필수' in file_path.name else '전문'
                        })
            # 데이터가 딕셔너리인 경우
            elif isinstance(data, dict):
                if 'question' in data and 'answer' in data:
                    processed_data.append({
                        'question': data['question'],
                        'answer': data['answer'],
                        'source': 'medical',
                        'type': '필수' if '필수' in file_path.name else '전문'
                    })
                    
        except Exception as e:
            print(f"파일 처리 실패: {file_path} - {e}")
            failed_files.append(file_path)
    
    # 처리된 데이터 저장
    output_file = output_path / "medical_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"처리 완료: {len(processed_data)}개 샘플")
    print(f"실패한 파일: {len(failed_files)}개")
    
    if failed_files:
        print("실패한 파일들:")
        for file_path in failed_files[:5]:  # 처음 5개만 출력
            print(f"  - {file_path}")
        if len(failed_files) > 5:
            print(f"  ... 외 {len(failed_files) - 5}개")

def main():
    parser = argparse.ArgumentParser(description="의료 데이터 BOM 문제 해결")
    parser.add_argument("--data-dir", default="data/processed/02.라벨링데이터", 
                       help="의료 데이터가 있는 디렉토리")
    parser.add_argument("--output-dir", default="data/processed", 
                       help="처리된 데이터를 저장할 디렉토리")
    
    args = parser.parse_args()
    
    print("의료 데이터 BOM 문제 해결 시작...")
    process_medical_files(args.data_dir, args.output_dir)
    print("완료!")

if __name__ == "__main__":
    main()
