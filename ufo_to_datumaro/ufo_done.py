# ufo 포맷 통일 + 확장자 추가 + 이미지 사이즈 매핑하는 코드
# 최하단에 경로 입력하는 곳 있음

import json
from pathlib import Path
import os

def read_json(filename):
    """JSON 파일을 읽어오는 함수"""
    with Path(filename).open(encoding='utf8') as f:
        return json.load(f)

def get_extension_map(train_json):
    """
    train.json에서 각 이미지 이름의 확장자를 추출하여 매핑 생성
    
    Parameters:
    - train_json: train.json 데이터
    
    Returns:
    - dict: {파일이름(확장자 제외): 확장자}
    """
    extension_map = {}
    for full_name in train_json["images"].keys():
        name, ext = os.path.splitext(full_name)
        if ext:  # 확장자가 있는 경우만 저장
            extension_map[name] = ext
    return extension_map

def convert_format(new_test_json, train_json):
    """
    NewTest.json을 train.json 형식으로 변환하는 함수
    파일 이름이 같은 경우 해당 train 데이터의 메타데이터와 확장자를 사용
    
    Parameters:
    - new_test_json: NewTest.json 데이터
    - train_json: train.json 데이터 (메타데이터 참조용)
    
    Returns:
    - 변환된 JSON 데이터
    """
    result = {"images": {}}
    
    # train.json의 이미지 이름에서 확장자 제거한 버전과 확장자 매핑 준비
    train_data_map = {}
    extension_map = {}
    
    for full_name, data in train_json["images"].items():
        name, ext = os.path.splitext(full_name)
        train_data_map[name] = data
        if ext:  # 확장자가 있는 경우만 저장
            extension_map[name] = ext
    
    # 기본 메타데이터 (매칭되는 데이터가 없을 경우 사용)
    default_metadata = {
        "chars": {},
        "img_w": 1024,
        "img_h": 1024,
        "num_patches": None,
        "tags": [],
        "relations": {},
        "annotation_log": {
            "worker": "worker",
            "timestamp": "2024-05-30",
            "tool_version": "",
            "source": None
        },
        "license_tag": {
            "usability": True,
            "public": False,
            "commercial": True,
            "type": None,
            "holder": "Upstage"
        }
    }
    
    # NewTest.json의 각 이미지에 대해 처리
    for image_name, image_data in new_test_json["images"].items():
        # train.json에서 매칭되는 메타데이터 찾기
        train_data = train_data_map.get(image_name)
        metadata = {
            "chars": train_data["chars"],
            "img_w": train_data["img_w"],
            "img_h": train_data["img_h"],
            "num_patches": train_data["num_patches"],
            "tags": train_data["tags"],
            "relations": train_data["relations"],
            "annotation_log": train_data["annotation_log"],
            "license_tag": train_data["license_tag"]
        } if train_data else default_metadata
        
        # 새로운 이미지 데이터 구조 생성
        new_image_data = {
            "paragraphs": {},
            "words": {},
            **metadata
        }
        
        # words 정보 복사 (transcription과 points만 유지)
        for word_id, word_data in image_data["words"].items():
            new_image_data["words"][word_id] = {
                "transcription": word_data["transcription"],
                "points": word_data["points"]
            }
        
        # train.json에서 찾은 확장자 또는 기본값(.jpg) 사용
        ext = extension_map.get(image_name, '.jpg')
        result["images"][f"{image_name}{ext}"] = new_image_data
    
    return result

def save_json(data, filename):
    """변환된 JSON을 파일로 저장하는 함수"""
    with Path(filename).open('w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    # JSON 파일 읽기
    
    # 편집하고 싶은 새로 만든 json
    new = read_json("/data/ephemeral/home/kjh/level2-cv-datacentric-cv-16/ufo_to_datumaro/datumaro2ufo/receipt_china_trans_2.json")
    
    # 포맷을 복사해올 원본 train.json의 경로
    train = read_json("/data/ephemeral/home/kjh/level2-cv-datacentric-cv-16/data/chinese_receipt/ufo/train.json")
    
    # 포맷 변환
    converted = convert_format(new, train)
    
    # 결과 저장
    save_json(converted, "china2.json")

if __name__ == "__main__":
    main()