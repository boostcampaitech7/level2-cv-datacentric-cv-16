# python bbox_to_polygon.py 바꿀json경로 새로출력할json경로

# 예시
# python bbox_to_polygon.py annotations/china2.json edit.json

import json
import argparse
import sys
from pathlib import Path

def convert_bbox_to_polygon(bbox):
    # bbox 형식 [x, y, width, height]를 polygon 형식의 points로 변환
    x, y, width, height = bbox
    points = [
        x, y,                    # 좌상단
        x + width, y,            # 우상단
        x + width, y + height,   # 우하단
        x, y + height            # 좌하단
    ]
    return points

def convert_annotations(input_file, output_file):
    # JSON 파일을 읽어서 bbox 타입을 polygon 타입으로 변환하여 저장
    try:
        # 입력 파일 확인
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_file}")
        
        # JSON 파일 읽기
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        bbox_count = 0
        
        # items 안의 각 항목의 annotations를 처리
        for item in data['items']:
            converted_annotations = []
            
            for ann in item['annotations']:
                if ann['type'] == 'bbox':
                    # bbox를 polygon으로 변환
                    new_ann = ann.copy()
                    new_ann['type'] = 'polygon'
                    new_ann['points'] = convert_bbox_to_polygon(ann['bbox'])
                    del new_ann['bbox']
                    converted_annotations.append(new_ann)
                    bbox_count += 1
                else:
                    # polygon이나 다른 타입은 그대로 유지
                    converted_annotations.append(ann)
            
            # 기존 annotations를 변환된 것으로 교체
            item['annotations'] = converted_annotations
            
        # 결과 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"- 입력 파일: {input_file}")
        print(f"- 출력 파일: {output_file}")
        print(f"- 변환된 bbox 개수: {bbox_count}")
        return True
        
    except Exception as e:
        print(f"에러 발생: {str(e)}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description='CVAT bbox 형식을 polygon 형식으로 변환')
    parser.add_argument('input_file', help='입력 JSON 파일 경로')
    parser.add_argument('output_file', help='출력 JSON 파일 경로')
    
    args = parser.parse_args()
    success = convert_annotations(args.input_file, args.output_file)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()