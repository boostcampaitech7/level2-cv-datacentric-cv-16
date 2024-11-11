import json 
from pathlib import Path
import os
import copy

def open_json(ufo_path: str) -> dict:
    with Path(ufo_path).open(encoding='utf8') as f:
        ufo_data = json.load(f)
    return ufo_data

def save_json(ufo_data: dict, output_path: str) -> None: 
    with open(output_path, 'w') as f:
        json.dump(ufo_data, f, indent=2, ensure_ascii=False)
    print(f"saved json at {output_path}")

def create_json_without_(ufo_path: str, output_path: str, key: str="") -> None:
    ufo_data = open_json(ufo_path)
    total = 0

    img_files = ufo_data['images'].keys()
    for img in img_files:
        items = copy.deepcopy(ufo_data['images'][img]['words'])
        
        for num, word in ufo_data['images'][img]['words'].items():
            if word['transcription'] == key:
                # print(f"erased : {img}, {num}, {word}")
                total += 1
                del items[num]
        
        ufo_data['images'][img]['words'] = items
    
    print(f"from {ufo_path}, total {total} bboxs with transcription='{key}' were erased")
    save_json(ufo_data, output_path)
    
if __name__=="__main__":
    base_data_path = './data'
    base_ufo_path = 'ufo/train.json'
    
    paths = ['vietnamese_receipt',
            'chinese_receipt',
             'japanese_receipt',
             'thai_receipt']
    
    output_file_name = 'erased_empty_transcription.json'
    
    key = ""
    
    for path in paths:
        ufo_path = os.path.join(base_data_path, path, base_ufo_path)
        output_path = ufo_path.replace('train.json', output_file_name)

        create_json_without_(ufo_path=ufo_path, output_path=output_path, key=key)
    