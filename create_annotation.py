# pip install easyocr 하세요잉~~~~~~~~~~
# github: https://github.com/JaidedAI/EasyOCR/tree/master

import easyocr
import os
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import time
import json

# # 다 똑같은거 같은데 뭔 차이인지 모르겠음
# # 'en'만 같이 들어갈 수 있고 아니면 하나만 들어가야함 (2개 이하)
# lang_list = ['ja', 'th', 'en', 'vi', 'ch_sim', 'ch_tra']
# reader = easyocr.Reader(['en', 'th'], gpu=True)


# ################## 이미지 존재하는 디렉터리 지정~
# img_dir = './data/thai_receipt/img/train'

# ################## 디렉터리에 넣으려면 디렉터리 존재해야함~
# annotation_path = './hihi.json'

# results = []
# img_names = sorted(os.listdir(img_dir))
# for name in tqdm(img_names):
#     img_path = os.path.join(img_dir, name)

#     img = Image.open(img_path)
#     img = ImageOps.exif_transpose(img)      # 자동 rotation 풀기 함수
#     numpy_array = np.array(img)             # shape 가져오기용(width, height) numpy 변환
#     img_shape=[]
#     for i in numpy_array.shape:
#         img_shape.append(int(i))
    
#     result = reader.readtext(numpy_array)
    
#     for i in range(len(result)):
#         for j in range(len(result[i][0])):
#             result[i][0][j] = [int(result[i][0][j][0]), int(result[i][0][j][1])]
    
#     results.append([result, img_shape])


# # # bbox numpy로 나오는거 int로 바꿔줘야 json에 들어감
# # for i in range(len(results)):
# #     for j in range(len(results[i][0])):
# #         points = results[i][0][j][0]
# #         for k in range(len(points)):
# #             points[k] = [int(points[k][0]), int(points[k][1])]

# # ufo 포맷 슈웃
# data = {'images':{}}
# for j in range(len(img_names)):
#     data['images'][img_names[j]] = {
#         'paragraphs': {}, 
#         'words': {num:{'transcription':results[j][0][num][1], 'confidence':results[j][0][num][2],'points':results[j][0][num][0]} for num in range(len(results[j][0]))}, 
#         'chars': {}, 
#         'img_w': results[j][1][1], 
#         'img_h': results[j][1][0], 
#         'num_patches': None, 
#         'tags': [], 
#         'relations': {}, 
#         'annotation_log': {
#             'worker': 'hobbang_easyocr',
#             'timestamp': str(time.localtime().tm_year)+'-'+str(time.localtime().tm_mon)+'-'+str(time.localtime().tm_mday),
#             'tool_version': '',
#             'source': None
#         }, 
#         'license_tag': {
#             'usability': True,
#             'public': False,
#             'commercial': True,
#             'type': None,
#             'holder': '데이터 주인 적기 !#$@$@!&$*(@&!*(#^&*(@!^$&*(@^#&@!)$(*&@!($)&@*(#)@!&*$'
#         }, 
#     }

# with open(annotation_path, 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent='\t')

class UFOAnnotationWithEasyocr:
    def __init__(self,
                 img_path: str,
                 annotation_path: str,
                 lang_list: list[str]=['en', 'th'],
                 ):
        self.img_path = img_path
        self.annotation_path = annotation_path
        self.lang_list = lang_list
        self.img_names = sorted(os.listdir(img_path))
        self.reader = easyocr.Reader(lang_list, gpu=True)
        
    def create_annotation(self) -> None:
        results = self.run_easyocr()
        
        data = {'images':{}}
        for j in range(len(self.img_names)):
            data['images'][self.img_names[j]] = {
                'paragraphs': {}, 
                'words': {num:{'transcription':results[j][0][num][1], 'confidence':results[j][0][num][2],'points':results[j][0][num][0]} for num in range(len(results[j][0]))}, 
                'chars': {}, 
                'img_w': results[j][1][1], 
                'img_h': results[j][1][0], 
                'num_patches': None, 
                'tags': [], 
                'relations': {}, 
                'annotation_log': {
                    'worker': 'hobbang_easyocr',
                    'timestamp': str(time.localtime().tm_year)+'-'+str(time.localtime().tm_mon)+'-'+str(time.localtime().tm_mday),
                    'tool_version': '',
                    'source': None
                }, 
                'license_tag': {
                    'usability': True,
                    'public': False,
                    'commercial': True,
                    'type': None,
                    'holder': '데이터 주인 적기 !#$@$@!&$*(@&!*(#^&*(@!^$&*(@^#&@!)$(*&@!($)&@*(#)@!&*$'
                }, 
            }

        with open(self.annotation_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent='\t')
             
    def run_easyocr(self) -> list:
        results = []
        for name in tqdm(self.img_names):
            img_path = os.path.join(self.img_path, name)

            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)      # 자동 rotation 풀기 함수
            numpy_array = np.array(img)             # shape 가져오기용(width, height) numpy 변환
            img_shape=[]
            for i in numpy_array.shape:
                img_shape.append(int(i))
            
            result = self.reader.readtext(numpy_array)
            
            for i in range(len(result)):
                for j in range(len(result[i][0])):
                    result[i][0][j] = [int(result[i][0][j][0]), int(result[i][0][j][1])]
            
            results.append([result, img_shape])
        
        return results

if __name__ == "__main__":
    ################## 이미지 존재하는 디렉터리 지정~
    img_dir = './data/thai_receipt/img/train'

    ################## 디렉터리에 넣으려면 디렉터리 존재해야함~
    annotation_path = './hihi.json'

    temp = UFOAnnotationWithEasyocr(img_path=img_dir, annotation_path=annotation_path, lang_list=['en', 'th'])

    temp.create_annotation()