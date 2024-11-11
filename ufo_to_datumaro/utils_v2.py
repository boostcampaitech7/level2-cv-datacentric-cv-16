import json
from pathlib import Path

def read_json(filename):
    """JSON 파일을 읽어오는 함수"""
    with Path(filename).open(encoding='utf8') as f:
        ann = json.load(f)
    return ann

def reshape_points(x): 
    """8개의 좌표값을 4개의 포인트(x,y 쌍)로 재구성"""
    return [x[0:2], x[2:4], x[4:6], x[6:8]]

def wrap_points(points):
    """
    포인트 좌표들을 UFO 형식의 단어(word) 객체로 감싸기
    - transcription: 텍스트 내용 (현재는 빈 문자열)
    - points: 바운딩 박스의 4개 꼭지점 좌표
    - 기타 메타데이터 (언어, 태그, 신뢰도 등)
    """
    return {
        "transcription": "",
        "points": points,
        "orientation": "",
        "language": None,
        "tags": [],
        "confidence": None,
        "illegibility": False,
    }

def wrap_words(words, wh):
    """
    단어들의 정보를 UFO 형식의 이미지 어노테이션으로 감싸기
    wh: (width, height) 튜플로 이미지 크기 정보
    """
    return {
        "paragraphs": {},  # 문단 정보 (현재는 미사용)
        "words": words,    # 단어별 바운딩 박스 정보
        "chars": {},       # 글자 단위 정보 (현재는 미사용)
        "img_w": wh[0],    # 이미지 너비
        "img_h": wh[1],    # 이미지 높이
        "num_patches": None,
        "tags": [],
        "relations": {},
        "annotation_log": {
            "worker": "worker",
            "timestamp": "2099-05-30",
            "tool_version": "",
            "source": None,
        },
        "license_tag": {
            "usability": True,
            "public": False,
            "commercial": True,
            "type": None,
            "holder": "Upstage",
        },
    }

def wrap_images(images): 
    """최종 UFO 형식으로 이미지들의 정보를 감싸기"""
    return {"images": images, "version": "990530", "tags": []}

def get_image_dimensions(image_datum_aro):
    """
    Datumaro 어노테이션에서 이미지 크기 정보 추출
    CVAT export 시 info 필드가 없는 경우를 대비해 여러 방법으로 시도
    
    1. info 필드에서 시도
    2. frame 속성에서 시도
    3. 모두 실패하면 기본값 사용
    """
    try:
        # 먼저 info 필드에서 찾아보기
        return (image_datum_aro["info"]["img_w"], image_datum_aro["info"]["img_h"])
    except (KeyError, TypeError):
        # info가 없으면 frame 속성에서 찾아보기
        try:
            return (
                image_datum_aro["frame"]["width"], 
                image_datum_aro["frame"]["height"]
            )
        except (KeyError, TypeError):
            # 크기 정보를 전혀 찾을 수 없는 경우 기본값 사용
            # 필요에 따라 이 값을 조정하거나 에러를 발생시킬 수 있음
            return (1024, 1024)

def make_img_bboxes_map(image_datum_aro):
    """
    Datumaro 형식의 단일 이미지 어노테이션을 UFO 형식으로 변환
    
    Parameters:
    - image_datum_aro: Datumaro 형식의 이미지 어노테이션
    
    Returns:
    - (파일명, UFO 형식의 이미지 어노테이션) 튜플
    """
    # 파일 경로에서 파일명만 추출 (경로 구분자 처리)
    filename = image_datum_aro["id"].split("/")[-1]
    
    # 이미지 크기 정보 가져오기 (여러 방법 시도)
    dimensions = get_image_dimensions(image_datum_aro)
    
    # 어노테이션별로 word 객체 생성
    words = {
        f"{idx+1:04}": wrap_points(reshape_points(annotation["points"]))
        for idx, annotation in enumerate(image_datum_aro["annotations"])
    }
    
    return (filename, wrap_words(words, dimensions))

def datum_aro_2_ufo_reduced(data):
    """
    Datumaro 형식 전체를 UFO 형식으로 변환
    
    Parameters:
    - data: Datumaro 형식의 전체 데이터
    
    Returns:
    - UFO 형식의 전체 데이터
    """
    return wrap_images(
        dict([
            make_img_bboxes_map(image_datum_aro) 
            for image_datum_aro in data['items']
        ])
    )