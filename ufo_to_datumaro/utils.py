import json
from pathlib import Path
from treescope import display as tdisp

######## Common codes ########

def read_json(filename):
    with Path(filename).open(encoding='utf8') as f:
        ann = json.load(f)
    return ann

######## Datumaro -> UFO codes ########
reshape_points = lambda x: [x[0:2], x[2:4], x[4:6], x[6:8]]

wrap_points = lambda points: {
    "transcription": "",
    "points": points,
    "orientation": "",
    "language": None,
    "tags": [],
    "confidence": None,
    "illegibility": False,
}

wrap_words = lambda words,wh: {
    "paragraphs": {},
    "words": words,
    "chars": {},
    "img_w ": wh[0],
    "img_h": wh[1],
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

wrap_images = lambda images: {"images": images, "version": "990530", "tags": []}

get_img_wh = lambda x: (x["info"]["img_w"], x["info"]["img_h"])

make_img_bboxes_map = lambda image_datum_aro: (
    image_datum_aro["id"].split("/")[-1],
    wrap_words(
        {
            f"{idx+1:04}": wrap_points(reshape_points(annotation["points"]))
            for idx, annotation in enumerate(image_datum_aro["annotations"])
        },
        get_img_wh(image_datum_aro),
    ),
)


datum_aro_2_ufo_reduced = lambda data: wrap_images(
    dict([make_img_bboxes_map(image_datum_aro) for image_datum_aro in data['items']])
)
######## UFO -> Datumaro codes ########

def boxify_polygon(pgn: list[list[float | int]]) -> list[list[list[float | int]]]:
    num_vertices = len(pgn)
    if num_vertices == 4:
        return [[pgn]]
    try:
        return [
            [[pgn[i], pgn[i + 1], pgn[-i - 2], pgn[-i - 1]]]
            for i in range(num_vertices // 2 - 1)
        ]
    except:
        pass
    return []


def flatten_points(boxes: list[list[float | int]]) -> list[float | int]:
    return [coordinate for box in boxes for point in box for coordinate in point]

def extract_flat_points(image: dict):
    boxified_list = [boxify_polygon(v["points"]) for v in image.values()]
    flat_box_list = [flatten_points(v) for vs in boxified_list for v in vs]
    return flat_box_list
