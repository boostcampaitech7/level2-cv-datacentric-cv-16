import json
import pandas as pd
import streamlit as st
from pathlib import Path

def read_json_from(filename):
    with Path(filename).open(encoding='utf8') as handle:
        data = json.load(handle)
    return data

@st.cache_data
def load_df(json_like_csv_path='./code/data/chinese_receipt/ufo/train.csv'):
    json_data = read_json_from(json_like_csv_path)
    images = json_data['images']

    df = pd.DataFrame()
    image_ids = []
    word_ids = []
    image_heights = []
    image_widths = []
    transcriptions = []
    x1, x2, x3, x4 = [], [], [], []
    y1, y2, y3, y4 = [], [], [], []

    for image_key, image_value in images.items():
        file_name = image_key
        word_ann = image_value['words']
        
        for word_id, word in word_ann.items():
            image_ids.append(file_name)
            word_ids.append(word_id)
            image_heights.append(image_value.get('img_h', ""))
            image_widths.append(image_value.get('img_w', ""))
            transcriptions.append(word.get('transcription', "")) 
            if not word['points']:
                x1.append("")
                y1.append("")
                x2.append("")
                y2.append("")
                x3.append("")
                y3.append("")
                x4.append("")
                y4.append("")
                continue
            x1.append(float(word['points'][0][0]))
            y1.append(float(word['points'][0][1]))
            x2.append(float(word['points'][1][0]))
            y2.append(float(word['points'][1][1]))
            x3.append(float(word['points'][2][0]))
            y3.append(float(word['points'][2][1]))
            x4.append(float(word['points'][3][0]))
            y4.append(float(word['points'][3][1]))

    df['image_id'] = image_ids
    df['word_ids'] = word_ids
    df['img_w'] = image_widths
    df['img_h'] = image_heights
    df['transcriptions'] = transcriptions
    df['x1'] = x1
    df['y1'] = y1
    df['x2'] = x2
    df['y2'] = y2
    df['x3'] = x3
    df['y3'] = y3
    df['x4'] = x4
    df['y4'] = y4

    return df