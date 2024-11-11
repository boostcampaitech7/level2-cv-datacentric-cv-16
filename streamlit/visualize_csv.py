import cv2
import numpy as np
import streamlit as st

def draw_bbox(image, coordinates, color=(0, 255, 0), thickness=2):
    points = np.array(coordinates).reshape(-1, 2).astype(int)
    for i in range(len(points)):
        start_point = tuple(points[i])
        end_point = tuple(points[(i + 1) % len(points)])
        cv2.line(image, start_point, end_point, color, thickness)

def show(df, image_ids):
    try:
        color = (208, 56, 78)
        nation_dict = {
            'vi': 'vietnamese_receipt',
            'th': 'thai_receipt',
            'zh': 'chinese_receipt',
            'ja': 'japanese_receipt',
        }
        cols = st.columns(len(image_ids))
        
        for i, image_id in enumerate(image_ids):
            image_data = df[df['image_id'] == image_id]
            receipt = nation_dict.get(image_id.split('.')[1])
            image = cv2.imread(f'data/{receipt}/img/test/' + image_id)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for _, row in image_data.iterrows():
                scaled_coordinates = [
                    int(row['x1']), int(row['y1']),
                    int(row['x2']), int(row['y2']),
                    int(row['x3']), int(row['y3']),
                    int(row['x4']), int(row['y4'])
                ]
                draw_bbox(image, scaled_coordinates, color=color, thickness=2)
            with cols[i]:
                st.image(image, caption=image_id, use_column_width=True)
    except Exception as e:
        st.write(e)