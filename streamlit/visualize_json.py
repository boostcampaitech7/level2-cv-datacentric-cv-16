import cv2
import numpy as np
import streamlit as st

def draw_bbox(image, coordinates, color=(0, 255, 0), thickness=2):
    points = np.array(coordinates).reshape(-1, 2).astype(int)
    for i in range(len(points)):
        start_point = tuple(points[i])
        end_point = tuple(points[(i + 1) % len(points)])
        cv2.line(image, start_point, end_point, color, thickness)

def show(df, image_ids, json_path):
    try:
        color = (208, 56, 78)
        cols = st.columns(len(image_ids))
        json_path = json_path.split('/')
        mode = "test" if json_path[-1].split('.')[0] == "test" else "train"

        for i, image_id in enumerate(image_ids):
            image_data = df[df['image_id'] == image_id]
            base_path = '/'.join(json_path[:-2]) + f"/img/{mode}/"
            image = cv2.imread(base_path + image_id)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            display_w, display_h = image.shape[1], image.shape[0]
            for _, row in image_data.iterrows():
                if mode == "train":
                    img_w, img_h = row['img_w'], row['img_h']
                    scaled_coordinates = [
                        int(row['x1'] * display_w / img_w), int(row['y1'] * display_h / img_h),
                        int(row['x2'] * display_w / img_w), int(row['y2'] * display_h / img_h),
                        int(row['x3'] * display_w / img_w), int(row['y3'] * display_h / img_h),
                        int(row['x4'] * display_w / img_w), int(row['y4'] * display_h / img_h)
                    ]
                    draw_bbox(image, scaled_coordinates, color=color, thickness=2)
        with cols[i]:
            st.image(image, caption=image_id, use_column_width=True)
    except Exception as e:
        st.write(e)
        