import pandas as pd
import streamlit as st

import load_json
import visualize_json, visualize_csv

if "train_df" not in st.session_state:
    st.session_state.train_df = pd.DataFrame()
if "output_df" not in st.session_state:
    st.session_state.output_df = pd.DataFrame()

st.sidebar.success("CV16 Forever💕")
st.markdown("<h2 style='text-align: center;'>OCR</h2>", unsafe_allow_html=True)
option = st.sidebar.radio("option", ["visualize images", "visualize csv"])
if option == "visualize images":
    with st.sidebar.form(key="json_form"):
        json_path = st.text_input("json file path")
        submit_button = st.form_submit_button("OK")
        if submit_button:
            try:
                st.session_state.train_df = load_json.load_df(json_path)
                st.sidebar.success("json file load successed :)")
            except Exception as e:
                st.sidebar.error("json file load failed :(")
    if st.session_state.train_df.empty:
        st.stop()
    st.session_state.image_ids = [img_id for img_id in st.session_state.train_df.groupby("image_id")["image_id"].first().tolist()]
    image_count = st.sidebar.slider('Select image count', 1, 4, 1)
    image_index = st.sidebar.slider('Select image index', 0, len(st.session_state.image_ids)-image_count, 0)
    image_index_input = st.sidebar.number_input('Enter image index', min_value=0, max_value=len(st.session_state.image_ids)-image_count, value=image_index, step=image_count)
    if image_index != image_index_input:
        image_index = image_index_input
    image_ids = [st.session_state.image_ids[i] for i in range(image_index, image_index + image_count)]
    with st.sidebar.form(key="image name form"):
        image_name = st.text_input("Enter image name")
        submit_button = st.form_submit_button("OK")
        if submit_button:
            try:
                image_ids = [image_name]
            except Exception as e:
                st.sidebar.error("failed :(")
    visualize_json.show(st.session_state.train_df, image_ids, json_path)

elif option == "visualize csv":
    with st.sidebar.form(key="csv_form"):
        csv_path = st.text_input("csv file path")
        submit_button = st.form_submit_button("OK")
        if submit_button:
            try:
                st.session_state.output_df = load_json.load_df(csv_path)
                st.sidebar.success("csv file load successed :)")
            except Exception as e:
                st.sidebar.error("csv file load failed :(")
    if st.session_state.output_df.empty:
        st.stop()
    st.session_state.image_ids = [img_id for img_id in st.session_state.output_df.groupby("image_id")["image_id"].first().tolist()]
    image_count = st.sidebar.slider('Select image count', 1, 4, 1)
    image_index = st.sidebar.slider('Select image index', 0, len(st.session_state.image_ids)-image_count, 0)
    image_index_input = st.sidebar.number_input('Enter image index', min_value=0, max_value=len(st.session_state.image_ids)-image_count, value=image_index, step=image_count)
    if image_index != image_index_input:
        image_index = image_index_input
    image_ids = [st.session_state.image_ids[i] for i in range(image_index, image_index + image_count)]
    with st.sidebar.form(key="image name form"):
        image_name = st.text_input("Enter image name")
        submit_button = st.form_submit_button("OK")
        if submit_button:
            try:
                image_ids = [image_name]
            except Exception as e:
                st.sidebar.error("failed :(")
    visualize_csv.show(st.session_state.output_df, image_ids)

else:
    pass
