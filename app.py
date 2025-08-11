import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

st.title("Draw Polygons on Image")

def pad_to_square(image):
    width, height = image.size
    max_dim = max(width, height)
    delta_w = max_dim - width
    delta_h = max_dim - height
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    return ImageOps.expand(image, padding, fill="white")

uploaded_file = st.file_uploader("Upload PNG image", type=["png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    squared_image = pad_to_square(image)
    squared_image.thumbnail((600, 600), Image.Resampling.LANCZOS)

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="#FFFFFF",
        background_image=squared_image,
        update_streamlit=True,
        height=squared_image.height,
        width=squared_image.width,
        drawing_mode="polygon",
        key="canvas",
    )

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        st.write(f"Number of polygons drawn: {len(objects)}")
        # Process polygons here, e.g. calculate area as before
