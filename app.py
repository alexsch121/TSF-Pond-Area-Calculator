import streamlit as st

# Patch streamlit_drawable_canvas to avoid removed st.image_to_url
import streamlit_drawable_canvas
import base64
from io import BytesIO
from PIL import Image as PILImage

def _image_to_url(image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(byte_im).decode()

streamlit_drawable_canvas.st_image = type(
    "st_image", (), {"image_to_url": staticmethod(_image_to_url)}
)

import numpy as np
import cv2
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="TSF Pond Area Calculator")

st.title("🟩 TSF Pond Area Calculator")


def pad_to_square(image):
    width, height = image.size
    max_dim = max(width, height)
    delta_w = max_dim - width
    delta_h = max_dim - height
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    return ImageOps.expand(image, padding, fill="white")


# Upload image
uploaded_file = st.file_uploader("Upload a PNG image", type=["png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    squared_image = pad_to_square(image)
    maxsize = 800
    # ✅ Pillow 10+ compatibility
    squared_image.thumbnail((maxsize, maxsize), Image.Resampling.LANCZOS)
    
    st.subheader("Draw a closed shape (polygon) around the area of interest")
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",  # Transparent red fill
        stroke_width=2,
        stroke_color="#FFFFFF",
        background_image=squared_image,
        update_streamlit=True,
        height=squared_image.height,
        width=squared_image.width,
        drawing_mode="polygon",  # polygonal shape
        key="canvas",
    )

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if objects and len(objects) >= 2:
            try:
                areas = []
                for i in range(2):  # only take first 2 polygons
                    obj = objects[i]
                    if obj["type"] == "path" and "path" in obj:
                        path = obj["path"]
                        points = [
                            [pt[1], pt[2]]
                            for pt in path
                            if isinstance(pt, list) and len(pt) >= 3
                        ]
                        if len(points) >= 3:
                            pts = np.array(points, np.int32).reshape((-1, 1, 2))
                            mask = np.zeros((image.height, image.width), dtype=np.uint8)
                            cv2.fillPoly(mask, [pts], 255)
                            area = cv2.countNonZero(mask)
                            areas.append(area)
                        else:
                            st.warning(f"Polygon {i+1} has too few points.")
                            areas.append(0)
                    else:
                        st.warning(f"Object {i+1} is not a valid polygon.")
                        areas.append(0)

                facility_area, pond_area = areas
                st.markdown(f"✅ **BRDA area**: {facility_area:,} pixels²")
                st.markdown(f"✅ **Pond area**: {pond_area:,} pixels²")

                if facility_area > 0:
                    pond_pct = (pond_area / facility_area) * 100
                    st.success(f"🧮 Pond covers **{pond_pct:.2f}%** of the facility area")
                else:
                    st.error("Facility area is zero — cannot calculate percentage.")

            except Exception as e:
                st.error(f"Error calculating areas: {e}")
        else:
            st.info("Please draw at least 2 polygons: first for facility, second for pond.")
