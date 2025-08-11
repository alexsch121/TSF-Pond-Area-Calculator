import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import streamlit_image_annotation.Detection as detection

st.set_page_config(page_title="TSF Pond Area Calculator")
st.title("🟩 TSF Pond Area Calculator")


def pad_to_square(image):
    """Pad an image to make it square with white background."""
    width, height = image.size
    max_dim = max(width, height)
    delta_w = max_dim - width
    delta_h = max_dim - height
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - (delta_w // 2),
        delta_h - (delta_h // 2),
    )
    return ImageOps.expand(image, padding, fill="white")


# Upload image
uploaded_file = st.file_uploader("Upload a PNG image", type=["png"])

if uploaded_file:
    # Load and resize image
    image = Image.open(uploaded_file)
    squared_image = pad_to_square(image)
    maxsize = 600
    squared_image.thumbnail((maxsize, maxsize), Image.Resampling.LANCZOS)

    st.subheader("Draw polygons around the BRDA and the pond (in that order)")

    # Convert image to numpy array for annotation
    bg_image_np = np.array(squared_image.convert("RGB"))

    # Annotation widget
    annotations = detection(
        bg_image_np,
        stroke_width=2,
        stroke_color="red",
        fill_color="rgba(255,0,0,0.3)",
        key="annotation",
    )

    if annotations:
        areas = []
        for i, poly in enumerate(annotations):
            if "points" in poly and len(poly["points"]) >= 3:
                # Convert points to OpenCV format
                pts = np.array(poly["points"], np.int32).reshape((-1, 1, 2))
                mask = np.zeros(
                    (squared_image.height, squared_image.width), dtype=np.uint8
                )
                cv2.fillPoly(mask, [pts], 255)
                area = cv2.countNonZero(mask)
                areas.append(area)
            else:
                st.warning(f"Polygon {i+1} has too few points.")
                areas.append(0)

        if len(areas) >= 2:
            facility_area, pond_area = areas[:2]
            st.markdown(f"✅ **BRDA area**: {facility_area:,} pixels²")
            st.markdown(f"✅ **Pond area**: {pond_area:,} pixels²")

            if facility_area > 0:
                pond_pct = (pond_area / facility_area) * 100
                st.success(f"🧮 Pond covers **{pond_pct:.2f}%** of the facility area")
            else:
                st.error("Facility area is zero — cannot calculate percentage.")
        else:
            st.info("Please draw at least 2 polygons: first for BRDA, second for pond.")
