import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

st.set_page_config(page_title="TSF Pond Area Calculator")

st.title("🟩 TSF Pond Area Calculator")


def pad_to_square(image):
    width, height = image.size
    max_dim = max(width, height)
    delta_w = max_dim - width
    delta_h = max_dim - height
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    return ImageOps.expand(image, padding, fill="white")


def resize_image(image, maxsize=600):
    w, h = image.size
    if w > maxsize or h > maxsize:
        scale = min(maxsize / w, maxsize / h)
        new_size = (int(w * scale), int(h * scale))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def plot_image_and_polygons(image, facility_pts, pond_pts):
    fig, ax = plt.subplots()
    ax.imshow(image)
    w, h = image.size

    # Plot facility polygon if exists
    if len(facility_pts) >= 2:
        pts = np.array(facility_pts + [facility_pts[0]])
        ax.plot(pts[:, 0], pts[:, 1], 'g-', linewidth=2, label="Facility")
        ax.scatter(pts[:, 0], pts[:, 1], c='green')
    
    # Plot pond polygon if exists
    if len(pond_pts) >= 2:
        pts = np.array(pond_pts + [pond_pts[0]])
        ax.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=2, label="Pond")
        ax.scatter(pts[:, 0], pts[:, 1], c='red')
    
    ax.set_xlim([0, w])
    ax.set_ylim([h, 0])  # invert y-axis to match image coords (top-left origin)
    ax.axis('off')
    ax.legend()
    st.pyplot(fig)


# Upload image
uploaded_file = st.file_uploader("Upload a PNG image", type=["png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    squared_image = pad_to_square(image)
    squared_image = resize_image(squared_image, maxsize=600)
    w, h = squared_image.size

    st.subheader("Draw polygons by entering points")

    st.markdown("""
    - Enter points as **X** and **Y** coordinates (pixels).
    - First, draw the **Facility polygon** (at least 3 points).
    - Then draw the **Pond polygon** (at least 3 points).
    - Use **Reset polygons** to start over.
    """)

    # Initialize session state variables
    if "facility_points" not in st.session_state:
        st.session_state.facility_points = []
    if "pond_points" not in st.session_state:
        st.session_state.pond_points = []
    if "drawing_phase" not in st.session_state:
        st.session_state.drawing_phase = "facility"  # or "pond" or "done"

    if st.button("Reset polygons"):
        st.session_state.facility_points = []
        st.session_state.pond_points = []
        st.session_state.drawing_phase = "facility"

    col1, col2 = st.columns(2)
    with col1:
        x = st.number_input("X coordinate", min_value=0, max_value=w, value=0, step=1)
    with col2:
        y = st.number_input("Y coordinate", min_value=0, max_value=h, value=0, step=1)

    if st.button("Add point"):
        if st.session_state.drawing_phase == "facility":
            st.session_state.facility_points.append((x, y))
        elif st.session_state.drawing_phase == "pond":
            st.session_state.pond_points.append((x, y))

    plot_image_and_polygons(
        squared_image, st.session_state.facility_points, st.session_state.pond_points
    )

    st.markdown(f"### Current polygon: **{st.session_state.drawing_phase.capitalize()}**")
    st.markdown(f"- Facility points: {len(st.session_state.facility_points)}")
    st.markdown(f"- Pond points: {len(st.session_state.pond_points)}")

    if st.session_state.drawing_phase == "facility":
        if len(st.session_state.facility_points) >= 3:
            if st.button("Start drawing Pond polygon"):
                st.session_state.drawing_phase = "pond"

    elif st.session_state.drawing_phase == "pond":
        if len(st.session_state.pond_points) >= 3:
            if st.button("Finish drawing polygons"):
                st.session_state.drawing_phase = "done"

    if st.session_state.drawing_phase == "done":
        def polygon_area(points, height, width):
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            return cv2.countNonZero(mask)

        try:
            facility_area = polygon_area(st.session_state.facility_points, h, w)
            pond_area = polygon_area(st.session_state.pond_points, h, w)

            st.markdown(f"✅ **Facility area:** {facility_area:,} pixels²")
            st.markdown(f"✅ **Pond area:** {pond_area:,} pixels²")

            if facility_area > 0:
                pond_pct = (pond_area / facility_area) * 100
                st.success(f"🧮 Pond covers **{pond_pct:.2f}%** of the facility area")
            else:
                st.error("Facility area is zero — cannot calculate percentage.")

        except Exception as e:
            st.error(f"Error calculating polygon areas: {e}")
