import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

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
    # Resize while keeping aspect ratio
    w, h = image.size
    if w > maxsize or h > maxsize:
        scale = min(maxsize / w, maxsize / h)
        new_size = (int(w * scale), int(h * scale))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image


# Upload image
uploaded_file = st.file_uploader("Upload a PNG image", type=["png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    squared_image = pad_to_square(image)
    squared_image = resize_image(squared_image, maxsize=600)

    w, h = squared_image.size
    img_np = np.array(squared_image)

    st.subheader("Draw polygons by clicking points on the image")

    # We will collect points for two polygons: facility and pond
    st.markdown(
        """
        Instructions:
        - Click points to draw the **Facility polygon** first (at least 3 points).
        - Then click points to draw the **Pond polygon** (at least 3 points).
        - Use the **Reset polygons** button to start over.
        """
    )

    # Session state to store polygon points
    if "facility_points" not in st.session_state:
        st.session_state.facility_points = []
    if "pond_points" not in st.session_state:
        st.session_state.pond_points = []
    if "drawing_phase" not in st.session_state:
        st.session_state.drawing_phase = "facility"  # or "pond"

    # Button to reset points
    if st.button("Reset polygons"):
        st.session_state.facility_points = []
        st.session_state.pond_points = []
        st.session_state.drawing_phase = "facility"

    # Create Plotly figure with image background
    fig = go.Figure()

    # Add background image
    fig.add_layout_image(
        dict(
            source=squared_image,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=w,
            sizey=h,
            sizing="stretch",
            layer="below",
        )
    )

    # Configure axes to match image coords (origin top-left)
    fig.update_xaxes(
        range=[0, w], showgrid=False, zeroline=False, visible=False, scaleanchor="y"
    )
    fig.update_yaxes(
        range=[h, 0], showgrid=False, zeroline=False, visible=False  # reversed Y axis
    )

    fig.update_layout(
        width=w,
        height=h,
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan",  # disable zoom/box select, you can change as needed
    )

    # Function to convert list of points to Plotly scatter trace
    def polygon_trace(points, color, name):
        if len(points) < 2:
            return None
        # Close polygon by appending first point at end
        pts = points + [points[0]]
        xs, ys = zip(*pts)
        return go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(color=color, size=8),
            name=name,
            fill="toself",
            fillcolor=color.replace("1)", "0.2)"),  # make fill transparent
        )

    # Add facility polygon trace
    if st.session_state.facility_points:
        trace = polygon_trace(st.session_state.facility_points, "rgba(0,128,0,1)", "Facility")
        if trace:
            fig.add_trace(trace)

    # Add pond polygon trace
    if st.session_state.pond_points:
        trace = polygon_trace(st.session_state.pond_points, "rgba(255,0,0,1)", "Pond")
        if trace:
            fig.add_trace(trace)

    # Show instructions about current drawing phase
    st.markdown(
        f"### Current polygon to draw: **{st.session_state.drawing_phase.capitalize()}**"
    )

    # Use plotly_events to capture clicks (left click only)
    clicked_points = plotly_events(fig, click_event=True, hover_event=False)

    # Handle clicks
    if clicked_points:
        x, y = clicked_points[-1]["x"], clicked_points[-1]["y"]
        if st.session_state.drawing_phase == "facility":
            st.session_state.facility_points.append((x, y))
            if len(st.session_state.facility_points) >= 3:
                st.markdown(
                    "**Facility polygon has at least 3 points. You can now switch to drawing pond polygon by clicking the button below.**"
                )
        elif st.session_state.drawing_phase == "pond":
            st.session_state.pond_points.append((x, y))

    # Button to switch polygon drawing phase
    if st.session_state.drawing_phase == "facility":
        if len(st.session_state.facility_points) >= 3:
            if st.button("Start drawing Pond polygon"):
                st.session_state.drawing_phase = "pond"
    else:
        if len(st.session_state.pond_points) >= 3:
            if st.button("Finish drawing polygons"):
                st.session_state.drawing_phase = "done"

    # Once done, calculate polygon areas in pixel² using OpenCV
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
