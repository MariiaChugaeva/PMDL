# code/deployment/app/app.py
import streamlit as st
import numpy as np
from PIL import Image
import requests
from streamlit_drawable_canvas import st_canvas
import os
import io
import pandas as pd

API_URL = os.environ.get("DIGIT_API_URL", "http://fastapi:8000/predict")

st.set_page_config(page_title="Digit Recognizer", layout="wide")

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0
if "stroke_width" not in st.session_state:
    st.session_state.stroke_width = 12

# Canvas and controls dimensions
CANVAS_SIZE = 320
SLIDER_WIDTH = CANVAS_SIZE

# CSS: make slider match canvas width
st.markdown(f"<style>div[data-testid='stSlider']{{width:{SLIDER_WIDTH}px !important;}}</style>", unsafe_allow_html=True)

# Main layout: left column for canvas & controls, right column for results
left_col, right_col = st.columns([1, 1])

# Prediction placeholders
prediction_made = False
pred = None
probs = None
preview_img = None
arr = None

with left_col:
    st.subheader("Draw here")

    # Put the canvas directly in the left column (not centered) so it sits flush left
    canvas_result = st_canvas(
        stroke_width=st.session_state.stroke_width,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )

    # Stroke width slider under the canvas (matches canvas width)
    st.slider_label = st.slider("Stroke width", min_value=10, max_value=40, value=st.session_state.stroke_width, key="stroke_width_slider")
    st.session_state.stroke_width = st.slider_label

    # Predict and Reset buttons left-aligned directly under the slider
    btn_left, btn_right = st.columns([1, 1])
    with btn_left:
        if st.button("Predict", key="predict_button"):
            if canvas_result is None or canvas_result.image_data is None:
                st.warning("Nothing drawn. Draw a digit first.")
            else:
                raw_img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA").convert("L")
                # Resize to MNIST format (28x28)
                img_resized = raw_img.resize((28, 28), resample=Image.BILINEAR)
                arr = np.array(img_resized).astype(float)
                arr = 255 - arr   # invert (white bg=0, black stroke=high)
                arr = arr / 255.0 # normalize to [0,1]

                flat = arr.reshape(-1).tolist()

                # Prepare preview image to display in right column
                preview_img = Image.fromarray((arr * 255).astype("uint8"), mode="L").resize((200, 200), Image.NEAREST)

                # Call API
                payload = {"pixels": flat}
                try:
                    resp = requests.post(API_URL, json=payload, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()

                    probs = np.array(data.get("probabilities", []), dtype=float)
                    pred = int(data.get("predicted", -1))
                    prediction_made = True

                except Exception as e:
                    st.error(f"Error calling API at {API_URL}: {e}")

with right_col:
    st.subheader("Preview & Prediction")

    if not prediction_made:
        st.info("Draw a digit on the left canvas, set stroke width, then press Predict.")
    else:
        # Display preview image and metric side-by-side
        top_left, top_right = st.columns([1, 1])
        with top_left:
            st.image(preview_img, caption="Preprocessed 28x28 (model input, enlarged)")
        with top_right:
            top_conf = float(np.max(probs)) if probs is not None and probs.size else 0.0
            st.metric(label="Predicted digit", value=str(pred), delta=f"{top_conf*100:.1f}% confidence")

        # Probability chart full width
        prob_df = pd.DataFrame({"digit": list(range(len(probs))), "prob": probs})
        prob_df = prob_df.set_index("digit")
        st.bar_chart(prob_df)

        # Download and numeric grid in two columns
        d1, d2 = st.columns([1, 1])
        with d1:
            buf = io.BytesIO()
            preview_img.save(buf, format="PNG")
            buf.seek(0)
            st.download_button("Download preprocessed image (PNG)", data=buf, file_name="preprocessed_28x28.png", mime="image/png")
        with d2:
            with st.expander("Show preprocessed 28x28 numeric values"):
                st.table(np.round(arr, 2))

# Footer
st.markdown("---")
st.caption("Model expects 28×28 grayscale input. If predictions are poor, try increasing stroke width or retraining a CNN on augmented samples.")