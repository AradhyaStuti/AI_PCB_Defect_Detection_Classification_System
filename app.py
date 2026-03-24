"""Streamlit frontend for PCB defect detection and classification."""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone

import streamlit as st
from PIL import Image

from config import MAX_UPLOAD_MB, configure_logging
from inference_new import ImageTooLargeError, PCBDefectPipeline

configure_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cached pipeline — survives Streamlit reruns without reloading the model
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def _get_pipeline() -> PCBDefectPipeline:
    return PCBDefectPipeline()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_log_text(anomalies: list[dict]) -> str:
    """Generate a CSV-style detection log."""
    timestamp = datetime.now(tz=timezone.utc).isoformat()
    lines = [f"PCB Defect Detection Log - {timestamp}", ""]

    if not anomalies:
        lines.append("No anomalies detected.")
        return "\n".join(lines)

    lines.append("label,confidence,x1,y1,x2,y2")
    for d in anomalies:
        x1, y1, x2, y2 = d["box"]
        lines.append(f"{d['label']},{d['confidence']:.4f},{x1},{y1},{x2},{y2}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="PCB Defect Detection", layout="wide")
    st.title("PCB Differential Defect Detection")

    # Eagerly warm the model so the first detection isn't slow
    pipeline = _get_pipeline()

    uploaded_file = st.file_uploader(
        "Upload PCB image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is None:
        return

    # --- Input validation ---------------------------------------------------
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_UPLOAD_MB:
        st.error(f"File is {file_size_mb:.1f} MB. Maximum allowed is {MAX_UPLOAD_MB} MB.")
        return

    input_image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input image")
        st.image(input_image, use_container_width=True)

    if not st.button("Run detection"):
        return

    with st.spinner("Running inference..."):
        try:
            result_image, anomalies = pipeline.run(input_image)
        except (ImageTooLargeError, ValueError) as exc:
            st.error(str(exc))
            return
        except Exception:
            logger.exception("Inference failed")
            st.error("Detection failed unexpectedly. Check logs for details.")
            return

    with col2:
        st.subheader("Result")
        st.image(result_image, use_container_width=True)

    if anomalies:
        st.subheader(f"Detected defects ({len(anomalies)})")
        st.dataframe([
            {
                "Label": d["label"],
                "Confidence": round(d["confidence"], 3),
                "x1": d["box"][0],
                "y1": d["box"][1],
                "x2": d["box"][2],
                "y2": d["box"][3],
            }
            for d in anomalies
        ])
    else:
        st.info("No defects detected.")

    st.subheader("Download outputs")

    img_buf = io.BytesIO()
    result_image.save(img_buf, format="PNG")

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="Download result image (PNG)",
            data=img_buf.getvalue(),
            file_name="pcb_result.png",
            mime="image/png",
            key="download-image",
        )
    with dl_col2:
        st.download_button(
            label="Download detection log (TXT)",
            data=_build_log_text(anomalies),
            file_name="pcb_detection_log.txt",
            mime="text/plain",
            key="download-log",
        )


if __name__ == "__main__":
    main()
