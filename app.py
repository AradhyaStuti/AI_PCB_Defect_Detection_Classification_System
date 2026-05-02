"""Streamlit UI for PCB defect detection."""

import io
import logging
from datetime import datetime, timezone

import streamlit as st
from PIL import Image

from config import MAX_UPLOAD_MB, configure_logging
from inference_new import ImageTooLargeError, PCBDefectPipeline

configure_logging()
logger = logging.getLogger(__name__)


# cache_resource so we don't reload the model on every rerun. Streamlit
# re-executes the script top to bottom every time the user clicks something.
@st.cache_resource(show_spinner="Loading model...")
def get_pipeline() -> PCBDefectPipeline:
    return PCBDefectPipeline()


def build_log_text(anomalies: list[dict]) -> str:
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


def main() -> None:
    st.set_page_config(page_title="PCB Defect Detection", layout="wide")
    st.title("PCB Differential Defect Detection")

    pipeline = get_pipeline()

    uploaded = st.file_uploader("Upload PCB image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return

    size_mb = uploaded.size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        st.error(f"File is {size_mb:.1f} MB. Max allowed is {MAX_UPLOAD_MB} MB.")
        return

    input_image = Image.open(uploaded).convert("RGB")

    col_in, col_out = st.columns(2)
    with col_in:
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
            st.error("Detection failed. Check logs for details.")
            return

    with col_out:
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

    st.subheader("Download")

    img_buf = io.BytesIO()
    result_image.save(img_buf, format="PNG")

    dl_img, dl_log = st.columns(2)
    with dl_img:
        st.download_button(
            "Download result image (PNG)",
            data=img_buf.getvalue(),
            file_name="pcb_result.png",
            mime="image/png",
            key="download-image",
        )
    with dl_log:
        st.download_button(
            "Download detection log (TXT)",
            data=build_log_text(anomalies),
            file_name="pcb_detection_log.txt",
            mime="text/plain",
            key="download-log",
        )


if __name__ == "__main__":
    main()
