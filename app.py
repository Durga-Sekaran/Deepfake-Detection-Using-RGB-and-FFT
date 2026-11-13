import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from textwrap import dedent
import subprocess
import imageio_ffmpeg

from model.model import RGBFFTFusionModel
from preprocessing.inference_preprocessing import preprocess_face
from preprocessing.video_processing import sample_video_frames
from face_detection.face_detector import FaceDetector

# Grad-CAM
from explainability.gradcam import (
    RGBFFTFusionGradCAM,
    create_gradcam_overlay,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "checkpoints"
    / "rgb_fft_fusion_component_pretrained_best.pth"
)

REAL_CLASS = 0
FAKE_CLASS = 1

VIDEO_SAMPLE_COUNT = 10

FAKE_THRESHOLD = 0.50


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Explainable Deepfake Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    :root {
        --navy-950: #061321;
        --navy-900: #081a2d;
        --navy-850: #0b2037;
        --navy-800: #102a46;
        --line: #294968;
        --text: #f5f8fc;
        --muted: #a9bdd1;
        --soft: #d4e0eb;
        --accent: #6da9d8;
    }


    /* --------------------------------------------------------
       MAIN APP
       -------------------------------------------------------- */

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: var(--navy-950);
        color: var(--text);
    }


    [data-testid="stHeader"] {
        background: transparent;
    }


    [data-testid="stMainBlockContainer"] {
        max-width: 1280px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }


    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        color: var(--text) !important;
    }


    p,
    label,
    span,
    [data-testid="stCaptionContainer"] {
        color: var(--soft);
    }


    /* --------------------------------------------------------
       HEADER
       -------------------------------------------------------- */

    .page-header {
        max-width: 820px;
        margin: 0 auto 2rem;
        text-align: center;
    }


    .page-kicker {
        color: var(--accent);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }


    .page-title {
        color: var(--text);
        font-size: clamp(2rem, 4vw, 3.35rem);
        font-weight: 750;
        line-height: 1.08;
        margin: 0;
    }


    .page-subtitle {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.7;
        margin: 0.9rem auto 0;
        max-width: 720px;
    }


    /* --------------------------------------------------------
       UPLOAD CARDS
       -------------------------------------------------------- */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--navy-850);
        border-color: var(--line) !important;
        border-radius: 12px;
    }


    [data-testid="stFileUploaderDropzone"] {
        background: var(--navy-900);
        border: 1px dashed #426784;
        border-radius: 8px;
        min-height: 8.5rem;
    }


    [data-testid="stFileUploaderDropzone"] button,
    .stDownloadButton button {
        background: var(--navy-800);
        color: var(--text);
        border: 1px solid #426784;
        border-radius: 7px;
    }


    [data-testid="stFileUploaderDropzone"] button:hover,
    .stDownloadButton button:hover {
        border-color: var(--accent);
        color: var(--text);
    }


    .upload-heading {
        color: var(--text);
        font-size: 1.08rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }


    .upload-description {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.55;
        min-height: 2.75rem;
        margin-bottom: 0.5rem;
    }


    .format-note {
        color: var(--muted);
        font-size: 0.76rem;
        margin-top: -0.35rem;
    }


    .info-panel {
        background: var(--navy-900);
        border: 1px solid var(--line);
        border-radius: 9px;
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.65;
        margin: 1.25rem 0 2.25rem;
        padding: 1rem 1.15rem;
    }


    /* --------------------------------------------------------
       SECTION TITLES
       -------------------------------------------------------- */

    .section-title {
        color: var(--text);
        font-size: 1.3rem;
        font-weight: 700;
        margin: 2rem 0 0.8rem;
    }


    .column-title {
        color: var(--text);
        font-size: 1rem;
        font-weight: 700;
        margin: 0.45rem 0 0.75rem;
    }


    /* --------------------------------------------------------
       RESULT CARDS
       -------------------------------------------------------- */

    .result-card {
        background: var(--navy-850);
        border: 1px solid var(--line);
        border-radius: 9px;
        margin-bottom: 0.65rem;
        padding: 0.85rem 1rem;
    }


    .result-card:last-child {
        margin-bottom: 0;
    }


    .result-card-title {
        color: var(--muted);
        font-size: 0.75rem;
        font-weight: 650;
        margin-bottom: 0.2rem;
        text-transform: uppercase;
    }


    .result-card-value {
        color: var(--text);
        font-size: 1.35rem;
        font-weight: 730;
    }


    /* --------------------------------------------------------
       PREDICTION EXPLANATION
       -------------------------------------------------------- */

    .explanation-box {
        background: var(--navy-900);
        border: 1px solid var(--line);
        border-radius: 10px;
        color: var(--soft);
        line-height: 1.75;
        padding: 1.2rem 1.3rem;
        margin-top: 0.5rem;
    }


    /* --------------------------------------------------------
       GRAD-CAM IMAGE
       -------------------------------------------------------- */

    .gradcam-note {
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.5;
        margin-top: 0.35rem;
    }


    /* --------------------------------------------------------
       VIDEO
       -------------------------------------------------------- */

    .video-preview-card {
        background: var(--navy-850);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }


    .video-preview-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }


    /* --------------------------------------------------------
       IMAGES
       -------------------------------------------------------- */

    [data-testid="stImage"] img {
        border: 1px solid var(--line);
        border-radius: 9px;
        max-width: 100%;
    }


    .st-key-image_result_input [data-testid="stImage"],
    .st-key-image_result_gradcam [data-testid="stImage"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }


    .st-key-image_result_input [data-testid="stImage"] img,
    .st-key-image_result_gradcam [data-testid="stImage"] img {
        border: 1px solid var(--line);
        border-radius: 9px;
        max-width: 100%;
        max-height: 310px;
        width: auto;
        object-fit: contain;
    }


    [data-testid="stVideo"] video {
        border: 1px solid var(--line);
        border-radius: 9px;
        max-width: 100%;
    }


    /* --------------------------------------------------------
       DATAFRAME (Dark Grey Background, Black Border, White Text)
       -------------------------------------------------------- */

    [data-testid="stDataFrame"] {
        background-color: #1e1e1e !important;
        border: 1px solid #000000 !important;
        border-radius: 9px;
        overflow: hidden;
    }

    [data-testid="stDataFrame"] *, 
    [data-testid="stDataFrame"] th, 
    [data-testid="stDataFrame"] td {
        color: #ffffff !important;
    }


    /* --------------------------------------------------------
       ALERTS
       -------------------------------------------------------- */

    [data-testid="stAlert"] {
        background: var(--navy-850);
        border: 1px solid var(--line);
        border-radius: 9px;
        color: var(--soft);
    }


    hr {
        border-color: var(--line);
    }


    /* --------------------------------------------------------
       MOBILE
       -------------------------------------------------------- */

    @media (max-width: 768px) {

        [data-testid="stMainBlockContainer"] {
            padding: 1.5rem 1rem 3rem;
        }

        .page-header {
            margin-bottom: 1.3rem;
        }

        .page-title {
            font-size: 2rem;
        }

        .upload-description {
            min-height: auto;
        }

        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
        }

        [data-testid="column"] {
            min-width: 100% !important;
            width: 100% !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """<header class="page-header"><div class="page-kicker">Explainable AI Research System</div><h1 class="page-title">Explainable Deepfake Detection</h1><p class="page-subtitle">Upload an image or video to detect potential deepfake content using the trained RGB + FFT fusion model.</p></header>""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = RGBFFTFusionModel(
        num_classes=2
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint


    clean_state_dict = {
        key.replace(
            "module.",
            "",
            1
        ): value

        for key, value in state_dict.items()
    }


    model.load_state_dict(
        clean_state_dict,
        strict=True
    )


    model = model.to(DEVICE)

    model.eval()

    return model


@st.cache_resource
def load_face_detector():

    return FaceDetector()


# ============================================================
# LOAD MODEL + DETECTOR + GRAD-CAM
# ============================================================

try:

    model = load_model()

    detector = load_face_detector()

    gradcam = RGBFFTFusionGradCAM(
        model
    )

except Exception as exc:

    st.error(
        f"Could not load the detection system: {exc}"
    )

    st.stop()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_face(face_bgr):

    """
    Run RGB + FFT prediction on one detected face.

    Class convention:

        0 = REAL
        1 = DEEPFAKE
    """

    rgb_tensor, fft_tensor = preprocess_face(
        face_bgr
    )

    rgb_tensor = rgb_tensor.to(
        DEVICE
    )

    fft_tensor = fft_tensor.to(
        DEVICE
    )


    # Normal prediction does NOT need gradients.
    with torch.no_grad():

        outputs = model(
            rgb_tensor,
            fft_tensor
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )


    real_probability = float(
        probabilities[
            0,
            REAL_CLASS
        ].item()
    )


    fake_probability = float(
        probabilities[
            0,
            FAKE_CLASS
        ].item()
    )


    prediction = (
        "DEEPFAKE"
        if fake_probability >= FAKE_THRESHOLD
        else "REAL"
    )


    confidence = max(
        real_probability,
        fake_probability
    )


    return (
        real_probability,
        fake_probability,
        prediction,
        confidence
    )


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def generate_face_gradcam(
    face_bgr,
    prediction
):

    """
    Generate Grad-CAM for the detected face.

    IMPORTANT:
    Grad-CAM must run with gradients enabled.
    """

    rgb_tensor, fft_tensor = preprocess_face(
        face_bgr
    )


    rgb_tensor = rgb_tensor.to(
        DEVICE
    )

    fft_tensor = fft_tensor.to(
        DEVICE
    )


    target_class = (
        FAKE_CLASS
        if prediction == "DEEPFAKE"
        else REAL_CLASS
    )


    # --------------------------------------------------------
    # GRADIENTS MUST BE ENABLED HERE
    # --------------------------------------------------------

    with torch.enable_grad():

        rgb_tensor.requires_grad_(True)

        cam, _ = gradcam.generate(
            rgb_tensor,
            fft_tensor,
            target_class
        )


    gradcam_image = create_gradcam_overlay(
        face_bgr,
        cam,
        alpha=0.45
    )


    return gradcam_image


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(image_bgr):

    """
    Detect the largest face,
    run prediction,
    and generate Grad-CAM.
    """

    face_crop = detector.detect_and_crop(
        image_bgr
    )


    if face_crop is None:

        return None


    (
        real_probability,
        fake_probability,
        prediction,
        confidence
    ) = predict_face(
        face_crop
    )


    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    gradcam_image = generate_face_gradcam(
        face_crop,
        prediction
    )


    return {

        "face_crop":
            face_crop,

        "real_probability":
            real_probability,

        "fake_probability":
            fake_probability,

        "prediction":
            prediction,

        "confidence":
            confidence,

        "gradcam_image":
            gradcam_image,

    }


# ============================================================
# ANALYSIS TYPE
# ============================================================

input_type = st.radio(

    "Analysis type",

    [
        "Image",
        "Video"
    ],

    horizontal=True,

    label_visibility="collapsed",
)


# ============================================================
# UPLOAD CARDS
# ============================================================

image_upload_col, video_upload_col = (
    st.columns(
        2,
        gap="large"
    )
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

with image_upload_col:

    with st.container(
        border=True
    ):

        st.markdown(
            '<div class="upload-heading">▧ &nbsp; Upload Image</div><div class="upload-description">Upload an image to analyze whether it contains potential deepfake content.</div>',
            unsafe_allow_html=True,
        )


        uploaded_file = st.file_uploader(

            "Choose an image",

            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ],

            key="image_uploader",
        )


        st.markdown(
            '<div class="format-note">Supported: JPG, JPEG, PNG, WEBP</div>',
            unsafe_allow_html=True,
        )


        if uploaded_file is not None:

            st.caption(
                f"Selected: {uploaded_file.name}"
            )


# ============================================================
# VIDEO UPLOAD
# ============================================================

with video_upload_col:

    with st.container(
        border=True
    ):

        st.markdown(
            '<div class="upload-heading">▷ &nbsp; Upload Video</div><div class="upload-description">Upload a video to analyze sampled frames and detect potential deepfake content.</div>',
            unsafe_allow_html=True,
        )


        uploaded_video = st.file_uploader(

            "Choose a video",

            type=[
                "mp4",
                "avi",
                "mov",
                "mkv"
            ],

            key="video_uploader",
        )


        st.markdown(
            '<div class="format-note">Supported: MP4, AVI, MOV, MKV</div>',
            unsafe_allow_html=True,
        )


        if uploaded_video is not None:

            st.caption(
                f"Selected: {uploaded_video.name}"
            )


# ============================================================
# INFO PANEL
# ============================================================

st.markdown(
    '<div class="info-panel">Images and videos are analyzed using an RGB + FFT fusion model. The system combines spatial RGB information with frequency-domain information to identify patterns associated with potential deepfake content.</div>',
    unsafe_allow_html=True,
)


# ============================================================
# IMAGE DETECTION
# ============================================================

if input_type == "Image":

    # --------------------------------------------------------
    # NO IMAGE
    # --------------------------------------------------------

    if uploaded_file is None:

        st.info(
            "Upload an image to begin detection."
        )

        st.stop()


    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    file_bytes = (
        uploaded_file.getvalue()
    )


    max_bytes = (
        200
        *
        1024
        *
        1024
    )


    if len(file_bytes) > max_bytes:

        st.error(
            "Uploaded image exceeds "
            "the 200 MB size limit."
        )

        st.stop()


    image_array = cv2.imdecode(

        np.frombuffer(
            file_bytes,
            dtype=np.uint8
        ),

        cv2.IMREAD_COLOR
    )


    if image_array is None:

        st.error(
            "Could not decode the uploaded image."
        )

        st.stop()


    image_bgr = image_array


    # Original uploaded image.
    pil_image = Image.fromarray(

        cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )
    )


    # --------------------------------------------------------
    # ANALYZE IMAGE
    # --------------------------------------------------------

    with st.spinner(
        "Analyzing image..."
    ):

        try:

            result = analyze_image(
                image_bgr
            )

        except Exception as exc:

            st.error(
                f"Image analysis failed: {exc}"
            )

            st.stop()


    # --------------------------------------------------------
    # NO FACE
    # --------------------------------------------------------

    if result is None:

        st.warning(
            "No face was detected in "
            "the uploaded image."
        )


        st.image(
            pil_image,
            width=420
        )


        st.stop()


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    prediction = result[
        "prediction"
    ]

    confidence = result[
        "confidence"
    ]

    real_probability = result[
        "real_probability"
    ]

    fake_probability = result[
        "fake_probability"
    ]

    gradcam_image = result[
        "gradcam_image"
    ]


    # ========================================================
    # ANALYSIS RESULT
    # ========================================================

    st.markdown(
        '<div class="section-title">Analysis Result</div>',
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # THREE COLUMNS
    #
    # Input Image
    # Grad-CAM Image
    # Detection Result
    # --------------------------------------------------------

    input_col, gradcam_col, result_col = (
        st.columns(
            [1.0, 1.0, 1.15],
            gap="medium"
        )
    )


    # ========================================================
    # INPUT IMAGE
    # ========================================================

    with input_col:

        st.markdown(
            '<div class="column-title">Input Image</div>',
            unsafe_allow_html=True,
        )


        with st.container(key="image_result_input"):

            try:
                st.image(
                    pil_image,
                    width="stretch"
                )
            except TypeError:
                st.image(
                    pil_image,
                    use_container_width=True
                )


    # ========================================================
    # GRAD-CAM IMAGE
    # ========================================================

    with gradcam_col:

        st.markdown(
            '<div class="column-title">Grad-CAM Image</div>',
            unsafe_allow_html=True,
        )


        with st.container(key="image_result_gradcam"):

            try:
                st.image(
                    gradcam_image,
                    width="stretch"
                )
            except TypeError:
                st.image(
                    gradcam_image,
                    use_container_width=True
                )


    # ========================================================
    # DETECTION RESULT
    # ========================================================

    with result_col:

        st.markdown(
            '<div class="column-title">Detection Result</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""<div class="result-card"><div class="result-card-title">Prediction</div><div class="result-card-value">{prediction}</div></div><div class="result-card"><div class="result-card-title">Confidence</div><div class="result-card-value">{confidence * 100:.2f}%</div></div><div class="result-card"><div class="result-card-title">Real Probability</div><div class="result-card-value">{real_probability * 100:.2f}%</div></div><div class="result-card"><div class="result-card-title">Fake Probability</div><div class="result-card-value">{fake_probability * 100:.2f}%</div></div>""",
            unsafe_allow_html=True,
        )


    # ========================================================
    # PREDICTION EXPLANATION
    # ========================================================

    st.markdown(
        '<div class="section-title">Prediction Explanation</div>',
        unsafe_allow_html=True,
    )


    if prediction == "DEEPFAKE":

        explanation = (

            f"The image was classified as DEEPFAKE "
            f"with a deepfake probability of "
            f"{fake_probability * 100:.2f}%. "

            f"The Grad-CAM visualization highlights "
            f"regions of the detected face that contributed "
            f"strongly to the DEEPFAKE prediction. "

            f"These highlighted regions represent "
            f"model evidence and should not by themselves "
            f"be considered proof that the image was manipulated."
        )

    else:

        explanation = (

            f"The image was classified as REAL "
            f"with a real probability of "
            f"{real_probability * 100:.2f}%. "

            f"The Grad-CAM visualization highlights "
            f"regions of the detected face that contributed "
            f"to the REAL prediction. "

            f"These highlighted regions represent "
            f"the visual evidence used by the model "
            f"when making its prediction."
        )


    st.markdown(
        f"""<div class="explanation-box">{explanation}</div>""",
        unsafe_allow_html=True,
    )


# ============================================================
# VIDEO DETECTION
# ============================================================

else:

    # --------------------------------------------------------
    # NO VIDEO
    # --------------------------------------------------------

    if uploaded_video is None:

        st.info(
            "Upload a video to begin detection."
        )

        st.stop()


    # --------------------------------------------------------
    # SAVE AND TRANSCODE VIDEO USING imageio-ffmpeg
    # --------------------------------------------------------

    temp_directory = Path(
        tempfile.gettempdir()
    )

    raw_input_path = (
        temp_directory
        /
        f"raw_input_{uploaded_video.name}"
    )

    with open(raw_input_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    saved_video = (
        temp_directory
        /
        "playable_output_video.mp4"
    )

    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        transcode_cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(raw_input_path),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-c:a",
            "aac",
            str(saved_video),
        ]
        
        subprocess.run(
            transcode_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        # Fallback if transcoding fails
        saved_video = raw_input_path


    # --------------------------------------------------------
    # VIDEO PREVIEW (PLAYABLE CONTAINER)
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Video Preview</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            '<div class="video-preview-label">Uploaded Video</div>',
            unsafe_allow_html=True,
        )

        st.video(str(saved_video))


    # --------------------------------------------------------
    # VIDEO PROCESSING
    # --------------------------------------------------------

    progress_bar = st.progress(0)

    status_text = st.empty()


    try:

        frames, fps, total_frames = (
            sample_video_frames(
                str(raw_input_path),
                num_frames=VIDEO_SAMPLE_COUNT
            )
        )


        frame_results = []


        total_samples = len(
            frames
        )


        for sample_number, item in enumerate(
            frames,
            start=1
        ):

            frame_index = item[
                "index"
            ]

            frame = item[
                "frame"
            ]


            status_text.write(

                f"Analyzing sampled frame "
                f"{sample_number}/{total_samples}..."
            )


            face_crop = (
                detector.detect_and_crop(
                    frame
                )
            )


            if face_crop is not None:

                (
                    real_probability,
                    fake_probability,
                    prediction,
                    confidence
                ) = predict_face(
                    face_crop
                )


                frame_results.append(

                    {
                        "frame_index":
                            frame_index,

                        "frame":
                            frame,

                        "face_crop":
                            face_crop,

                        "real_probability":
                            real_probability,

                        "fake_probability":
                            fake_probability,

                        "prediction":
                            prediction,

                        "confidence":
                            confidence,
                    }
                )


            progress_bar.progress(
                sample_number
                /
                total_samples
            )


        status_text.empty()

        progress_bar.empty()


    except Exception as exc:

        status_text.empty()

        progress_bar.empty()


        st.error(
            f"Could not process the video: {exc}"
        )

        st.stop()


    # --------------------------------------------------------
    # NO FACES
    # --------------------------------------------------------

    if not frame_results:

        st.error(
            "No faces were detected in "
            "the sampled video frames."
        )

        st.stop()


    # --------------------------------------------------------
    # VIDEO PROBABILITIES
    # --------------------------------------------------------

    mean_fake_probability = float(

        np.mean(

            [
                result[
                    "fake_probability"
                ]

                for result
                in frame_results
            ]
        )
    )


    mean_real_probability = (
        1.0
        -
        mean_fake_probability
    )


    video_prediction = (

        "DEEPFAKE"

        if mean_fake_probability
        >= FAKE_THRESHOLD

        else "REAL"
    )


    video_confidence = max(

        mean_real_probability,

        mean_fake_probability
    )


    # --------------------------------------------------------
    # REPRESENTATIVE FRAME
    # --------------------------------------------------------

    representative_result = min(

        frame_results,

        key=lambda result:
            abs(

                result[
                    "fake_probability"
                ]

                -
                mean_fake_probability
            )
    )


    representative_frame = (
        representative_result[
            "frame"
        ]
    )


    representative_frame_index = (
        representative_result[
            "frame_index"
        ]
    )


    # ========================================================
    # VIDEO RESULT
    # ========================================================

    st.markdown(
        '<div class="section-title">Video Analysis Result</div>',
        unsafe_allow_html=True,
    )


    frame_col, video_result_col = (
        st.columns(
            [1, 1],
            gap="large"
        )
    )


    # --------------------------------------------------------
    # REPRESENTATIVE FRAME
    # --------------------------------------------------------

    with frame_col:

        st.markdown(
            '<div class="column-title">Representative Frame</div>',
            unsafe_allow_html=True,
        )


        representative_rgb = (
            cv2.cvtColor(
                representative_frame,
                cv2.COLOR_BGR2RGB
            )
        )


        st.image(
            representative_rgb,
            width=420
        )


        st.caption(

            f"Representative sampled frame: "
            f"{representative_frame_index}"
        )


    # --------------------------------------------------------
    # VIDEO DETECTION RESULT
    # --------------------------------------------------------

    with video_result_col:

        st.markdown(
            '<div class="column-title">Detection Result</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""<div class="result-card"><div class="result-card-title">Video Prediction</div><div class="result-card-value">{video_prediction}</div></div><div class="result-card"><div class="result-card-title">Video Confidence</div><div class="result-card-value">{video_confidence * 100:.2f}%</div></div><div class="result-card"><div class="result-card-title">Real Probability</div><div class="result-card-value">{mean_real_probability * 100:.2f}%</div></div><div class="result-card"><div class="result-card-title">Deepfake Probability</div><div class="result-card-value">{mean_fake_probability * 100:.2f}%</div></div><div class="result-card"><div class="result-card-title">Frames Analyzed</div><div class="result-card-value">{len(frame_results)}</div></div>""",
            unsafe_allow_html=True,
        )


    # ========================================================
    # FRAME ANALYSIS
    # ========================================================

    st.markdown(
        '<div class="section-title">Frame Analysis</div>',
        unsafe_allow_html=True,
    )


    frame_analysis_rows = [

        {
            "Frame":
                result["frame_index"],

            "Real Probability":
                (
                    f"{result['real_probability'] * 100:.2f}%"
                ),

            "Deepfake Probability":
                (
                    f"{result['fake_probability'] * 100:.2f}%"
                ),

            "Prediction":
                result["prediction"],
        }

        for result
        in frame_results
    ]


    frame_analysis_df = pd.DataFrame(
        frame_analysis_rows
    )


    st.dataframe(

        frame_analysis_df,

        use_container_width=True,

        hide_index=True,
    )


    # ========================================================
    # VIDEO PREDICTION EXPLANATION
    # ========================================================

    st.markdown(
        '<div class="section-title">Prediction Explanation</div>',
        unsafe_allow_html=True,
    )


    if video_prediction == "DEEPFAKE":

        video_explanation = (

            f"The video was classified as DEEPFAKE "
            f"with a deepfake probability of "
            f"{mean_fake_probability * 100:.2f}%. "

            f"{len(frame_results)} sampled frames "
            f"contained detectable faces and were analyzed. "

            f"The final video probability is calculated "
            f"from the mean deepfake probability across "
            f"the analyzed frames. "

            f"The representative frame is the sampled "
            f"frame whose prediction is closest to the "
            f"overall video prediction."
        )

    else:

        video_explanation = (

            f"The video was classified as REAL "
            f"with a real probability of "
            f"{mean_real_probability * 100:.2f}%. "

            f"{len(frame_results)} sampled frames "
            f"contained detectable faces and were analyzed. "

            f"The final video player probability is calculated "
            f"from the mean deepfake probability across "
            f"the analyzed frames. "

            f"The representative frame is the sampled "
            f"frame whose prediction is closest to the "
            f"overall video prediction."
        )


    st.markdown(
        f"""<div class="explanation-box">{video_explanation}</div>""",
        unsafe_allow_html=True,
    )


    # ========================================================
    # VIDEO DETAILS
    # ========================================================

    st.caption(

        f"Video details: "
        f"{total_frames} total frames · "
        f"{fps:.2f} FPS · "
        f"{len(frame_results)} frames with "
        f"detected faces analyzed"
    )