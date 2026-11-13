import os

import cv2
import torch
import torch.nn.functional as F

from model import RGBFFTFusionModel
from preprocessing.video_processing import sample_video_frames
from preprocessing import preprocess_face
from face_detection.face_detector import FaceDetector


# ============================================================
# PATHS
# ============================================================

VIDEO_PATH = os.path.join(
    "test_videos",
    "test.mp4"
)

MODEL_PATH = os.path.join(
    "models",
    "Copy of rgb_fft_fusion_component_pretrained_best.pth"
)


# ============================================================
# REPRESENTATIVE FRAME
# ============================================================

REPRESENTATIVE_FRAME_INDEX = 49


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 65)
print("VIDEO REPRESENTATIVE FRAME INFERENCE TEST")
print("=" * 65)

print(
    f"Device : {device}"
)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(VIDEO_PATH):

    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Checkpoint not found: {MODEL_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading RGB+FFT model...")

model = RGBFFTFusionModel(
    num_classes=2
).to(device)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = (
            checkpoint["model_state_dict"]
        )

    elif "state_dict" in checkpoint:

        state_dict = (
            checkpoint["state_dict"]
        )

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


# Remove DataParallel prefix if present.

state_dict = {
    key.replace("module.", "", 1): value
    for key, value in state_dict.items()
}


# ============================================================
# LOAD MODEL WEIGHTS
# ============================================================

model.load_state_dict(
    state_dict,
    strict=True
)

model.eval()

print(
    "RGB+FFT model loaded successfully."
)


# ============================================================
# LOAD VIDEO
# ============================================================

print("\nLoading video...")

frames, fps, total_frames = sample_video_frames(
    VIDEO_PATH,
    num_frames=10
)

print(
    f"FPS          : {fps:.2f}"
)

print(
    f"Total frames : {total_frames}"
)

print(
    f"Sampled      : {len(frames)}"
)


# ============================================================
# FIND REPRESENTATIVE FRAME
# ============================================================

selected_frame = None


for item in frames:

    if (
        item["index"]
        == REPRESENTATIVE_FRAME_INDEX
    ):

        selected_frame = item["frame"]

        break


if selected_frame is None:

    available_indices = [
        item["index"]
        for item in frames
    ]

    raise ValueError(
        "Representative frame was not found. "
        f"Requested: {REPRESENTATIVE_FRAME_INDEX}. "
        f"Available frames: {available_indices}"
    )


print(
    f"\nRepresentative frame : "
    f"{REPRESENTATIVE_FRAME_INDEX}"
)


# ============================================================
# FACE DETECTION
# ============================================================

print("\nDetecting face...")

detector = FaceDetector()

face_crop = detector.detect_and_crop(
    selected_frame
)


if face_crop is None:

    raise ValueError(
        "No face detected in representative frame."
    )


print(
    f"Face crop shape      : "
    f"{face_crop.shape}"
)


# ============================================================
# PREPROCESS
# ============================================================

print("\nPreprocessing representative face...")

rgb_tensor, fft_tensor = preprocess_face(
    face_crop
)

rgb_tensor = rgb_tensor.to(device)
fft_tensor = fft_tensor.to(device)


print(
    f"RGB tensor           : "
    f"{tuple(rgb_tensor.shape)}"
)

print(
    f"FFT tensor           : "
    f"{tuple(fft_tensor.shape)}"
)


# ============================================================
# MODEL PREDICTION
# ============================================================

print("\nRunning RGB+FFT inference...")

with torch.no_grad():

    outputs = model(
        rgb_tensor,
        fft_tensor
    )

    probabilities = F.softmax(
        outputs,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()


    real_probability = (
        probabilities[0, 0].item()
    )


    deepfake_probability = (
        probabilities[0, 1].item()
    )


# ============================================================
# PREDICTION
# ============================================================

# Current class convention:
#
# 0 = REAL
# 1 = DEEPFAKE

if predicted_class == 0:

    prediction = "REAL"

    confidence = (
        real_probability
    )

else:

    prediction = "DEEPFAKE"

    confidence = (
        deepfake_probability
    )


# ============================================================
# REPRESENTATIVE FRAME RESULT
# ============================================================

print("\n" + "=" * 65)
print("REPRESENTATIVE FRAME PREDICTION")
print("=" * 65)

print(
    f"Frame index         : "
    f"{REPRESENTATIVE_FRAME_INDEX}"
)

print(
    f"Prediction          : "
    f"{prediction}"
)

print(
    f"Confidence          : "
    f"{confidence * 100:.2f}%"
)

print(
    f"REAL probability    : "
    f"{real_probability * 100:.2f}%"
)

print(
    f"DEEPFAKE probability: "
    f"{deepfake_probability * 100:.2f}%"
)

print("=" * 65)


# ============================================================
# GRAD-CAM STATUS
# ============================================================

print()
print("=" * 65)
print("GRAD-CAM STATUS")
print("=" * 65)

print(
    "Grad-CAM was not generated."
)

print(
    "The current RGB+FFT model uses a timm "
    "Xception RGB branch, so the old "
    "model.rgb_branch.conv4.conv1 target "
    "is not valid."
)

print(
    "This test therefore verifies the "
    "representative-frame RGB+FFT prediction only."
)

print("=" * 65)