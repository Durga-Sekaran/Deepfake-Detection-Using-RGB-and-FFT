import os
import torch
import cv2

from model import RGBFFTFusionModel
from face_detection.face_detector import FaceDetector
from preprocessing import preprocess_face


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("RGB + FFT DEEPFAKE DETECTION INFERENCE TEST")
print("=" * 60)
print("Device:", device)


# ============================================================
# IMAGE PATH
# ============================================================

IMAGE_PATH = os.path.join(
    "test_images",
    "test.jpg"
)

print("Testing Image:", IMAGE_PATH)

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

image_bgr = cv2.imread(IMAGE_PATH)

if image_bgr is None:
    raise ValueError(
        f"Could not read image: {IMAGE_PATH}"
    )


# ============================================================
# FACE DETECTION
# ============================================================

print("\nDetecting face...")

detector = FaceDetector()

face_crop = detector.detect_and_crop(
    image_bgr
)

if face_crop is not None:

    print(
        f"Face detected and cropped: "
        f"{face_crop.shape}"
    )

    analysis_image = face_crop

else:

    print(
        "No face detected by detector; "
        "falling back to full image."
    )

    analysis_image = image_bgr


# ============================================================
# PREPROCESSING
# ============================================================

print("\nPreprocessing image...")

rgb_tensor, fft_tensor = preprocess_face(
    analysis_image
)

rgb_tensor = rgb_tensor.to(device)
fft_tensor = fft_tensor.to(device)

print(
    "RGB tensor shape:",
    tuple(rgb_tensor.shape)
)

print(
    "FFT tensor shape:",
    tuple(fft_tensor.shape)
)


# ============================================================
# LOAD MODEL & CHECKPOINT
# ============================================================

CHECKPOINT_PATH = os.path.join(
    "models",
    "Copy of rgb_fft_fusion_component_pretrained_best.pth"
)

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

print("\nLoading trained RGB+FFT model...")

model = RGBFFTFusionModel(
    num_classes=2
).to(device)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)


# Support different checkpoint formats.
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        clean_state_dict = (
            checkpoint["model_state_dict"]
        )

    elif "state_dict" in checkpoint:

        clean_state_dict = (
            checkpoint["state_dict"]
        )

    else:

        clean_state_dict = checkpoint

else:

    clean_state_dict = checkpoint


# Remove DataParallel "module." prefix if present.
clean_state_dict = {
    key.replace("module.", "", 1): value
    for key, value in clean_state_dict.items()
}


# ============================================================
# LOAD WEIGHTS
# ============================================================

model.load_state_dict(
    clean_state_dict,
    strict=True
)

model.eval()

print(
    "Trained RGB+FFT model loaded successfully."
)


# ============================================================
# LOCAL MODEL INFERENCE
# ============================================================

print("\nRunning local model inference...")

with torch.no_grad():

    output = model(
        rgb_tensor,
        fft_tensor
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )


# ============================================================
# EXTRACT PROBABILITIES
# ============================================================

local_real = probabilities[
    0, 0
].item()

local_fake = probabilities[
    0, 1
].item()


local_class = torch.argmax(
    probabilities,
    dim=1
).item()


# Class convention:
# 0 = REAL
# 1 = DEEPFAKE

if local_class == 0:

    local_prediction = "REAL"

else:

    local_prediction = "DEEPFAKE"


local_confidence = max(
    local_real,
    local_fake
)


# ============================================================
# LOCAL MODEL RESULT
# ============================================================

print("\n" + "-" * 60)
print("LOCAL MODEL OUTPUT")
print("-" * 60)

print(
    f"Local REAL     : "
    f"{local_real * 100:.2f}%"
)

print(
    f"Local DEEPFAKE : "
    f"{local_fake * 100:.2f}%"
)

print(
    f"Prediction     : "
    f"{local_prediction}"
)

print(
    f"Confidence     : "
    f"{local_confidence * 100:.2f}%"
)

print("-" * 60)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("FINAL DEEPFAKE DETECTION RESULT")
print("=" * 60)

print(
    f"Prediction : {local_prediction}"
)

print(
    f"Confidence : "
    f"{local_confidence * 100:.2f}%"
)

print(
    f"REAL Probability     : "
    f"{local_real * 100:.2f}%"
)

print(
    f"DEEPFAKE Probability : "
    f"{local_fake * 100:.2f}%"
)

print("=" * 60)