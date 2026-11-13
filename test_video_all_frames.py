import os
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
# SETTINGS
# ============================================================

NUM_SAMPLED_FRAMES = 10

# Deepfake decision threshold
DEEPFAKE_THRESHOLD = 0.60


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
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 65)
print("VIDEO - RGB + FFT INFERENCE TEST")
print("=" * 65)

print(
    f"Device : {device}"
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


# Support different checkpoint formats.
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
    num_frames=NUM_SAMPLED_FRAMES
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
# LOAD FACE DETECTOR
# ============================================================

print("\nLoading YuNet face detector...")

detector = FaceDetector()

print(
    "YuNet loaded successfully."
)


# ============================================================
# PROCESS ALL SAMPLED FRAMES
# ============================================================

results = []

print("\n" + "=" * 65)
print("FRAME-BY-FRAME PREDICTIONS")
print("=" * 65)


for frame_number, item in enumerate(
    frames,
    start=1
):

    frame_index = item["index"]
    frame = item["frame"]


    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    face_crop = detector.detect_and_crop(
        frame
    )


    if face_crop is None:

        print(
            f"Frame {frame_index:>4} "
            f"-> NO FACE"
        )

        continue


    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    rgb_tensor, fft_tensor = preprocess_face(
        face_crop
    )

    rgb_tensor = rgb_tensor.to(device)
    fft_tensor = fft_tensor.to(device)


    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            rgb_tensor,
            fft_tensor
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )


    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    real_probability = (
        probabilities[0, 0].item()
    )

    deepfake_probability = (
        probabilities[0, 1].item()
    )


    # --------------------------------------------------------
    # FRAME PREDICTION
    # --------------------------------------------------------

    if (
        deepfake_probability
        >= DEEPFAKE_THRESHOLD
    ):

        prediction = "DEEPFAKE"

        confidence = (
            deepfake_probability
        )

    else:

        prediction = "REAL"

        confidence = (
            real_probability
        )


    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------

    results.append({

        "frame_index": frame_index,

        "real_probability":
            real_probability,

        "deepfake_probability":
            deepfake_probability,

        "prediction":
            prediction,

        "confidence":
            confidence
    })


    # --------------------------------------------------------
    # DISPLAY FRAME RESULT
    # --------------------------------------------------------

    print(
        f"Frame {frame_index:>4} "
        f"-> {prediction:<9} "
        f"| REAL: "
        f"{real_probability * 100:6.2f}% "
        f"| DEEPFAKE: "
        f"{deepfake_probability * 100:6.2f}%"
    )


# ============================================================
# CHECK VALID RESULTS
# ============================================================

if len(results) == 0:

    raise ValueError(
        "No valid frames with detected faces "
        "were found."
    )


# ============================================================
# VIDEO-LEVEL AGGREGATION
# ============================================================

deepfake_probabilities = [

    result["deepfake_probability"]

    for result in results
]


# Mean deepfake probability
mean_deepfake_probability = (

    sum(deepfake_probabilities)
    / len(deepfake_probabilities)

)


# Since the model has two classes:
# REAL + DEEPFAKE = 1

mean_real_probability = (
    1.0
    - mean_deepfake_probability
)


# ============================================================
# VIDEO-LEVEL PREDICTION
# ============================================================

if (
    mean_deepfake_probability
    >= DEEPFAKE_THRESHOLD
):

    video_prediction = "DEEPFAKE"

    video_confidence = (
        mean_deepfake_probability
    )

else:

    video_prediction = "REAL"

    video_confidence = (
        mean_real_probability
    )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 65)
print("VIDEO-LEVEL DETECTION RESULT")
print("=" * 65)

print(
    f"Valid frames             : "
    f"{len(results)}"
)

print(
    f"Mean REAL probability    : "
    f"{mean_real_probability * 100:.2f}%"
)

print(
    f"Mean DEEPFAKE probability: "
    f"{mean_deepfake_probability * 100:.2f}%"
)

print(
    f"Deepfake threshold       : "
    f"{DEEPFAKE_THRESHOLD * 100:.0f}%"
)

print(
    f"Video prediction         : "
    f"{video_prediction}"
)

print(
    f"Video confidence         : "
    f"{video_confidence * 100:.2f}%"
)

print("=" * 65)