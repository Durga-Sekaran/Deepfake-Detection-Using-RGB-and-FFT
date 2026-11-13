import os
import torch
import torch.nn.functional as F
from pathlib import Path

from model.model import RGBFFTFusionModel
from preprocessing.video_processing import sample_video_frames
from preprocessing.inference_preprocessing import preprocess_face
from face_detection.face_detector import FaceDetector


# ============================================================
# PATHS
# ============================================================

VIDEO_PATH = Path("test_videos") / "test.mp4"

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "checkpoints"
    / "rgb_fft_fusion_component_pretrained_best.pth"
)


# ============================================================
# SETTINGS
# ============================================================

NUM_SAMPLED_FRAMES = 10

REAL_CLASS = 0
DEEPFAKE_CLASS = 1

FAKE_THRESHOLD = 0.50


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("VIDEO FRAME INFERENCE TEST")
print("=" * 60)

print(f"Device : {device}")


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

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


# ============================================================
# REMOVE DATAPARALLEL PREFIX
# ============================================================

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
# SAMPLE VIDEO FRAMES
# ============================================================

print("\nLoading video...")

frames, fps, total_frames = sample_video_frames(
    VIDEO_PATH,
    num_frames=NUM_SAMPLED_FRAMES
)


if len(frames) == 0:

    raise ValueError(
        "No frames could be sampled from the video."
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
# SHOW SAMPLE INDICES
# ============================================================

print("\nSampled frame indices:")

for item in frames:

    print(
        f"  Frame {item['index']}"
    )


# ============================================================
# FACE DETECTOR
# ============================================================

print("\nLoading YuNet face detector...")

detector = FaceDetector()

print(
    "YuNet loaded successfully."
)


# ============================================================
# STORE RESULTS
# ============================================================

results = []


# ============================================================
# PROCESS EVERY SAMPLED FRAME
# ============================================================

for sample_number, item in enumerate(
    frames,
    start=1
):

    frame = item["frame"]

    frame_index = item["index"]


    print("\n" + "-" * 60)

    print(
        f"Processing sample "
        f"{sample_number}/{len(frames)}"
    )

    print(
        f"Frame index : {frame_index}"
    )


    # ========================================================
    # FACE DETECTION
    # ========================================================

    face_crop = detector.detect_and_crop(
        frame
    )


    if face_crop is None:

        print(
            "No face detected. "
            "Skipping this frame."
        )

        continue


    print(
        f"Face crop shape : "
        f"{face_crop.shape}"
    )


    # ========================================================
    # PREPROCESS
    # ========================================================

    rgb_tensor, fft_tensor = preprocess_face(
        face_crop
    )


    rgb_tensor = rgb_tensor.to(device)

    fft_tensor = fft_tensor.to(device)


    print(
        f"RGB tensor      : "
        f"{tuple(rgb_tensor.shape)}"
    )

    print(
        f"FFT tensor      : "
        f"{tuple(fft_tensor.shape)}"
    )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

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
            probabilities[
                0,
                REAL_CLASS
            ].item()
        )


        deepfake_probability = (
            probabilities[
                0,
                DEEPFAKE_CLASS
            ].item()
        )


    # ========================================================
    # FRAME RESULT
    # ========================================================

    if predicted_class == REAL_CLASS:

        prediction = "REAL"

        confidence = (
            real_probability
        )

    else:

        prediction = "DEEPFAKE"

        confidence = (
            deepfake_probability
        )


    # ========================================================
    # STORE FRAME RESULT
    # ========================================================

    results.append({

        "frame_index": frame_index,

        "frame": frame,

        "face_crop": face_crop,

        "real_probability":
            real_probability,

        "deepfake_probability":
            deepfake_probability,

        "prediction":
            prediction,

        "confidence":
            confidence

    })


    # ========================================================
    # PRINT FRAME RESULT
    # ========================================================

    print(
        f"\nFrame {frame_index} result:"
    )

    print(
        f"Prediction           : "
        f"{prediction}"
    )

    print(
        f"Confidence           : "
        f"{confidence * 100:.2f}%"
    )

    print(
        f"REAL probability     : "
        f"{real_probability * 100:.2f}%"
    )

    print(
        f"DEEPFAKE probability : "
        f"{deepfake_probability * 100:.2f}%"
    )


# ============================================================
# CHECK WHETHER ANY FRAMES WERE USABLE
# ============================================================

if len(results) == 0:

    raise ValueError(
        "No usable frames were found. "
        "A face could not be detected in any "
        "of the sampled frames."
    )


# ============================================================
# VIDEO-LEVEL PROBABILITIES
# ============================================================

mean_real_probability = sum(
    result["real_probability"]
    for result in results
) / len(results)


mean_deepfake_probability = sum(
    result["deepfake_probability"]
    for result in results
) / len(results)


# ============================================================
# VIDEO-LEVEL PREDICTION
# ============================================================

if mean_deepfake_probability >= FAKE_THRESHOLD:

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
# SELECT REPRESENTATIVE FRAME
# ============================================================
#
# This is NOT simply the last frame.
#
# If the video is classified as DEEPFAKE:
#     choose the frame with the highest
#     DEEPFAKE probability.
#
# If the video is classified as REAL:
#     choose the frame with the highest
#     REAL probability.
#
# This frame is the best representative frame
# for later Grad-CAM generation.
# ============================================================

if video_prediction == "DEEPFAKE":

    representative_result = max(
        results,
        key=lambda result:
            result["deepfake_probability"]
    )

else:

    representative_result = max(
        results,
        key=lambda result:
            result["real_probability"]
    )


# ============================================================
# REPRESENTATIVE FRAME INFORMATION
# ============================================================

representative_frame_index = (
    representative_result["frame_index"]
)

representative_prediction = (
    representative_result["prediction"]
)

representative_confidence = (
    representative_result["confidence"]
)

representative_real_probability = (
    representative_result["real_probability"]
)

representative_deepfake_probability = (
    representative_result["deepfake_probability"]
)


# ============================================================
# FINAL VIDEO RESULT
# ============================================================

print("\n\n" + "=" * 60)
print("VIDEO-LEVEL DETECTION RESULT")
print("=" * 60)

print(
    f"Video path             : "
    f"{VIDEO_PATH}"
)

print(
    f"FPS                    : "
    f"{fps:.2f}"
)

print(
    f"Total video frames     : "
    f"{total_frames}"
)

print(
    f"Sampled frames         : "
    f"{len(frames)}"
)

print(
    f"Usable frames          : "
    f"{len(results)}"
)

print(
    f"Mean REAL probability  : "
    f"{mean_real_probability * 100:.2f}%"
)

print(
    f"Mean DEEPFAKE probability: "
    f"{mean_deepfake_probability * 100:.2f}%"
)

print(
    f"\nVIDEO PREDICTION       : "
    f"{video_prediction}"
)

print(
    f"VIDEO CONFIDENCE       : "
    f"{video_confidence * 100:.2f}%"
)


# ============================================================
# REPRESENTATIVE FRAME RESULT
# ============================================================

print("\n" + "-" * 60)
print("REPRESENTATIVE FRAME")
print("-" * 60)

print(
    f"Selected frame index   : "
    f"{representative_frame_index}"
)

print(
    f"Frame prediction       : "
    f"{representative_prediction}"
)

print(
    f"Frame confidence       : "
    f"{representative_confidence * 100:.2f}%"
)

print(
    f"Frame REAL probability : "
    f"{representative_real_probability * 100:.2f}%"
)

print(
    f"Frame DEEPFAKE probability: "
    f"{representative_deepfake_probability * 100:.2f}%"
)


# ============================================================
# GRAD-CAM NOTE
# ============================================================

print("\n" + "-" * 60)

print(
    "This selected representative frame can now be used"
)

print(
    "for Grad-CAM generation in the Streamlit application."
)

print(
    f"Grad-CAM target class : "
    f"{video_prediction}"
)


# ============================================================
# PER-FRAME SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PER-FRAME SUMMARY")
print("=" * 60)

for result in results:

    print(
        f"Frame {result['frame_index']:>4} | "
        f"{result['prediction']:<9} | "
        f"REAL: "
        f"{result['real_probability'] * 100:6.2f}% | "
        f"DEEPFAKE: "
        f"{result['deepfake_probability'] * 100:6.2f}%"
    )


print("=" * 60)
print("VIDEO INFERENCE COMPLETED")
print("=" * 60)