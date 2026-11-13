import os

from preprocessing.video_processing import sample_video_frames
from preprocessing import preprocess_face
from face_detection.face_detector import FaceDetector


# ============================================================
# VIDEO PATH
# ============================================================

VIDEO_PATH = os.path.join(
    "test_videos",
    "test.mp4"
)


# ============================================================
# CHECK VIDEO
# ============================================================

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )


# ============================================================
# SAMPLE VIDEO
# ============================================================

frames, fps, total_frames = sample_video_frames(
    VIDEO_PATH,
    num_frames=10
)


if len(frames) == 0:
    raise ValueError(
        "No frames could be sampled from the video."
    )


# ============================================================
# LOAD YUNET
# ============================================================

detector = FaceDetector()


# ============================================================
# USE FIRST SAMPLED FRAME
# ============================================================

first_frame = frames[0]["frame"]
frame_index = frames[0]["index"]


# ============================================================
# DETECT AND CROP FACE
# ============================================================

face_crop = detector.detect_and_crop(
    first_frame
)


if face_crop is None:
    raise ValueError(
        f"No face detected in frame {frame_index}."
    )


# ============================================================
# PREPROCESS FACE
# ============================================================

rgb_tensor, fft_tensor = preprocess_face(
    face_crop
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 60)
print("RGB + FFT PREPROCESSING TEST")
print("=" * 60)

print(
    f"Video FPS         : {fps:.2f}"
)

print(
    f"Total video frames: {total_frames}"
)

print(
    f"Sampled frames    : {len(frames)}"
)

print(
    f"Frame index       : {frame_index}"
)

print(
    f"Face crop shape   : {face_crop.shape}"
)

print(
    f"RGB tensor shape  : {tuple(rgb_tensor.shape)}"
)

print(
    f"FFT tensor shape  : {tuple(fft_tensor.shape)}"
)

print(
    f"RGB tensor dtype  : {rgb_tensor.dtype}"
)

print(
    f"FFT tensor dtype  : {fft_tensor.dtype}"
)

print("=" * 60)


# ============================================================
# EXPECTED SHAPE CHECK
# ============================================================

expected_rgb_shape = (
    1, 3, 224, 224
)

expected_fft_shape = (
    1, 1, 224, 224
)


if tuple(rgb_tensor.shape) != expected_rgb_shape:

    raise ValueError(
        f"Unexpected RGB tensor shape: "
        f"{tuple(rgb_tensor.shape)}. "
        f"Expected {expected_rgb_shape}."
    )


if tuple(fft_tensor.shape) != expected_fft_shape:

    raise ValueError(
        f"Unexpected FFT tensor shape: "
        f"{tuple(fft_tensor.shape)}. "
        f"Expected {expected_fft_shape}."
    )


print()
print("RGB tensor shape check : PASSED")
print("FFT tensor shape check : PASSED")
print()
print("Preprocessing test PASSED.")
print("=" * 60)