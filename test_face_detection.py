import os
import cv2

from preprocessing.video_processing import sample_video_frames
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
# SAMPLE VIDEO FRAMES
# ============================================================

frames, fps, total_frames = sample_video_frames(
    VIDEO_PATH,
    num_frames=10
)


# ============================================================
# LOAD YUNET FACE DETECTOR
# ============================================================

detector = FaceDetector()


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("YUNET FACE DETECTION TEST")
print("=" * 60)

print(
    f"Video path         : {VIDEO_PATH}"
)

print(
    f"Video FPS          : {fps:.2f}"
)

print(
    f"Total video frames : {total_frames}"
)

print(
    f"Sampled frames     : {len(frames)}"
)

print()


# ============================================================
# COUNTERS
# ============================================================

detected_count = 0
no_face_count = 0


# ============================================================
# FACE DETECTION
# ============================================================

for item in frames:

    frame_index = item["index"]
    frame = item["frame"]

    face_crop = detector.detect_and_crop(
        frame
    )

    if face_crop is not None:

        detected_count += 1

        height, width = face_crop.shape[:2]

        print(
            f"Frame {frame_index:4d} -> "
            f"FACE DETECTED "
            f"({width} x {height})"
        )

    else:

        no_face_count += 1

        print(
            f"Frame {frame_index:4d} -> "
            f"NO FACE"
        )


# ============================================================
# SUMMARY
# ============================================================

print()

print("=" * 60)
print("SUMMARY")
print("=" * 60)

print(
    f"Frames with face    : {detected_count}"
)

print(
    f"Frames without face : {no_face_count}"
)

print(
    f"Total checked       : "
    f"{detected_count + no_face_count}"
)

print("=" * 60)


# ============================================================
# FINAL STATUS
# ============================================================

if detected_count > 0:

    print(
        "\nYuNet face detection test PASSED."
    )

else:

    print(
        "\nWARNING: No faces were detected "
        "in any sampled frame."
    )