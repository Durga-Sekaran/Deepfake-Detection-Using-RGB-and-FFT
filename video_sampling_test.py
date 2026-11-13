from preprocessing.video_processing import sample_video_frames


# --------------------------------------------------
# VIDEO PATH
# --------------------------------------------------

VIDEO_PATH = "test_videos/test.mp4"


# --------------------------------------------------
# SAMPLE VIDEO FRAMES
# --------------------------------------------------

frames, fps, total_frames = sample_video_frames(
    VIDEO_PATH,
    num_frames=10
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("=" * 50)
print("VIDEO FRAME SAMPLING TEST")
print("=" * 50)

print(f"Video path    : {VIDEO_PATH}")
print(f"FPS           : {fps:.2f}")
print(f"Total frames  : {total_frames}")
print(f"Sampled frames: {len(frames)}")

print("\nSelected frame indices:")

for item in frames:
    print(item["index"])

print("=" * 50)