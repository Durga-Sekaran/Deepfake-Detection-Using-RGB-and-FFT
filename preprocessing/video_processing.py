import cv2
import numpy as np


# ============================================================
# VIDEO FRAME SAMPLING
# ============================================================

def sample_video_frames(video_path, num_frames=10):
    """
    Sample evenly spaced frames from a video.

    Input:
        video_path:
            Path to the input video.

        num_frames:
            Number of frames to sample.

    Returns:
        frames:
            List of dictionaries:
            {
                "index": frame_index,
                "frame": BGR frame
            }

        fps:
            Video FPS.

        total_frames:
            Total number of frames in the video.
    """

    # --------------------------------------------------------
    # Validate num_frames
    # --------------------------------------------------------

    if num_frames <= 0:
        raise ValueError(
            "num_frames must be greater than 0."
        )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    try:

        # ----------------------------------------------------
        # Get video information
        # ----------------------------------------------------

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = float(
            cap.get(cv2.CAP_PROP_FPS)
        )

        if total_frames <= 0:
            raise ValueError(
                "Could not determine video frame count."
            )

        # Some videos may not report a valid FPS
        if fps <= 0:
            fps = 0.0

        # ----------------------------------------------------
        # Determine frame indices
        # ----------------------------------------------------

        # If the video contains fewer frames than requested,
        # use every available frame.
        actual_num_frames = min(
            num_frames,
            total_frames
        )

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            actual_num_frames,
            dtype=int
        )

        # Remove duplicate indices
        frame_indices = np.unique(
            frame_indices
        )

        # ----------------------------------------------------
        # Read sampled frames
        # ----------------------------------------------------

        frames = []

        for frame_index in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index)
            )

            success, frame = cap.read()

            if not success:
                continue

            if frame is None:
                continue

            if frame.size == 0:
                continue

            frames.append(
                {
                    "index": int(frame_index),
                    "frame": frame
                }
            )

        # ----------------------------------------------------
        # Make sure at least one frame was read
        # ----------------------------------------------------

        if len(frames) == 0:
            raise ValueError(
                "Could not read any frames from the video."
            )

        return frames, fps, total_frames

    finally:

        # Always release the video
        cap.release()