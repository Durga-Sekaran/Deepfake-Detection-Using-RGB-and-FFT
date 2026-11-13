# ============================================================
# REPRESENTATIVE FRAME SELECTION
# ============================================================

frame_results = [
    {"frame_index": 0,   "deepfake_probability": 0.8063},
    {"frame_index": 49,  "deepfake_probability": 0.8260},
    {"frame_index": 99,  "deepfake_probability": 0.8610},
    {"frame_index": 149, "deepfake_probability": 0.8471},
    {"frame_index": 199, "deepfake_probability": 0.7777},
    {"frame_index": 249, "deepfake_probability": 0.8351},
    {"frame_index": 299, "deepfake_probability": 0.8138},
    {"frame_index": 349, "deepfake_probability": 0.8325},
    {"frame_index": 399, "deepfake_probability": 0.8490},
    {"frame_index": 449, "deepfake_probability": 0.8313},
]


# ============================================================
# VIDEO-LEVEL PROBABILITY
# ============================================================

video_probability = 0.8280


# ============================================================
# VALIDATION
# ============================================================

if not frame_results:
    raise ValueError(
        "No frame results available for representative "
        "frame selection."
    )


if not 0.0 <= video_probability <= 1.0:
    raise ValueError(
        "video_probability must be between 0 and 1."
    )


for result in frame_results:

    if "frame_index" not in result:
        raise KeyError(
            "Each frame result must contain 'frame_index'."
        )

    if "deepfake_probability" not in result:
        raise KeyError(
            "Each frame result must contain "
            "'deepfake_probability'."
        )

    probability = result[
        "deepfake_probability"
    ]

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"Invalid deepfake probability "
            f"for frame {result['frame_index']}: "
            f"{probability}"
        )


# ============================================================
# SELECT REPRESENTATIVE FRAME
# ============================================================

representative = min(
    frame_results,
    key=lambda result: abs(
        result["deepfake_probability"]
        - video_probability
    )
)


# ============================================================
# CALCULATE DIFFERENCE
# ============================================================

probability_difference = abs(
    representative["deepfake_probability"]
    - video_probability
)


# ============================================================
# OUTPUT
# ============================================================

print("=" * 60)
print("REPRESENTATIVE FRAME SELECTION")
print("=" * 60)

print()

print(
    f"Video deepfake probability : "
    f"{video_probability * 100:.2f}%"
)

print(
    f"Representative frame       : "
    f"{representative['frame_index']}"
)

print(
    f"Frame deepfake probability : "
    f"{representative['deepfake_probability'] * 100:.2f}%"
)

print(
    f"Difference                 : "
    f"{probability_difference * 100:.2f}%"
)

print()

print(
    "Selection method           : "
    "Closest frame probability to video probability"
)

print("=" * 60)