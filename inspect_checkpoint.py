import torch
import os


# ============================================================
# CHECKPOINT PATH
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "Copy of rgb_fft_fusion_component_pretrained_best.pth"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("RGB + FFT CHECKPOINT INSPECTION")
print("=" * 60)

print()
print("File:", MODEL_PATH)


# ============================================================
# CHECK FILE
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Checkpoint not found: {MODEL_PATH}"
    )


file_size_mb = (
    os.path.getsize(MODEL_PATH)
    / (1024 * 1024)
)

print(
    f"File size: {file_size_mb:.2f} MB"
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("Loading checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

print(
    "Checkpoint loaded successfully."
)


# ============================================================
# CHECKPOINT TYPE
# ============================================================

print()
print("Checkpoint type:")
print(type(checkpoint))


# ============================================================
# TOP-LEVEL KEYS
# ============================================================

print()
print("Top-level keys:")

if isinstance(checkpoint, dict):

    for key in checkpoint.keys():

        print("  ", key)

else:

    print(
        "Checkpoint is not a dictionary."
    )


# ============================================================
# DETERMINE STATE DICTIONARY
# ============================================================

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = (
            checkpoint["model_state_dict"]
        )

        print()
        print(
            "Using: model_state_dict"
        )

    elif "state_dict" in checkpoint:

        state_dict = (
            checkpoint["state_dict"]
        )

        print()
        print(
            "Using: state_dict"
        )

    else:

        state_dict = checkpoint

        print()
        print(
            "Using checkpoint directly "
            "as state dictionary."
        )

else:

    state_dict = checkpoint


# ============================================================
# VERIFY STATE DICTIONARY
# ============================================================

if not isinstance(state_dict, dict):

    raise TypeError(
        "The extracted state dictionary "
        "is not a dictionary."
    )


# ============================================================
# REMOVE DataParallel PREFIX
# ============================================================

clean_state_dict = {
    key.replace("module.", "", 1): value
    for key, value in state_dict.items()
}


# ============================================================
# MODEL STATE DICTIONARY INFORMATION
# ============================================================

print()
print("=" * 60)
print("MODEL STATE DICTIONARY")
print("=" * 60)

print()
print(
    "Number of state-dictionary entries:",
    len(clean_state_dict)
)


# ============================================================
# FIRST 30 KEYS
# ============================================================

print()
print("First 30 parameter keys:")

for i, key in enumerate(
    clean_state_dict.keys()
):

    if i >= 30:
        break

    print(
        f"{i + 1:02d}. {key}"
    )


# ============================================================
# BRANCH INFORMATION
# ============================================================

print()
print("=" * 60)
print("BRANCH INFORMATION")
print("=" * 60)


rgb_keys = []
fft_keys = []
classifier_keys = []


for key in clean_state_dict.keys():

    if key.startswith("rgb_branch."):

        rgb_keys.append(key)

    elif key.startswith("fft_branch."):

        fft_keys.append(key)

    elif key.startswith("classifier."):

        classifier_keys.append(key)


# ------------------------------------------------------------
# RGB BRANCH
# ------------------------------------------------------------

print()
print(
    "RGB branch parameters:",
    len(rgb_keys)
)

for key in rgb_keys[:10]:

    print(
        "  ",
        key
    )


if len(rgb_keys) > 10:

    print(
        "  ...",
        len(rgb_keys) - 10,
        "more"
    )


# ------------------------------------------------------------
# FFT BRANCH
# ------------------------------------------------------------

print()
print(
    "FFT branch parameters:",
    len(fft_keys)
)

for key in fft_keys[:10]:

    print(
        "  ",
        key
    )


if len(fft_keys) > 10:

    print(
        "  ...",
        len(fft_keys) - 10,
        "more"
    )


# ------------------------------------------------------------
# CLASSIFIER
# ------------------------------------------------------------

print()
print(
    "Classifier parameters:",
    len(classifier_keys)
)

for key in classifier_keys:

    print(
        "  ",
        key
    )


# ============================================================
# IMPORTANT TENSOR SHAPES
# ============================================================

print()
print("=" * 60)
print("IMPORTANT TENSOR SHAPES")
print("=" * 60)


# Current RGB + FFT architecture:
#
# RGB branch:
#     timm Xception
#     output features = 2048
#
# FFT branch:
#     Conv 1 -> 32
#     Conv 32 -> 64
#     Conv 64 -> 128
#     Conv 128 -> 256
#     output features = 256
#
# Fusion:
#     2048 + 256 = 2304
#
# Classifier:
#     2304 -> 512 -> 2


important_keys = [

    # FFT branch
    "fft_branch.0.weight",
    "fft_branch.1.weight",
    "fft_branch.1.bias",
    "fft_branch.4.weight",
    "fft_branch.5.weight",
    "fft_branch.8.weight",
    "fft_branch.9.weight",
    "fft_branch.12.weight",
    "fft_branch.13.weight",

    # Fusion classifier
    "classifier.0.weight",
    "classifier.0.bias",
    "classifier.3.weight",
    "classifier.3.bias",
]


for key in important_keys:

    if key in clean_state_dict:

        print(
            f"{key}: "
            f"{tuple(clean_state_dict[key].shape)}"
        )

    else:

        print(
            f"{key}: NOT FOUND"
        )


# ============================================================
# RGB BRANCH FEATURE INFORMATION
# ============================================================

print()
print("=" * 60)
print("RGB BRANCH INFORMATION")
print("=" * 60)


rgb_weight_keys = [
    key
    for key in clean_state_dict.keys()
    if key.startswith("rgb_branch.")
    and key.endswith(".weight")
]


if len(rgb_weight_keys) > 0:

    print(
        "RGB branch contains",
        len(rgb_weight_keys),
        "weight tensors."
    )

    print()
    print("First 10 RGB weight tensors:")

    for key in rgb_weight_keys[:10]:

        print(
            f"{key}: "
            f"{tuple(clean_state_dict[key].shape)}"
        )

else:

    print(
        "No RGB branch weight tensors found."
    )


# ============================================================
# CHECK CLASSIFIER DIMENSIONS
# ============================================================

print()
print("=" * 60)
print("CLASSIFIER DIMENSION CHECK")
print("=" * 60)


classifier_input_key = (
    "classifier.0.weight"
)

classifier_output_key = (
    "classifier.3.weight"
)


if classifier_input_key in clean_state_dict:

    shape = tuple(
        clean_state_dict[
            classifier_input_key
        ].shape
    )

    print(
        "classifier.0.weight:",
        shape
    )

    if shape == (512, 2304):

        print(
            "Fusion input dimension: 2304 ✓"
        )

    else:

        print(
            "WARNING: Expected "
            "(512, 2304)"
        )


if classifier_output_key in clean_state_dict:

    shape = tuple(
        clean_state_dict[
            classifier_output_key
        ].shape
    )

    print(
        "classifier.3.weight:",
        shape
    )

    if shape == (2, 512):

        print(
            "Output classes: 2 ✓"
        )

    else:

        print(
            "WARNING: Expected "
            "(2, 512)"
        )


# ============================================================
# TRAINING INFORMATION
# ============================================================

print()
print("=" * 60)
print("TRAINING INFORMATION")
print("=" * 60)


if isinstance(checkpoint, dict):

    found_training_info = False

    for key in [
        "epoch",
        "val_accuracy",
        "val_loss",
        "train_loss",
        "train_accuracy",
        "best_val_accuracy",
        "best_val_loss"
    ]:

        if key in checkpoint:

            print(
                f"{key}: "
                f"{checkpoint[key]}"
            )

            found_training_info = True


    if not found_training_info:

        print(
            "No standard training metrics "
            "found in checkpoint."
        )

else:

    print(
        "No training metadata available."
    )


# ============================================================
# FINAL ARCHITECTURE SUMMARY
# ============================================================

print()
print("=" * 60)
print("EXPECTED CURRENT ARCHITECTURE")
print("=" * 60)

print()
print("RGB Branch : timm Xception")
print("RGB Features: 2048")

print()
print("FFT Branch:")
print("  Conv1 : 1 -> 32")
print("  Conv2 : 32 -> 64")
print("  Conv3 : 64 -> 128")
print("  Conv4 : 128 -> 256")
print("  Features: 256")

print()
print("Fusion:")
print("  2048 + 256 = 2304")

print()
print("Classifier:")
print("  2304 -> 512 -> 2")

print()
print("Classes:")
print("  0 = REAL")
print("  1 = DEEPFAKE")


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)