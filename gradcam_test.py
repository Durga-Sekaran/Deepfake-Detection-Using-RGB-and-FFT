import os
import cv2
import torch
import torch.nn.functional as F

from model import RGBFFTFusionModel
from preprocessing import preprocess_face


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "Copy of rgb_fft_fusion_component_pretrained_best.pth"
)

IMAGE_PATH = os.path.join(
    "test_images",
    "test.jpg"
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("Loading RGB + FFT model...")


# ============================================================
# 1. LOAD MODEL
# ============================================================

model = RGBFFTFusionModel(
    num_classes=2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    state_dict = checkpoint["model_state_dict"]

elif (
    isinstance(checkpoint, dict)
    and "state_dict" in checkpoint
):
    state_dict = checkpoint["state_dict"]

else:
    state_dict = checkpoint


clean_state_dict = {
    key.replace("module.", "", 1): value
    for key, value in state_dict.items()
}


model.load_state_dict(
    clean_state_dict,
    strict=True
)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# 2. LOAD IMAGE
# ============================================================

print("Loading image...")

image_bgr = cv2.imread(
    IMAGE_PATH
)

if image_bgr is None:
    raise FileNotFoundError(
        f"Could not read image: {IMAGE_PATH}"
    )


# ============================================================
# 3. PREPROCESSING
# ============================================================

print("Preprocessing image...")

# IMPORTANT:
# preprocess_face() performs the exact RGB + FFT
# preprocessing used by the current inference pipeline.

rgb_tensor, fft_tensor = preprocess_face(
    image_bgr
)

rgb_tensor = rgb_tensor.to(device)
fft_tensor = fft_tensor.to(device)


print(
    "RGB tensor shape:",
    rgb_tensor.shape
)

print(
    "FFT tensor shape:",
    fft_tensor.shape
)


# ============================================================
# 4. PREDICTION
# ============================================================

print("Running prediction...")

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


# Class convention:
# 0 = REAL
# 1 = DEEPFAKE

real_probability = float(
    probabilities[0, 0].item()
)

fake_probability = float(
    probabilities[0, 1].item()
)


if predicted_class == 0:

    label = "REAL"
    confidence = real_probability

else:

    label = "DEEPFAKE"
    confidence = fake_probability


# ============================================================
# 5. DISPLAY RESULT
# ============================================================

print()
print("=" * 50)
print("RGB + FFT PREDICTION")
print("=" * 50)

print(
    "Prediction :", label
)

print(
    "Confidence : {:.2f}%".format(
        confidence * 100
    )
)

print(
    "REAL       : {:.2f}%".format(
        real_probability * 100
    )
)

print(
    "DEEPFAKE   : {:.2f}%".format(
        fake_probability * 100
    )
)

print("=" * 50)


# ============================================================
# IMPORTANT
# ============================================================
#
# Grad-CAM has been removed from this test script.
#
# The current trained model uses the timm Xception RGB branch
# plus the FFT branch. The previous code targeted:
#
#     model.rgb_branch.conv4.conv1
#
# from the older custom RGBFFTModel architecture.
#
# That target is NOT valid for the current RGBFFTFusionModel.
#
# Therefore this file performs only prediction and does not
# generate a Grad-CAM visualization.
#
# ============================================================

print()
print("Prediction test completed successfully.")