import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


IMAGE_SIZE = 224


# ============================================================
# RGB PREPROCESSING
# ============================================================

# Exactly matches the RGB preprocessing used during training
rgb_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_rgb(face_bgr):
    """
    Convert a BGR face crop from OpenCV
    into the RGB tensor expected by the model.

    Input:
        face_bgr: NumPy array in BGR format

    Output:
        Tensor with shape [1, 3, 224, 224]
    """

    if face_bgr is None:
        raise ValueError("face_bgr is None.")

    if not isinstance(face_bgr, np.ndarray):
        raise TypeError(
            "face_bgr must be a NumPy array."
        )

    if face_bgr.size == 0:
        raise ValueError("face_bgr is empty.")

    # BGR -> RGB
    face_rgb = cv2.cvtColor(
        face_bgr,
        cv2.COLOR_BGR2RGB
    )

    # NumPy -> PIL
    image = Image.fromarray(face_rgb)

    # Apply training transform
    rgb_tensor = rgb_transform(image)

    # Add batch dimension
    return rgb_tensor.unsqueeze(0)


# ============================================================
# FFT PREPROCESSING
# ============================================================

def preprocess_fft(face_bgr):
    """
    Create the FFT representation used during
    RGB+FFT model training.

    Input:
        face_bgr: NumPy array in BGR format

    Output:
        Tensor with shape [1, 1, 224, 224]
    """

    if face_bgr is None:
        raise ValueError("face_bgr is None.")

    if not isinstance(face_bgr, np.ndarray):
        raise TypeError(
            "face_bgr must be a NumPy array."
        )

    if face_bgr.size == 0:
        raise ValueError("face_bgr is empty.")

    # --------------------------------------------------------
    # BGR -> Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        face_bgr,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Resize to 224 x 224
    # --------------------------------------------------------

    gray = cv2.resize(
        gray,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Match training preprocessing exactly.
    #
    # Training used grayscale values in [0, 1]
    # before calculating FFT.
    # --------------------------------------------------------

    gray = np.asarray(
        gray,
        dtype=np.float32
    ) / 255.0

    # --------------------------------------------------------
    # 2D FFT
    # --------------------------------------------------------

    fft = np.fft.fft2(gray)

    # Shift low frequencies to the center
    fft_shifted = np.fft.fftshift(fft)

    # --------------------------------------------------------
    # Magnitude
    # --------------------------------------------------------

    magnitude = np.abs(
        fft_shifted
    )

    # --------------------------------------------------------
    # Log transform
    # --------------------------------------------------------

    log_magnitude = np.log1p(
        magnitude
    )

    # --------------------------------------------------------
    # Min-max normalization
    #
    # +1e-8 matches training preprocessing
    # --------------------------------------------------------

    min_value = log_magnitude.min()
    max_value = log_magnitude.max()

    fft_normalized = (
        (log_magnitude - min_value)
        / (max_value - min_value + 1e-8)
    )

    # --------------------------------------------------------
    # NumPy -> PyTorch
    # --------------------------------------------------------

    fft_tensor = torch.from_numpy(
        fft_normalized.astype(np.float32)
    )

    # [224, 224]
    #     ↓
    # [1, 224, 224]
    fft_tensor = fft_tensor.unsqueeze(0)

    # [1, 224, 224]
    #     ↓
    # [1, 1, 224, 224]
    fft_tensor = fft_tensor.unsqueeze(0)

    return fft_tensor


# ============================================================
# COMPLETE FACE PREPROCESSING
# ============================================================

def preprocess_face(face_bgr):
    """
    Prepare one detected face for the RGB+FFT model.

    Input:
        face_bgr: OpenCV BGR face crop

    Returns:
        rgb_tensor:
            [1, 3, 224, 224]

        fft_tensor:
            [1, 1, 224, 224]
    """

    rgb_tensor = preprocess_rgb(
        face_bgr
    )

    fft_tensor = preprocess_fft(
        face_bgr
    )

    return rgb_tensor, fft_tensor