import cv2
import numpy as np
import torch


IMAGE_SIZE = 224


def create_fft_tensor(image):
    """
    Create the FFT representation used by the RGB+FFT model.

    Input:
        image: OpenCV BGR image

    Output:
        Tensor with shape [1, 1, 224, 224]
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if image is None:
        raise ValueError("Input image is None.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Input image is empty.")

    # --------------------------------------------------------
    # Convert BGR to grayscale
    # --------------------------------------------------------

    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    elif image.ndim == 2:
        gray = image.copy()

    else:
        raise ValueError(
            "Input image must be a BGR image or grayscale image."
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
    # Training used:
    # np.asarray(gray, dtype=np.float32) / 255.0
    # --------------------------------------------------------

    gray = np.asarray(
        gray,
        dtype=np.float32
    ) / 255.0

    # --------------------------------------------------------
    # 2D FFT
    # --------------------------------------------------------

    fft = np.fft.fft2(gray)

    # Move low frequencies to the center
    fft = np.fft.fftshift(fft)

    # --------------------------------------------------------
    # Magnitude spectrum
    # --------------------------------------------------------

    magnitude = np.log1p(
        np.abs(fft)
    )

    # --------------------------------------------------------
    # Min-max normalization
    # --------------------------------------------------------

    min_value = magnitude.min()
    max_value = magnitude.max()

    magnitude = (
        (magnitude - min_value)
        / (max_value - min_value + 1e-8)
    )

    # --------------------------------------------------------
    # Convert to PyTorch tensor
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        magnitude.astype(np.float32)
    )

    # --------------------------------------------------------
    # Add channel dimension
    # [224, 224]
    #      ↓
    # [1, 224, 224]
    # --------------------------------------------------------

    tensor = tensor.unsqueeze(0)

    return tensor