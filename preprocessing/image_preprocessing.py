from PIL import Image
from torchvision import transforms


IMAGE_SIZE = 224


# ============================================================
# RGB TRANSFORM
# ============================================================

rgb_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# RGB PREPROCESSING
# ============================================================

def preprocess_rgb(image):
    """
    Convert an image into the RGB tensor used by
    the RGB branch of the RGB+FFT model.

    Input:
        image:
            PIL.Image

    Output:
        Tensor with shape:
            [3, 224, 224]
    """

    if image is None:
        raise ValueError("Input image is None.")

    if not isinstance(image, Image.Image):
        raise TypeError(
            "preprocess_rgb() expects a PIL.Image."
        )

    # --------------------------------------------------------
    # Make sure the image has exactly 3 RGB channels
    # --------------------------------------------------------

    image = image.convert("RGB")

    # --------------------------------------------------------
    # Apply the SAME transform used during training
    # --------------------------------------------------------

    tensor = rgb_transform(image)

    return tensor