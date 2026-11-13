import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBFFTFusionGradCAM:
    """
    Grad-CAM for the RGB branch of the RGB + FFT fusion model.

    The model prediction uses both RGB and FFT branches.
    Grad-CAM is generated from the convolutional features
    of the RGB Xception branch.
    """

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

        # Automatically find the last Conv2d layer
        # inside the RGB/Xception branch.
        self.target_layer = self._find_last_conv_layer(
            self.model.rgb_branch
        )

        if self.target_layer is None:
            raise RuntimeError(
                "Could not find a Conv2d layer in the RGB branch."
            )

        self.forward_handle = self.target_layer.register_forward_hook(
            self._forward_hook
        )

    def _find_last_conv_layer(self, module):
        """
        Find the last convolutional layer in the RGB branch.
        """

        for layer in reversed(list(module.modules())):
            if isinstance(layer, nn.Conv2d):
                return layer

        return None

    def _forward_hook(self, module, inputs, output):
        """
        Store activations and register a gradient hook.
        """

        self.activations = output

        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        self.gradients = gradient

    def generate(
        self,
        rgb_tensor,
        fft_tensor,
        target_class
    ):
        """
        Generate a Grad-CAM heatmap.

        Args:
            rgb_tensor: [1, 3, 224, 224]
            fft_tensor: [1, 1, 224, 224]
            target_class: 0 for REAL, 1 for DEEPFAKE

        Returns:
            cam: normalized NumPy array of shape 224x224
            probabilities: model probabilities
        """

        self.model.zero_grad(set_to_none=True)

        # Forward pass WITH gradients.
        outputs = self.model(
            rgb_tensor,
            fft_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        # Score corresponding to the predicted class.
        target_score = outputs[:, target_class].sum()

        # Backpropagation.
        target_score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        activations = self.activations.detach()
        gradients = self.gradients.detach()

        # Global average pooling of gradients.
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted combination of feature maps.
        cam = torch.sum(
            weights * activations,
            dim=1,
            keepdim=True
        )

        # Keep only positive contributions.
        cam = F.relu(cam)

        # Resize to input image size.
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0]

        # Normalize to 0-1.
        cam_min = cam.min()
        cam_max = cam.max()

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min + 1e-8
        )

        cam = cam.cpu().numpy()

        return cam, probabilities.detach()


def create_gradcam_overlay(
    face_bgr,
    cam,
    alpha=0.45
):
    """
    Create a Grad-CAM heatmap overlay on the detected face.

    Args:
        face_bgr: detected face in BGR format.
        cam: normalized Grad-CAM heatmap.
        alpha: heatmap transparency.

    Returns:
        RGB image suitable for Streamlit.
    """

    if face_bgr is None:
        raise ValueError("face_bgr is None.")

    if face_bgr.size == 0:
        raise ValueError("face_bgr is empty.")

    height, width = face_bgr.shape[:2]

    cam_resized = cv2.resize(
        cam,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    heatmap = np.uint8(
        255 * cam_resized
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        face_bgr,
        1.0 - alpha,
        heatmap,
        alpha,
        0
    )

    # Convert BGR → RGB for Streamlit.
    overlay_rgb = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )

    return overlay_rgb