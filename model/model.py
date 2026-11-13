import torch
import torch.nn as nn
import timm


# ============================================================
# FFT CNN BRANCH
# ============================================================

class FFTCNN(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(

            # 1 -> 32
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 32 -> 64
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 64 -> 128
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 128 -> 256
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ============================================================
# RGB + FFT FUSION MODEL
# ============================================================

class RGBFFTFusionModel(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        # ----------------------------------------------------
        # RGB BRANCH
        # EXACTLY MATCHES TRAINING ARCHITECTURE
        # ----------------------------------------------------

        self.rgb_branch = timm.create_model(
            "xception",
            pretrained=False,
            num_classes=0
        )

        rgb_features = self.rgb_branch.num_features

        if rgb_features != 2048:
            raise ValueError(
                f"Unexpected Xception feature size: "
                f"{rgb_features}. Expected 2048."
            )

        # ----------------------------------------------------
        # FFT BRANCH
        # ----------------------------------------------------

        fft_model = FFTCNN(
            num_classes=num_classes
        )

        # Use only the feature extractor
        self.fft_branch = fft_model.features

        fft_features = 256

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        fusion_features = rgb_features + fft_features

        if fusion_features != 2304:
            raise ValueError(
                f"Unexpected fusion feature size: "
                f"{fusion_features}. Expected 2304."
            )

        self.classifier = nn.Sequential(

            nn.Linear(
                fusion_features,
                512
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(
                512,
                num_classes
            )
        )

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    def forward(self, rgb, fft):

        # RGB features
        rgb_features = self.rgb_branch(rgb)

        # FFT features
        fft_features = self.fft_branch(fft)

        fft_features = torch.flatten(
            fft_features,
            start_dim=1
        )

        # Combine RGB + FFT
        combined = torch.cat(
            [
                rgb_features,
                fft_features
            ],
            dim=1
        )

        # Classification
        logits = self.classifier(combined)

        return logits