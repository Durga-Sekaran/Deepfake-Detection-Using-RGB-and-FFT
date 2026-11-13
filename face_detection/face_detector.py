import cv2
import os


# ============================================================
# YUNET MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "face_detection_yunet_2023mar.onnx"
)


# ============================================================
# SETTINGS
# ============================================================

FACE_MARGIN = 0.20

MAX_DETECTION_SIZE = 640


# ============================================================
# FACE DETECTOR
# ============================================================

class FaceDetector:

    def __init__(self):

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"YuNet model not found: {MODEL_PATH}"
            )

        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError(
                "cv2.FaceDetectorYN is not available. "
                "Please install a compatible OpenCV version."
            )

        self.detector = cv2.FaceDetectorYN.create(
            MODEL_PATH,
            "",
            (320, 320),
            0.6,
            0.3,
            5000
        )


    # ========================================================
    # DETECT AND CROP LARGEST FACE
    # ========================================================

    def detect_and_crop(
        self,
        frame,
        margin=FACE_MARGIN
    ):
        """
        Detect the largest face in a BGR frame and return
        a cropped face with the same 20% margin used during
        the training preprocessing pipeline.

        Input:
            frame: OpenCV BGR image

        Output:
            Cropped face as BGR NumPy array,
            or None if no usable face is detected.
        """

        if frame is None:
            return None

        if frame.size == 0:
            return None

        if len(frame.shape) != 3:
            return None

        original_height, original_width = frame.shape[:2]

        if original_height <= 0 or original_width <= 0:
            return None


        # ====================================================
        # RESIZE IMAGE FOR DETECTION
        # ====================================================
        #
        # Same principle used during Day 7 training:
        # maximum detection dimension = 640.
        #

        scale = min(
            1.0,
            MAX_DETECTION_SIZE /
            max(original_width, original_height)
        )

        detection_width = max(
            1,
            int(original_width * scale)
        )

        detection_height = max(
            1,
            int(original_height * scale)
        )

        if scale != 1.0:

            detection_frame = cv2.resize(
                frame,
                (
                    detection_width,
                    detection_height
                ),
                interpolation=cv2.INTER_AREA
            )

        else:

            detection_frame = frame


        # ====================================================
        # YUNET DETECTION
        # ====================================================

        self.detector.setInputSize(
            (
                detection_width,
                detection_height
            )
        )

        _, faces = self.detector.detect(
            detection_frame
        )


        # ====================================================
        # NO FACE
        # ====================================================

        if faces is None or len(faces) == 0:
            return None


        # ====================================================
        # SELECT LARGEST FACE
        # ====================================================

        largest_face = max(
            faces,
            key=lambda face: face[2] * face[3]
        )

        x, y, w, h = largest_face[:4]


        # ====================================================
        # CONVERT COORDINATES BACK TO ORIGINAL FRAME
        # ====================================================

        if scale != 1.0:

            x = x / scale
            y = y / scale
            w = w / scale
            h = h / scale


        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)


        # ====================================================
        # ADD 20% FACE MARGIN
        # ====================================================

        margin_x = w * margin
        margin_y = h * margin

        x1 = int(
            max(
                0,
                x - margin_x
            )
        )

        y1 = int(
            max(
                0,
                y - margin_y
            )
        )

        x2 = int(
            min(
                original_width,
                x + w + margin_x
            )
        )

        y2 = int(
            min(
                original_height,
                y + h + margin_y
            )
        )


        # ====================================================
        # VALIDATE CROP
        # ====================================================

        if x2 <= x1 or y2 <= y1:
            return None


        # ====================================================
        # CROP FACE
        # ====================================================

        face_crop = frame[
            y1:y2,
            x1:x2
        ]


        if face_crop.size == 0:
            return None


        # ====================================================
        # RESIZE TO MODEL INPUT SIZE
        # ====================================================

        face_crop = cv2.resize(
            face_crop,
            (
                224,
                224
            ),
            interpolation=cv2.INTER_AREA
        )


        return face_crop