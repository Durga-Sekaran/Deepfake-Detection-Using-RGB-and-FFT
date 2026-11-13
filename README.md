# Deepfake Detection Using RGB and Frequency-Domain Features

### An Explainable Deepfake Detection System Using Spatial and Frequency-Domain Analysis

## Project Overview

Deepfake technology uses Artificial Intelligence and Deep Learning techniques to manipulate or generate realistic images and videos of people. Although this technology has creative applications, it can also be misused to spread misinformation, impersonate individuals and create misleading digital content.

This project presents an AI-based Deepfake Detection System that analyzes both spatial (RGB) and frequency-domain features to identify manipulated videos.

The system explores three different approaches:

1. **RGB-Based Detection:** Uses spatial information from video frames to identify visual manipulation.
2. **FFT-Based Detection:** Uses Fast Fourier Transform (FFT) to extract frequency-domain features and identify manipulation patterns.
3. **RGB + FFT Fusion:** Combines spatial and frequency-domain information to improve deepfake detection.

The project also includes an Explainable Deepfake Detection web application that allows users to upload videos and obtain prediction results with visual explanations.

---

## Objectives

- Develop an AI-based system to detect deepfake videos.
- Analyze spatial features using RGB images.
- Extract frequency-domain features using Fast Fourier Transform (FFT).
- Compare RGB-only, FFT-only and RGB+FFT fusion approaches.
- Implement explainable AI techniques using Grad-CAM.
- Develop an interactive Streamlit application for deepfake detection.

---

## Technologies Used

| Category                | Technologies                 |
| ----------------------- | ---------------------------- |
| Programming Language    | Python                       |
| Deep Learning           | PyTorch                      |
| Computer Vision         | OpenCV                       |
| Image Processing        | NumPy                        |
| Frequency Analysis      | Fast Fourier Transform (FFT) |
| Face Detection          | YuNet                        |
| Explainable AI          | Grad-CAM                     |
| Web Application         | Streamlit                    |
| Development Environment | Google Colab, VS Code        |
| Dataset                 | FaceForensics++ (C23)        |
| Version Control         | Git, GitHub                  |

---

## Dataset

**FaceForensics++ (C23)**

This project uses the FaceForensics++ dataset with the C23 compression setting.

The working dataset consists of a selected subset of 1,200 videos:

| Category        | Number of Videos |
| --------------- | ---------------: |
| Original Videos |              600 |
| Fake Videos     |              600 |
| Total           |            1,200 |

### Dataset Split

| Dataset    | Number of Videos | Percentage |
| ---------- | ---------------: | ---------: |
| Training   |              840 |        70% |
| Validation |              180 |        15% |
| Testing    |              180 |        15% |
| Total      |            1,200 |       100% |

The dataset is organized into real and manipulated video samples for model training and evaluation.

**Note:** The original dataset is not included in this repository because of its large size.

---

## System Architecture

The proposed system follows three major detection approaches.

### 1. RGB-Based Detection

- Extract representative frames from uploaded videos.
- Detect and crop facial regions.
- Resize face images to 224 × 224 pixels.
- Extract spatial features using the RGB-based deep learning model.
- Predict whether the video is real or fake.

### 2. FFT-Based Detection

- Convert the preprocessed face images into the frequency domain.
- Apply Fast Fourier Transform (FFT).
- Generate frequency representations.
- Extract frequency-domain features.
- Use the FFT-based model to classify the video.

### 3. RGB + FFT Fusion Detection

- Extract spatial features from RGB images.
- Extract frequency-domain features using FFT.
- Combine both feature representations.
- Perform classification using the fusion model.
- Generate the final real or fake prediction.

---

## Explainable AI

This project incorporates Grad-CAM (Gradient-weighted Class Activation Mapping) to provide visual explanations for deepfake predictions.

Grad-CAM generates heatmaps that highlight important image regions influencing the model's prediction.

### Benefits

- Visualizes important facial regions.
- Helps understand model predictions.
- Improves interpretability.
- Provides additional insights into detected manipulations.

---

## Project Structure

```text
deepfake_detection/
│
├── .streamlit/
│   └── secrets.toml.example
│
├── explainability/
│   └── gradcam.py
│
├── face_detection/
│   └── face_detector.py
│
├── model/
│   └── model.py
│
├── preprocessing/
│   ├── fft.py
│   ├── image_preprocessing.py
│   ├── inference_preprocessing.py
│   └── video_processing.py
│
├── app.py
├── inference_test.py
├── gradcam_test.py
├── video_gradcam_test.py
├── select_representative_frame.py
├── requirements.txt
├── README.md
└── .gitignore
```

*The above structure highlights the major project components. Additional testing and utility scripts are also included in the repository.*

---

## Models Developed

Three different model configurations are implemented and evaluated.

| Model                  | Input Features                | Purpose                              |
| ---------------------- | ----------------------------- | ------------------------------------ |
| RGB Model              | Spatial RGB Features          | Detect visual manipulation           |
| FFT Model              | Frequency-Domain Features     | Detect frequency inconsistencies     |
| RGB + FFT Fusion Model | Combined RGB and FFT Features | Utilize both feature representations |

---

## Web Application

The project includes an interactive Streamlit application titled:

**Explainable Deepfake Detection**

### Application Features

- Upload video files for analysis.
- Perform face detection and preprocessing.
- Analyze video frames using trained models.
- Generate real or fake predictions.
- Display prediction confidence.
- Visualize Grad-CAM heatmaps.
- Support multiple detection approaches.

### Prediction Classes

- **Real:** The model classifies the uploaded video as authentic.
- **Fake:** The model classifies the uploaded video as manipulated.

*Predictions are model-based estimates and should not be treated as definitive proof of authenticity.*

---

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Durga-Sekaran/Deepfake-Detection-Using-RGB-and-FFT.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Deepfake-Detection-Using-RGB-and-FFT
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Add Model Checkpoints

Download or copy the trained model checkpoints into the appropriate locations expected by the application.

The trained model weights are not included in this GitHub repository.

### Step 7: Run the Application

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

## Model Evaluation

The three model configurations are evaluated independently to understand their performance.

| Evaluation Metric | Description                                |
| ----------------- | ------------------------------------------ |
| Accuracy          | Overall classification correctness         |
| Precision         | Correct positive predictions               |
| Recall            | Ability to identify fake samples           |
| F1-Score          | Balance between precision and recall       |
| Confusion Matrix  | Comparison of actual and predicted classes |

### Model Comparison

| Model                  | Accuracy | Precision | Recall | F1-Score |
| ---------------------- | -------- | --------- | ------ | -------- |
| RGB-Based Model        | 83.66 %  | 0.8315    | 0.8444 | 0.8379   |
| FFT-Based Model        | 64.04 %  | 0.6869    | 0.5167 | 0.5897   |
| RGB + FFT Fusion Model | 85.71 %  | 0.8600    | 0.8533 | 0.8567   |

---

## Future Enhancements

- Improve detection accuracy using larger and more diverse datasets.
- Optimize model inference time.
- Support additional video formats.
- Improve frequency-domain feature extraction.
- Explore advanced feature fusion techniques.
- Enhance explainability using additional XAI methods.
- Deploy the application as a publicly accessible web service.

---

## Applications

- Digital media verification.
- Social media content analysis.
- Misinformation detection.
- Cybersecurity applications.
- Digital forensics.
- Online identity protection.

---

## Disclaimer

This project is developed for academic, research and educational purposes.

The system provides AI-based predictions and visual explanations. Its predictions may contain errors and should not be considered conclusive evidence of video authenticity.

---

## Author

**Durga Sekaran**

B.Tech – Artificial Intelligence and Data Science

GitHub: [Durga-Sekaran](https://github.com/Durga-Sekaran)

Project Repository: [Deepfake Detection Using RGB and FFT](https://github.com/Durga-Sekaran/Deepfake-Detection-Using-RGB-and-FFT)

---

## Acknowledgement

We acknowledge the researchers and contributors of the FaceForensics++ dataset and the open-source communities behind Python, PyTorch, OpenCV and Streamlit for providing valuable resources that supported this project.
