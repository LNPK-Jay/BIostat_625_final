
# U-Net Based Retinal Vessel Segmentation

## Table of Contents
1. [Project Description](#project-description)
2. [Dataset](#dataset)
3. [Method](#method)
    - [Image Processing](#image-processing)
    - [Loss Function](#loss-function)
    - [Model and Optimizer](#model-and-optimizer)
    - [Adaptive Batch Normalization](#adaptive-batch-normalization)
    - [Training](#training)
    - [Evaluation and Visualization](#evaluation-and-visualization)
4. [Results](#results)
5. [How to Run](#how-to-run)
6. [References](#references)

---

## Project Description

This project focuses on **retinal vessel segmentation** using a U-Net architecture with a ResNet-34 encoder. The goal is to detect and segment blood vessels in retinal images, which is essential for diagnosing diseases like diabetic retinopathy and cardiovascular conditions.

We address the challenge of domain variation in retinal images caused by different imaging equipment. By combining **image preprocessing** and **deep learning**, the model adapts effectively to different datasets.

---

## Dataset

We use two datasets for this project:

1. **DRIVE Dataset** (Training):
    - 40 retinal images (565×584 resolution) captured with a Canon CR5 camera.
    - Provides expert-annotated vessel segmentation maps.

2. **CHASE_DB1 Dataset** (Testing):
    - 28 retinal images (1280×960 resolution) captured with a Nikon NM 210 camera.
    - Differences in resolution, quality, and field of view make it ideal for testing generalizability.

---

## Method

### Image Processing

To improve model robustness, we apply the following preprocessing steps:
1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**:
   - Enhances vessel visibility while preserving color integrity.
2. **Image Resizing**:
   - All images and masks are resized to **256×256** pixels.
3. **Normalization**:
   - Images are normalized using ImageNet mean and standard deviation.
4. **Binarization**:
   - Segmentation masks are binarized for clean input.

---

### Loss Function

We use a **Combined Loss** that integrates:
1. **Dice Loss**: Measures the overlap between predicted masks and ground truth.
2. **BCEWithLogitsLoss**: Binary cross-entropy loss for pixel-level classification.

**Combined Loss Formula**:
\[
\text{Combined Loss} = 0.5 \times \text{Dice Loss} + 0.5 \times \text{BCE Loss}
\]

---

### Model and Optimizer

- **Model**: U-Net architecture with **ResNet-34** encoder initialized with ImageNet weights.
- **Optimizer**: Adam optimizer with a learning rate of **1×10⁻⁴**.

---

### Adaptive Batch Normalization (AdaBN)

To adapt the model to the target domain, we update batch normalization statistics using **AdaBN**. This improves model performance on test images without retraining.

---

### Training

- **Epochs**: 300
- **Batch Size**: 4
- **Loss**: Combined Loss
- **Optimizer**: Adam

---

### Evaluation and Visualization

We use two primary metrics:
1. **Mean Dice Score**:
   - Measures the overlap between predicted and ground truth masks.
2. **Mean IoU Score** (Intersection over Union):
   - Measures the ratio of intersection to union between predicted and ground truth masks.

**Qualitative Visualization**:
- **Input Image**: Original retinal image.
- **Predicted Mask**: Model output after segmentation.
- **True Mask**: Ground truth segmentation map.

---

## Results

The evaluation metrics are as follows:

- **Mean Dice Score**: 0.3019
- **Mean IoU Score**: 0.1779

While the predicted masks capture the overall vessel structure, further improvements are needed to reduce noise and enhance fine-grained vessel details.

---

## How to Run

### Prerequisites
- Python 3.x
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- segmentation-models-pytorch

### Steps to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your_username/retinal-vessel-segmentation.git
   cd retinal-vessel-segmentation
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Training Script**:
   ```bash
   python main.py
   ```

4. **Evaluate the Model**:
   ```bash
   python main.py --evaluate
   ```

5. **View Results**:
   - The predicted masks and visualizations will be displayed and saved in the `results/` directory.

---

## References

1. M. D. Abràmoff, M. K. Garvin, and M. Sonka, *"Retinal imaging and image analysis"*, IEEE Reviews in Biomedical Engineering, 2010.
2. O. Ronneberger, P. Fischer, and T. Brox, *"U-Net: Convolutional networks for biomedical image segmentation"*, 2015.
3. G. Litjens et al., *"A survey on deep learning in medical image analysis"*, Medical Image Analysis, 2017.

---

## Contact

For any questions or issues, feel free to contact:
- **Name**: [Your Name]
- **Email**: [Your Email]
- **GitHub**: [Your GitHub Profile Link]
