
# EgoVision Dataset

The **EgoVision** dataset is a curated collection of 2,904 high-quality (1920×1440) images collected from seven construction-related environments. Designed for egocentric vision tasks in construction safety and automation, this dataset supports various vision-related applications (e.g., image classification, object detection, and semantic segmentation).

## 📁 Dataset Structure

```
EgoVision/
├── EgoVision/
│   ├── scene1-outdoor/
│   ├── scene2-outdoor/
│   ├── scene3-outdoor/
│   ├── scene4-outdoor/
│   ├── scene5-outdoor/
│   ├── scene6-indoor/
│   ├── scene7-indoor/
│   └── scene8-indoor/
├── processing/
│   ├── chessboard/
│   ├── camera_calibration.py
│   ├── preprocessing.py
│   └── augmentation.py
└── README.md
```

## 📷 Dataset Overview

- **Total Images**: 2,904  
- **Indoor Scenes**: 1,464 images from 3 environments  
- **Outdoor Scenes**: 1,440 images from 5 environments  
- **Perspective**: Egocentric (first-person) view  
- **Image Size**: 1920×1440 pixels  

### 1. 📌 Download our EgoVision

```bash
git clone https://github.com/JiuyiX/EgoVision.git
cd EgoVision
```

Ensure you have installed OpenCV, NumPy, PIL, and torchvision as required by the scripts.

### 2. 🔍 Camera Calibration (Optional)

**Purpose**: Correct lens distortion and align egocentric images into a normalized projection space.

**Script**: `calibration/camera_calibration.py`

**How it works**:
- Uses checkerboard patterns for calibration.
- Extracts corner points, refines them, and calculates camera matrix and distortion coefficients.
- Optionally undistorts a sample image using the computed parameters.

**Usage**:
```bash
python camera_calibration.py
```

Make sure your calibration images are in a folder called `chessboard/`.

### 3. 🧼 Image Preprocessing (Optional)

**Purpose**: Resize, normalize, grayscale conversion, and format conversion (you can try more image preprocessing methods).

**Script**: `preprocessing/preprocessing.py`

**Functions**:
- `resize_image`: Resize to a standard shape.
- `normalize_image`: Normalize pixel values to [0,1].
- `to_grayscale`: Convert to single-channel grayscale.
- `pil_to_cv2` and `cv2_to_pil`: Convert between OpenCV and PIL formats.

**Usage**: Import and apply the desired transformations in your pipeline.

### 4. 🔁 Data Augmentation (Optional)

**Purpose**: Enrich dataset through transformation and increase robustness (you can try more data augmentation methods).

**Script**: `augmentation/augmentation.py`

**Pipeline Includes**:
- Horizontal/vertical flips
- Random rotation
- Color jitter (brightness, contrast, saturation, hue)
- Resize

**Usage**:
```python
from augmentation import apply_augmentation
augmented_tensor = apply_augmentation(image_cv2)
```
Use OpenCV to read your image, and pass it to the augmentation pipeline.

### 5. ✅ Dataset Usage Example

Use the processed data in your deep learning pipeline (e.g., PyTorch, TensorFlow). The provided scripts support camera calibration method and part of preprocessing and augmentation pipelines.

## 📚 Reference

If you use this dataset, please cite:

> Liu, Z., Xu, J., Suen, C. W. K., Chen, M., Zou, Z., & Shi, Y. (2025). Egocentric camera-based method for detecting static hazardous objects on construction sites. *Automation in Construction*, 172, 106048.

## 🧠 Applications

The EgoVision dataset is suitable for:
- STF Risk assessment automation
- PPE detection
- Scene understanding in egocentric construction robotics
- More to be explored...
