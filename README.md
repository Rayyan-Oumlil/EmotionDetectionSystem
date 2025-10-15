<<<<<<< HEAD
# Emotion Detection Project

This repository contains code and resources for an emotion detection system using deep learning and computer vision. The project includes data preprocessing, augmentation, model training (CNN and ResNet), and a Tkinter-based GUI for real-time emotion recognition from webcam input.

## Project Structure

### CNN Folder (`CNN/`):
- `app_cnn.py` — CNN GUI application for emotion detection
- `cnn.py` — Implementation of the EmotionCNN model
- `emotionCNN.pth` — Trained EmotionCNN model weights
- `predictions_cnn.csv` — CNN predictions output
- `images/` — Emotion icon images for GUI display

### ResNet18 Folder (`ResNet18/`):
- `app_resnet18.py` — ResNet18 GUI application for emotion detection
- `emotion_resnet18_best.pth` — Trained ResNet18 model weights
- `predictions_resnet18.csv` — ResNet18 predictions output
- `images/` — Emotion icon images for GUI display

### Root Files:
- `data_augmentation.py` — Tools for augmenting image datasets
- `preprocess.py` — Scripts for preprocessing images (grayscale, resize to 48x48)
- `resnet.ipynb` — Jupyter notebook for training and evaluating models
- `test_template.csv` — Template for test image IDs
- `images/` — Shared emotion icon images

## Usage

### 1. Preprocess Dataset

Run `preprocess.py` to convert images to grayscale and resize to 48x48 pixels.

### 2. Data Augmentation

Use `data_augmentation.py` to create augmented versions of your dataset for improved model robustness.

### 3. Model Training

Train models using `resnet.ipynb`. You can choose between custom CNN (`cnn.py`) and ResNet architectures.

### 4. GUI Applications

Launch the emotion detector GUI:

**CNN Version:**
```bash
cd CNN && python app_cnn.py
```

**ResNet18 Version:**
```bash
cd ResNet18 && python app_resnet18.py
```

The apps will use your webcam to detect faces and predict emotions in real time.

### 5. Prediction Submission

Generate predictions for test images using the notebook and save results to:
- `CNN/predictions_cnn.csv` for CNN model
- `ResNet18/predictions_resnet18.csv` for ResNet18 model

## Requirements

Install dependencies with:

```sh
pip install -r requirements.txt
```

## Dataset Setup

### ⚠️ Important: Datasets are excluded from Git
The training datasets (`train/` and `test/` folders) are **not included** in this repository due to their size (~150MB). 

### For Training (Required for model training):
1. **Download the emotion dataset** and place it in the project root
2. **Folder structure should be:**
   ```
   train/
   ├── angry/
   ├── disgust/
   ├── fear/
   ├── happy/
   ├── neutral/
   ├── sad/
   └── surprise/
   ```
3. **Run preprocessing:**
   ```bash
   python preprocess.py
   ```

### For Testing Only (Using pre-trained models):
- **No dataset required** - use the provided pre-trained models
- **CNN version:**
  ```bash
  cd CNN && python app_cnn.py
  ```
- **ResNet18 version:**
  ```bash
  cd ResNet18 && python app_resnet18.py
  ```

### Getting the Dataset
If you need the training dataset, please:
1. Contact the project maintainer
2. Or use your own emotion dataset with the same folder structure
3. Or download from the original competition source

## Notes

- **Training**: Requires the full dataset 
- **Testing**: Uses pre-trained models, no dataset needed
- Make sure emotion icon images are available in the `images/` directory

## License

MIT License

---

For details on each module, see the source files:
- [CNN/app_cnn.py](CNN/app_cnn.py) - CNN GUI application
- [CNN/cnn.py](CNN/cnn.py) - CNN model implementation
- [ResNet18/app_resnet18.py](ResNet18/app_resnet18.py) - ResNet18 GUI application
- [data_augmentation.py](data_augmentation.py) - Data augmentation tools
- [preprocess.py](preprocess.py) - Image preprocessing

---

**Credits:** Inspired by [CodeML_PolyAI_Winners_2025](https://github.com/GharibSidney/CodeML_PolyAI_Winners_2025) by [@GharibSidney](https://github.com/GharibSidney) and [@RahmaAmmari](https://github.com/RahmaAmmari).
=======
# EmotionDetectionSystem
Real-time facial emotion detection system using deep learning and computer vision. Includes preprocessing, augmentation, model training, and live webcam inference with GUI.
>>>>>>> fd13723741d19c67fe791c4b418dc824cd748934
