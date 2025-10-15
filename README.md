# Emotion Detection System

Real-time facial emotion detection system using deep learning and computer vision. Includes preprocessing, augmentation, model training, and live webcam inference with GUI.

## About

This project implements a CNN-based emotion detection system that can classify facial expressions into 7 different emotions:
- Angry (Colère)
- Disgust (Dégoût) 
- Fear (Peur)
- Happy (Joie)
- Neutral (Neutre)
- Sad (Tristesse)
- Surprise (Surprise)

## Features

- CNN model for emotion classification
- Real-time webcam emotion detection
- Data preprocessing and augmentation
- Model training with metrics visualization
- GUI application for live inference

## Project Structure

```
EmotionDetectionSystem/
├── CNN/                    # CNN model implementation
│   ├── cnn.py             # CNN model architecture
│   ├── app_cnn.py         # GUI application
│   ├── emotionCNN.pth     # Trained model weights
│   ├── metrics.png        # Training metrics
│   ├── predictions_cnn.csv # Model predictions
│   └── images/            # Emotion sample images
├── train/                 # Training data
│   ├── angry/            # Angry emotion images
│   ├── disgust/          # Disgust emotion images
│   ├── fear/             # Fear emotion images
│   ├── happy/            # Happy emotion images
│   ├── neutral/          # Neutral emotion images
│   ├── sad/              # Sad emotion images
│   └── surprise/         # Surprise emotion images
├── preprocess.py          # Data preprocessing
├── data_augmentation.py   # Data augmentation
├── model_training.ipynb   # Training notebook
├── requirements.txt       # Dependencies
└── test_template.csv      # Test data template
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Rayyan-Oumlil/EmotionDetectionSystem.git
cd EmotionDetectionSystem
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Model

1. Open `model_training.ipynb` in Jupyter Notebook
2. Run all cells to train the CNN model
3. Model will be saved in the CNN folder

### Running Application

#### CNN Model
```bash
python CNN/app_cnn.py
```

## Model Performance

The CNN model is trained on the FER2013 dataset and can classify 7 different emotions with high accuracy.

## Dependencies

- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Dataset Setup

The training dataset (`train/` folder) is not included in this repository due to size constraints.

To use the pre-trained models, simply run:
```bash
python CNN/app_cnn.py
```

## Contact

For questions or suggestions, please open an issue on GitHub.

---
