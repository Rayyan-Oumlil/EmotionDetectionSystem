import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn import EmotionCNN

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'emotionCNN.pth')
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'train_preprocessed_augmented')


def load_model(device: torch.device) -> EmotionCNN:
    model = EmotionCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


def build_loader(data_dir: str, batch_size: int = 64) -> DataLoader:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    dataset = ImageFolder(root=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def evaluate(model: EmotionCNN, loader: DataLoader, device: torch.device) -> tuple[list, list]:
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return all_labels, all_preds


def plot_confusion_matrix(labels: list, preds: list, output_path: str) -> None:
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTIONS)

    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, colorbar=False, cmap='Blues', xticks_rotation=45)
    ax.set_title('Confusion Matrix — EmotionCNN', fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {output_path}")


def main() -> None:
    if not os.path.exists(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
        print("Point DATA_DIR at your preprocessed dataset folder and re-run.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(device)
    loader = build_loader(DATA_DIR)

    print(f"Evaluating on {len(loader.dataset)} images...")
    labels, preds = evaluate(model, loader, device)

    acc = np.mean(np.array(labels) == np.array(preds)) * 100
    print(f"\nOverall accuracy: {acc:.2f}%\n")
    print(classification_report(labels, preds, target_names=EMOTIONS))

    output_path = os.path.join(BASE_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(labels, preds, output_path)


if __name__ == '__main__':
    main()
