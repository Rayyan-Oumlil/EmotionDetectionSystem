import cv2
import numpy as np
from pathlib import Path


def preprocess_image(image_path: str | Path, output_path: str | Path) -> bool:
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Could not load image at {image_path}")
            return False

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_img, (48, 48), interpolation=cv2.INTER_AREA)
        return cv2.imwrite(str(output_path), resized_img)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def preprocess_dataset(input_path: str, output_path: str | None = None) -> None:
    input_dir = Path(input_path)
    output_dir = (
        Path(output_path)
        if output_path
        else input_dir.parent / f"{input_dir.name}_preprocessed"
    )

    if not input_dir.exists():
        print(f"Error: Dataset path {input_dir} does not exist.")
        return

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print("Transformations: Grayscale + Resize to 48x48")
    print("=" * 60)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    total_processed = total_failed = 0

    for class_folder in input_dir.iterdir():
        if not class_folder.is_dir():
            continue

        print(f"\nProcessing class: {class_folder.name}")
        output_class = output_dir / class_folder.name
        output_class.mkdir(parents=True, exist_ok=True)

        image_files = [
            f for f in class_folder.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            print(f"  No images found in {class_folder.name}")
            continue

        print(f"  Found {len(image_files)} images")
        class_processed = class_failed = 0

        for img_file in image_files:
            if preprocess_image(img_file, output_class / img_file.name):
                class_processed += 1
            else:
                class_failed += 1

        print(f"  Processed: {class_processed}  Failed: {class_failed}")
        total_processed += class_processed
        total_failed += class_failed

    print("\n" + "=" * 60)
    print(f"Done — {total_processed} processed, {total_failed} failed")
    print(f"Output: {output_dir}")
    print("=" * 60)


def analyze_preprocessed_dataset(dataset_path: str) -> None:
    dataset_dir = Path(dataset_path)

    if not dataset_dir.exists():
        print(f"Error: Dataset path {dataset_dir} does not exist.")
        return

    print(f"\nAnalyzing: {dataset_dir}")
    print("=" * 60)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    total_images = 0

    for class_folder in sorted(dataset_dir.iterdir()):
        if not class_folder.is_dir():
            continue

        image_files = [
            f for f in class_folder.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        count = len(image_files)
        total_images += count

        if image_files:
            sample = cv2.imread(str(image_files[0]))
            if sample is not None:
                h, w = sample.shape[:2]
                is_gray = sample.ndim == 2 or sample.shape[2] == 1
                print(f"{class_folder.name:15s}: {count:5d} images  [{w}x{h}, {'Gray' if is_gray else 'Color'}]")

    print(f"\n{'Total':15s}: {total_images:5d} images")
    print("=" * 60)


if __name__ == "__main__":
    input_dataset = "train"
    output_dataset = "train_preprocessed"

    print("Dataset Preprocessing Tool")
    print("=" * 60)

    if not Path(input_dataset).exists():
        print(f"Error: '{input_dataset}' not found.")
    else:
        _exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        total = sum(
            len([f for f in class_folder.iterdir()
                 if f.is_file() and f.suffix.lower() in _exts])
            for class_folder in Path(input_dataset).iterdir()
            if class_folder.is_dir()
        )
        print(f"Total images to process: {total}")
        print(f"Output: {output_dataset}")

        if input("Proceed? (yes/no): ").strip().lower() in ('yes', 'y'):
            preprocess_dataset(input_dataset, output_dataset)
            analyze_preprocessed_dataset(output_dataset)
