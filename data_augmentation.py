import cv2
import numpy as np
import os
from pathlib import Path
import shutil

def add_gaussian_noise(image_path, mean=4, var=100):
    """
    Adds Gaussian noise to an image.
    Args:
        image_path (str): The path to the input image.
        mean (float): The mean of the Gaussian distribution.
        var (float): The variance of the Gaussian distribution.
    Returns:
        numpy.ndarray: The image with added Gaussian noise.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None
    
    sigma = var ** 0.5
    gaussian_noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    
    noisy_image = img.astype(np.float32) + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def rotate_image(image_path, angle):
    """
    Rotates an image by a given angle.
    Args:
        image_path (str): Path to the input image
        angle (float): Rotation angle in degrees
    Returns:
        numpy.ndarray: Rotated image
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated

def flip_image(image_path, flip_code):
    """
    Flips an image horizontally, vertically, or both.
    Args:
        image_path (str): Path to the input image
        flip_code (int): 0 for vertical flip, 1 for horizontal flip, -1 for both
    Returns:
        numpy.ndarray: Flipped image
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    flipped = cv2.flip(img, flip_code)
    return flipped

def adjust_brightness(image_path, alpha=1.2, beta=30):
    """
    Adjusts brightness and contrast of an image.
    Args:
        image_path (str): Path to the input image
        alpha (float): Contrast control (1.0-3.0)
        beta (int): Brightness control (0-100)
    Returns:
        numpy.ndarray: Brightness adjusted image
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted

def create_augmented_dataset(dataset_path: str, output_path: str = None, augmentation_factor: int = 3):
    """
    Creates an augmented dataset by applying various transformations to images.
    Works with ImageFolder structure: dataset_path/class_name/images
    
    Args:
        dataset_path (str): Path to the original dataset
        output_path (str): Path where augmented dataset will be saved (optional)
        augmentation_factor (int): Number of augmented versions per original image
    """
    dataset_path = Path(dataset_path)
    
    if output_path is None:
        output_path = dataset_path.parent / f"{dataset_path.name}_augmented"
    else:
        output_path = Path(output_path)
    
    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist.")
        return
    
    print(f"Creating augmented dataset from: {dataset_path}")
    print(f"Output will be saved to: {output_path}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    total_original = 0
    total_augmented = 0
    
    # Iterate through each class folder
    for class_folder in dataset_path.iterdir():
        if not class_folder.is_dir():
            continue
            
        print(f"\nProcessing class: {class_folder.name}")
        output_class = output_path / class_folder.name
        output_class.mkdir(parents=True, exist_ok=True)
        
        # Get all image files in the class
        image_files = [f for f in class_folder.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"  No images found in {class_folder.name}")
            continue
        
        print(f"  Found {len(image_files)} images")
        total_original += len(image_files)
        
        class_augmented = 0
        
        # Process each image
        for img_file in image_files:
            try:
                # Copy original image
                original_output = output_class / img_file.name
                shutil.copy2(img_file, original_output)
                
                # Create augmented versions
                base_name = img_file.stem
                extension = img_file.suffix
                
                augmentation_count = 0
                
                # Apply different augmentations
                augmentations = [
                    ("noise", lambda: add_gaussian_noise(str(img_file))),
                    ("rot15", lambda: rotate_image(str(img_file), 15)),
                    ("rot_15", lambda: rotate_image(str(img_file), -15)),
                    ("flip_h", lambda: flip_image(str(img_file), 1)),
                    # ("flip_v", lambda: flip_image(str(img_file), 0)),
                    # ("bright", lambda: adjust_brightness(str(img_file), 1.3, 40)),
                    # ("dark", lambda: adjust_brightness(str(img_file), 0.7, -20)),
                ]
                
                # Apply augmentations up to the specified factor
                for aug_name, aug_func in augmentations:
                    if augmentation_count >= augmentation_factor:
                        break
                        
                    augmented_img = aug_func()
                    if augmented_img is not None:
                        aug_filename = f"{base_name}_{aug_name}{extension}"
                        aug_path = output_class / aug_filename
                        
                        success = cv2.imwrite(str(aug_path), augmented_img)
                        if success:
                            augmentation_count += 1
                            class_augmented += 1
                        else:
                            print(f"    Failed to save {aug_filename}")
                
            except Exception as e:
                print(f"    Error processing {img_file.name}: {str(e)}")
                continue
        
        print(f"  Created {class_augmented} augmented images for {class_folder.name}")
        total_augmented += class_augmented
    
    print(f"\n" + "="*60)
    print(f"Augmentation complete!")
    print(f"Original images: {total_original}")
    print(f"Augmented images created: {total_augmented}")
    print(f"Total images in new dataset: {total_original + total_augmented}")
    print(f"Output location: {output_path}")
    print("="*60)

def analyze_dataset_structure(dataset_path: str):
    """
    Analyzes and prints the structure of the dataset.
    Works with ImageFolder structure: dataset_path/class_name/images
    Args:
        dataset_path (str): Path to the dataset
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist.")
        return
    
    print(f"\nDataset structure for: {dataset_path}")
    print("="*60)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    total_images = 0
    class_counts = {}
    
    # Iterate through class folders
    for class_folder in sorted(dataset_path.iterdir()):
        if not class_folder.is_dir():
            continue
            
        # Count images in class
        image_count = len([f for f in class_folder.iterdir() 
                         if f.is_file() and f.suffix.lower() in image_extensions])
        
        class_counts[class_folder.name] = image_count
        total_images += image_count
    
    # Print results
    if class_counts:
        print(f"\nClass distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name:15s}: {count:5d} images")
        print(f"\n{'Total':15s}: {total_images:5d} images")
    else:
        print("No class folders found or folders are empty.")
    
    print("="*60)
    
    return class_counts, total_images

if __name__ == "__main__":
    dataset_path = "train_preprocessed"
    
    # Analyze dataset structure first
    print("Analyzing original dataset...")
    analyze_dataset_structure(dataset_path)
    
    # Ask user if they want to proceed
    print("\nDo you want to create an augmented dataset?")
    print("This will create augmented versions of all images.")
    response = input("Proceed? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        # Create augmented dataset
        create_augmented_dataset(
            dataset_path=dataset_path,
            output_path="train_preprocessed_augmented",
            augmentation_factor=4  # Number of augmented versions per original image
        )
        
        # Analyze the augmented dataset
        print("\nAnalyzing augmented dataset...")
        analyze_dataset_structure("train_preprocessed_augmented")
    else:
        print("Augmentation cancelled.")

    # Original single image example (your existing code):
    # input_image_path = 'train_preprocessed/angry/Training_99518394.jpg'
    # output_image_path = 'noisy_image.jpg'
    
    # if os.path.exists(input_image_path):
    #     noisy_img = add_gaussian_noise(input_image_path)
    #     if noisy_img is not None:
    #         cv2.imwrite(output_image_path, noisy_img)
    #         print(f"Noisy image saved to {output_image_path}")
            
    #         # Display images (comment out if running without display)
    #         cv2.imshow('Original Image', cv2.imread(input_image_path))
    #         cv2.imshow('Noisy Image', noisy_img)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    # else:
    #     print("Example image path not found. Please update the path or use the dataset augmentation functions.")