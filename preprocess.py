import cv2
import numpy as np
from pathlib import Path
import shutil

def preprocess_image(image_path, output_path):
    """
    Preprocess a single image: convert to grayscale and resize to 48x48
    
    Args:
        image_path (str or Path): Path to input image
        output_path (str or Path): Path to save preprocessed image
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Could not load image at {image_path}")
            return False
        
        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to 48x48
        resized_img = cv2.resize(gray_img, (48, 48), interpolation=cv2.INTER_AREA)
        
        # Save preprocessed image
        success = cv2.imwrite(str(output_path), resized_img)
        return success
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def preprocess_dataset(input_path, output_path=None):
    """
    Preprocess entire dataset: convert all images to grayscale 48x48
    Works with ImageFolder structure: dataset_path/class_name/images
    
    Args:
        input_path (str): Path to the original dataset
        output_path (str): Path where preprocessed dataset will be saved (optional)
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.name}_preprocessed"
    else:
        output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"Error: Dataset path {input_path} does not exist.")
        return
    
    print(f"Preprocessing dataset from: {input_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Transformations: Grayscale + Resize to 48x48")
    print("="*60)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    total_processed = 0
    total_failed = 0
    
    # Iterate through each class folder
    for class_folder in input_path.iterdir():
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
        
        class_processed = 0
        class_failed = 0
        
        # Process each image
        for img_file in image_files:
            output_file = output_class / img_file.name
            
            if preprocess_image(img_file, output_file):
                class_processed += 1
            else:
                class_failed += 1
        
        print(f"  ✓ Successfully processed: {class_processed}")
        if class_failed > 0:
            print(f"  ✗ Failed: {class_failed}")
        
        total_processed += class_processed
        total_failed += class_failed
    
    print(f"\n" + "="*60)
    print(f"Preprocessing complete!")
    print(f"Successfully processed: {total_processed} images")
    if total_failed > 0:
        print(f"Failed: {total_failed} images")
    print(f"Output location: {output_path}")
    print("="*60)

def analyze_preprocessed_dataset(dataset_path):
    """
    Analyze preprocessed dataset and verify dimensions
    
    Args:
        dataset_path (str): Path to the preprocessed dataset
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist.")
        return
    
    print(f"\nAnalyzing preprocessed dataset: {dataset_path}")
    print("="*60)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    total_images = 0
    correct_size = 0
    grayscale_count = 0
    
    # Check a few images to verify preprocessing
    for class_folder in sorted(dataset_path.iterdir()):
        if not class_folder.is_dir():
            continue
        
        image_files = [f for f in class_folder.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        class_count = len(image_files)
        total_images += class_count
        
        # Check first image in each class for verification
        if image_files:
            test_img = cv2.imread(str(image_files[0]))
            if test_img is not None:
                height, width = test_img.shape[:2]
                is_gray = len(test_img.shape) == 2 or test_img.shape[2] == 1
                
                print(f"{class_folder.name:15s}: {class_count:5d} images ", end="")
                print(f"[Sample: {width}x{height}, {'Grayscale' if is_gray else 'Color'}]")
                
                if width == 48 and height == 48:
                    correct_size += class_count
                if is_gray:
                    grayscale_count += class_count
    
    print(f"\n{'Total':15s}: {total_images:5d} images")
    print(f"\nVerification (based on samples):")
    print(f"  Expected to be 48x48 and grayscale")
    print("="*60)

if __name__ == "__main__":
    # Configuration
    input_dataset = "train"  # Change this to your dataset folder
    output_dataset = "train_preprocessed"  # Change this to desired output folder
    
    print("Dataset Preprocessing Tool")
    print("="*60)
    print("This will convert all images to:")
    print("  - Grayscale (1 channel)")
    print("  - Size: 48x48 pixels")
    print("="*60)
    
    # Check if input exists
    if not Path(input_dataset).exists():
        print(f"\nError: Input dataset '{input_dataset}' not found!")
        print("Please update the 'input_dataset' variable in the script.")
    else:
        # Show what will be processed
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        total = 0
        for class_folder in Path(input_dataset).iterdir():
            if class_folder.is_dir():
                count = len([f for f in class_folder.iterdir() 
                           if f.is_file() and f.suffix.lower() in image_extensions])
                if count > 0:
                    print(f"  {class_folder.name}: {count} images")
                    total += count
        print(f"  Total: {total} images")
        
        # Ask for confirmation
        print(f"\nOutput will be saved to: {output_dataset}")
        response = input("\nProceed with preprocessing? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            # Preprocess the dataset
            preprocess_dataset(input_dataset, output_dataset)
            
            # Analyze the results
            analyze_preprocessed_dataset(output_dataset)
            
            print("\n✓ Preprocessing complete! You can now use the preprocessed dataset for training.")
        else:
            print("Preprocessing cancelled.")