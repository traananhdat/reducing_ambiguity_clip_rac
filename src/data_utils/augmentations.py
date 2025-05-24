# data_utils/augmentations.py

import torchvision.transforms as T
from PIL import Image # For RandomResizedCrop with bicubic interpolation if needed

# --- Standard Augmentations for Training ---
def get_train_transforms(image_size=224):
    """
    Defines standard data augmentation and preprocessing for training.
    """
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.7, 1.0)), # Zoom in a bit
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet a_means_stds
    ])

# --- Standard Preprocessing for Validation/Testing ---
def get_val_test_transforms(image_size=224):
    """
    Defines standard preprocessing for validation and testing (no heavy augmentation).
    """
    return T.Compose([
        T.Resize(image_size + 32), # Resize to a slightly larger size
        T.CenterCrop(image_size),   # Then center crop to the target size
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet a_means_stds
    ])

# --- CLIP Specific Preprocessing ---
def get_clip_preprocess(image_size=224):
    """
    Provides the standard preprocessing pipeline expected by many CLIP models.
    This often involves a specific resize, center crop, ToTensor, and normalization.
    The exact values might vary slightly based on the CLIP checkpoint.
    For official OpenAI CLIP, the normalization is often:
    mean=(0.48145466, 0.4578275, 0.40821073)
    std=(0.26862954, 0.26130258, 0.27577711)
    """
    # Note: The official CLIP models often use BICUBIC interpolation for resizing.
    # torchvision.transforms.Resize by default uses BILINEAR for PIL Images.
    # To ensure BICUBIC for PIL Image inputs:
    # T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),

    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC), # CLIP often uses BICUBIC
        T.CenterCrop(image_size),
        lambda image: image.convert("RGB") if image.mode != "RGB" else image, # Ensure RGB
        T.ToTensor(), # This scales images from [0, 255] to [0, 1]
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), # CLIP-specific means
            std=(0.26862954, 0.26130258, 0.27577711)   # CLIP-specific stds
        )
    ])


# --- Example Usage (can be run directly for testing) ---
if __name__ == '__main__':
    # Create a dummy PIL image for testing
    try:
        dummy_image = Image.new('RGB', (300, 400), color = 'skyblue')
        print(f"Created dummy image with mode: {dummy_image.mode}, size: {dummy_image.size}")

        print("\n--- Testing Training Transforms ---")
        train_transform = get_train_transforms(image_size=224)
        transformed_train_image = train_transform(dummy_image.copy())
        print(f"Training transformed image shape: {transformed_train_image.shape}")
        print(f"Min value: {transformed_train_image.min()}, Max value: {transformed_train_image.max()}")


        print("\n--- Testing Validation/Test Transforms ---")
        val_transform = get_val_test_transforms(image_size=224)
        transformed_val_image = val_transform(dummy_image.copy())
        print(f"Validation transformed image shape: {transformed_val_image.shape}")
        print(f"Min value: {transformed_val_image.min()}, Max value: {transformed_val_image.max()}")

        print("\n--- Testing CLIP Preprocessing ---")
        clip_transform = get_clip_preprocess(image_size=224)
        transformed_clip_image = clip_transform(dummy_image.copy())
        print(f"CLIP preprocessed image shape: {transformed_clip_image.shape}")
        print(f"Min value: {transformed_clip_image.min()}, Max value: {transformed_clip_image.max()}")

        # Test with a grayscale image for CLIP preprocess
        dummy_gray_image = Image.new('L', (300, 400), color = 'gray')
        print(f"\nCreated dummy grayscale image with mode: {dummy_gray_image.mode}, size: {dummy_gray_image.size}")
        transformed_clip_gray_image = clip_transform(dummy_gray_image.copy())
        print(f"CLIP preprocessed grayscale (converted to RGB) image shape: {transformed_clip_gray_image.shape}")


    except ImportError:
        print("Pillow (PIL) is not installed. Skipping example image creation.")
    except Exception as e:
        print(f"An error occurred during testing: {e}")