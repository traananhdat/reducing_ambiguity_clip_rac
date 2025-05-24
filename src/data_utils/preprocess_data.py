# data_utils/preprocess_data.py

import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm # For progress bars
import argparse
import sys

# Assuming augmentations.py is in the same directory or accessible via PYTHONPATH
try:
    from .augmentations import get_clip_preprocess # Or any other preprocess you need
except ImportError:
    # This allows the script to be run directly for testing if it's in the same folder
    # as augmentations.py, otherwise, it assumes it's part of a package.
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from augmentations import get_clip_preprocess


def create_processed_pt_file(
    authentic_dir,
    counterfeit_dir,
    output_dir,
    output_filename,
    image_size=224,
    label_mapping=None,
    preprocess_transform=None
):
    """
    Processes images from authentic and counterfeit directories, applies transformations,
    and saves them into a .pt file containing image tensors and labels.

    Args:
        authentic_dir (str): Path to the directory containing authentic images.
        counterfeit_dir (str): Path to the directory containing counterfeit images.
        output_dir (str): Directory where the .pt file will be saved.
        output_filename (str): Name of the output .pt file (e.g., "product_A_train.pt").
        image_size (int): The target size for images (e.g., 224).
        label_mapping (dict, optional): Mapping for labels.
                                        Defaults to {'authentic': 0, 'counterfeit': 1}.
        preprocess_transform (callable, optional): The torchvision transform to apply.
                                                   Defaults to CLIP preprocessing.
    """
    if label_mapping is None:
        label_mapping = {'authentic': 0, 'counterfeit': 1}
    if preprocess_transform is None:
        preprocess_transform = get_clip_preprocess(image_size=image_size)

    all_images_list = []
    all_labels_list = []
    all_image_paths_list = [] # To optionally store paths for reference

    # Process authentic images
    print(f"Processing authentic images from: {authentic_dir}")
    if not os.path.isdir(authentic_dir):
        print(f"Warning: Authentic directory not found: {authentic_dir}")
    else:
        for img_name in tqdm(os.listdir(authentic_dir)):
            img_path = os.path.join(authentic_dir, img_name)
            try:
                # Check if it's a file and a common image type
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image = Image.open(img_path).convert("RGB")
                    image_tensor = preprocess_transform(image)
                    all_images_list.append(image_tensor)
                    all_labels_list.append(label_mapping['authentic'])
                    all_image_paths_list.append(os.path.relpath(img_path, start=os.path.join(output_dir, "..", ".."))) # Relative path
                else:
                    print(f"Skipping non-image file or unsupported extension: {img_name}")
            except Exception as e:
                print(f"Error processing authentic image {img_path}: {e}")

    # Process counterfeit images
    print(f"\nProcessing counterfeit images from: {counterfeit_dir}")
    if not os.path.isdir(counterfeit_dir):
        print(f"Warning: Counterfeit directory not found: {counterfeit_dir}")
    else:
        for img_name in tqdm(os.listdir(counterfeit_dir)):
            img_path = os.path.join(counterfeit_dir, img_name)
            try:
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image = Image.open(img_path).convert("RGB")
                    image_tensor = preprocess_transform(image)
                    all_images_list.append(image_tensor)
                    all_labels_list.append(label_mapping['counterfeit'])
                    all_image_paths_list.append(os.path.relpath(img_path, start=os.path.join(output_dir, "..", ".."))) # Relative path
                else:
                    print(f"Skipping non-image file or unsupported extension: {img_name}")
            except Exception as e:
                print(f"Error processing counterfeit image {img_path}: {e}")

    if not all_images_list:
        print("No images were processed. Output file will not be created.")
        return

    # Stack all image tensors into a single tensor: (N, C, H, W)
    images_tensor = torch.stack(all_images_list)
    # Convert labels list to a tensor
    labels_tensor = torch.tensor(all_labels_list, dtype=torch.long)

    print(f"\nTotal images processed: {len(all_images_list)}")
    print(f"Images tensor shape: {images_tensor.shape}")
    print(f"Labels tensor shape: {labels_tensor.shape}")

    # Save to .pt file
    os.makedirs(output_dir, exist_ok=True)
    full_output_path = os.path.join(output_dir, output_filename)

    save_data = {
        'images': images_tensor,
        'labels': labels_tensor,
        'image_paths': all_image_paths_list, # Storing relative paths
        'class_mapping': {v: k for k, v in label_mapping.items()} # Invert for easy lookup
    }

    torch.save(save_data, full_output_path)
    print(f"\nProcessed data saved to: {full_output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess agricultural product images and save to a .pt file.")
    parser.add_argument('--product_name', type=str, required=True, help="Name of the product (e.g., 'product_A').")
    parser.add_argument('--raw_data_root', type=str, default="./data/raw/", help="Root directory of raw image data.")
    parser.add_argument('--output_data_root', type=str, default="./data/processed/", help="Directory to save the processed .pt file.")
    parser.add_argument('--image_size', type=int, default=224, help="Target image size.")
    parser.add_argument('--split', type=str, default="train", choices=["train", "val", "test"], help="Data split to process (cosmetic for filename, assumes dirs exist).")

    args = parser.parse_args()

    authentic_dir_name = f"authentic_{args.product_name}" # e.g., authentic_product_A
    counterfeit_dir_name = f"counterfeit_{args.product_name}" # e.g., counterfeit_product_A

    # Construct full paths
    # You might need more sophisticated logic if your train/val/test splits are in subdirectories
    # e.g., ./data/raw/train/authentic_product_A/
    full_authentic_dir = os.path.join(args.raw_data_root, authentic_dir_name)
    full_counterfeit_dir = os.path.join(args.raw_data_root, counterfeit_dir_name)

    output_filename = f"{args.product_name}_{args.split}.pt" # e.g., product_A_train.pt

    print(f"--- Starting Preprocessing for {args.product_name} ({args.split} set) ---")
    create_processed_pt_file(
        authentic_dir=full_authentic_dir,
        counterfeit_dir=full_counterfeit_dir,
        output_dir=args.output_data_root,
        output_filename=output_filename,
        image_size=args.image_size
    )
    print(f"--- Preprocessing for {args.product_name} ({args.split} set) Complete ---")

    # Example command to run this script:
    # python src/data_utils/preprocess_data.py --product_name product_A --split train
    #
    # This assumes you have:
    # ./data/raw/authentic_product_A/ (with images)
    # ./data/raw/counterfeit_product_A/ (with images)
    # And it will create:
    # ./data/processed/product_A_train.pt