# data_utils/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd # For loading few-shot task CSVs

# --- Standard Dataset for loading from .pt files ---
class ProcessedProductDataset(Dataset):
    """
    Dataset to load preprocessed image tensors and labels from a .pt file.
    The .pt file is expected to contain a dictionary with 'images' and 'labels' tensors.
    """
    def __init__(self, pt_file_path, transform=None):
        """
        Args:
            pt_file_path (string): Path to the .pt file.
            transform (callable, optional): Optional transform to be applied on a sample.
                                            (Usually, if data is preprocessed, this might be None
                                             or used for on-the-fly augmentations not saved in .pt)
        """
        if not os.path.exists(pt_file_path):
            raise FileNotFoundError(f"The file {pt_file_path} does not exist.")

        try:
            self.data_dict = torch.load(pt_file_path)
        except Exception as e:
            raise IOError(f"Error loading .pt file {pt_file_path}: {e}")

        if 'images' not in self.data_dict or 'labels' not in self.data_dict:
            raise ValueError("The .pt file must contain 'images' and 'labels' keys.")

        self.images = self.data_dict['images']
        self.labels = self.data_dict['labels']
        self.transform = transform

        if len(self.images) != len(self.labels):
            raise ValueError("Mismatch in the number of images and labels.")

        print(f"Loaded dataset from {pt_file_path}: {len(self.images)} samples.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # If images in .pt are already tensors and preprocessed,
        # additional transforms here might be for specific augmentations
        # or if the saved tensors are not in the final desired format.
        if self.transform:
            # If images are PIL Images (not tensors) in .pt, this would be:
            # image = self.transform(image)
            # If images are tensors and transform expects PIL, you might need:
            # from torchvision.transforms.functional import to_pil_image
            # image = self.transform(to_pil_image(image))
            # For now, assuming transform can handle tensor input if provided
            image = self.transform(image)


        return image, label

# --- Dataset for Few-Shot Learning Tasks from CSV ---
class FewShotTaskDataset(Dataset):
    """
    Dataset to load images for a specific few-shot task defined in a CSV file.
    The CSV should specify image paths and class names.
    """
    def __init__(self, task_csv_path, data_root_path, image_transform=None, set_type="all"):
        """
        Args:
            task_csv_path (string): Path to the CSV file defining the task.
            data_root_path (string): Root directory where image paths in CSV are relative to.
                                     (Often the project root or data/raw/)
            image_transform (callable, optional): Transform to apply to images.
            set_type (string): "support", "query", or "all" to load specific parts of the task.
        """
        self.task_df = pd.read_csv(task_csv_path)
        self.data_root_path = data_root_path
        self.image_transform = image_transform

        if set_type != "all":
            self.task_df = self.task_df[self.task_df['set_type'] == set_type].reset_index(drop=True)

        if self.task_df.empty:
            raise ValueError(f"No data found for set_type '{set_type}' in {task_csv_path}")

        # Create a mapping from class_name to integer label for this task
        self.unique_classes = sorted(self.task_df['class_name'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.unique_classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.unique_classes)}

        print(f"Loaded few-shot task from {task_csv_path} for set_type '{set_type}'. Found {len(self.task_df)} samples across {len(self.unique_classes)} classes.")
        print(f"Class mapping for this task: {self.class_to_idx}")


    def __len__(self):
        return len(self.task_df)

    def __getitem__(self, idx):
        row = self.task_df.iloc[idx]
        img_relative_path = row['file_path']
        class_name = row['class_name']

        img_full_path = os.path.join(self.data_root_path, img_relative_path)

        try:
            image = Image.open(img_full_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found at {img_full_path} (relative: {img_relative_path}) as specified in {os.path.basename(self.task_df.attrs.get('csv_path', 'task CSV'))}")


        if self.image_transform:
            image = self.image_transform(image)

        label = self.class_to_idx[class_name]

        # Optionally, return more info
        # return image, label, class_name, img_relative_path
        return image, label

    def get_way(self):
        return len(self.unique_classes)

# --- Helper function to create DataLoaders ---
def get_dataloader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True):
    """
    Creates a DataLoader for a given dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle # Drop last incomplete batch if shuffling (for training)
    )

# --- Example Usage (can be run directly for testing) ---
if __name__ == '__main__':
    # --- Test ProcessedProductDataset ---
    print("--- Testing ProcessedProductDataset ---")
    # First, create a dummy .pt file (similar to your create_product_A_pt.py script)
    dummy_pt_path = "./dummy_train_data.pt"
    if not os.path.exists(dummy_pt_path):
        print(f"Creating dummy data file at {dummy_pt_path} for testing...")
        num_samples = 10
        dummy_images = torch.randn(num_samples, 3, 224, 224) # N, C, H, W
        dummy_labels = torch.randint(0, 2, (num_samples,))
        torch.save({'images': dummy_images, 'labels': dummy_labels}, dummy_pt_path)
        print("Dummy data file created.")

    try:
        processed_dataset = ProcessedProductDataset(pt_file_path=dummy_pt_path)
        if len(processed_dataset) > 0:
            img, lbl = processed_dataset[0]
            print(f"Sample from ProcessedProductDataset: Image shape: {img.shape}, Label: {lbl}")
            processed_loader = get_dataloader(processed_dataset, batch_size=4, shuffle=False)
            for batch_idx, (images, labels) in enumerate(processed_loader):
                print(f"ProcessedProductDataset - Batch {batch_idx}: Images shape: {images.shape}, Labels: {labels}")
                if batch_idx == 0: break # Just show one batch
        else:
            print("ProcessedProductDataset is empty.")
    except Exception as e:
        print(f"Error testing ProcessedProductDataset: {e}")
    finally:
        if os.path.exists(dummy_pt_path):
            os.remove(dummy_pt_path) # Clean up dummy file
            print(f"Cleaned up {dummy_pt_path}")


    # --- Test FewShotTaskDataset ---
    print("\n--- Testing FewShotTaskDataset ---")
    # Create a dummy CSV file for a few-shot task
    dummy_task_csv_path = "./dummy_few_shot_task.csv"
    dummy_data_root = "./" # Assuming images are relative to project root for this dummy
    
    # Create dummy image files for the CSV to point to
    os.makedirs(os.path.join(dummy_data_root, "data/raw/dummy_class_A"), exist_ok=True)
    os.makedirs(os.path.join(dummy_data_root, "data/raw/dummy_class_B"), exist_ok=True)
    Image.new('RGB', (60, 30), color = 'red').save(os.path.join(dummy_data_root, "data/raw/dummy_class_A/img1.png"))
    Image.new('RGB', (60, 30), color = 'blue').save(os.path.join(dummy_data_root, "data/raw/dummy_class_B/img2.png"))
    Image.new('RGB', (60, 30), color = 'green').save(os.path.join(dummy_data_root, "data/raw/dummy_class_A/img3.png"))

    csv_content = """set_type,class_name,file_path
support,class_A,data/raw/dummy_class_A/img1.png
support,class_B,data/raw/dummy_class_B/img2.png
query,class_A,data/raw/dummy_class_A/img3.png
"""
    with open(dummy_task_csv_path, 'w') as f:
        f.write(csv_content)
    
    # Define a simple transform for testing FewShotTaskDataset
    from torchvision import transforms
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    try:
        few_shot_support_dataset = FewShotTaskDataset(
            task_csv_path=dummy_task_csv_path,
            data_root_path=dummy_data_root,
            image_transform=test_transform,
            set_type="support"
        )
        if len(few_shot_support_dataset) > 0:
            img_fs, lbl_fs = few_shot_support_dataset[0]
            print(f"Sample from FewShotTaskDataset (support): Image shape: {img_fs.shape}, Label: {lbl_fs}")
            print(f"Task Way: {few_shot_support_dataset.get_way()}")

        few_shot_query_dataset = FewShotTaskDataset(
            task_csv_path=dummy_task_csv_path,
            data_root_path=dummy_data_root,
            image_transform=test_transform,
            set_type="query"
        )
        if len(few_shot_query_dataset) > 0:
            img_fs_q, lbl_fs_q = few_shot_query_dataset[0]
            print(f"Sample from FewShotTaskDataset (query): Image shape: {img_fs_q.shape}, Label: {lbl_fs_q}")
            
            few_shot_loader = get_dataloader(few_shot_query_dataset, batch_size=2, shuffle=False)
            for batch_idx, (images, labels) in enumerate(few_shot_loader):
                print(f"FewShotTaskDataset - Batch {batch_idx}: Images shape: {images.shape}, Labels: {labels}")
                if batch_idx == 0: break

    except Exception as e:
        print(f"Error testing FewShotTaskDataset: {e}")
    finally:
        # Clean up dummy files for FewShotTaskDataset
        if os.path.exists(dummy_task_csv_path):
            os.remove(dummy_task_csv_path)
            print(f"Cleaned up {dummy_task_csv_path}")
        
        img_paths_to_remove = [
            os.path.join(dummy_data_root, "data/raw/dummy_class_A/img1.png"),
            os.path.join(dummy_data_root, "data/raw/dummy_class_B/img2.png"),
            os.path.join(dummy_data_root, "data/raw/dummy_class_A/img3.png")
        ]
        for p in img_paths_to_remove:
            if os.path.exists(p): os.remove(p)
        if os.path.exists(os.path.join(dummy_data_root, "data/raw/dummy_class_A")):
            os.rmdir(os.path.join(dummy_data_root, "data/raw/dummy_class_A"))
        if os.path.exists(os.path.join(dummy_data_root, "data/raw/dummy_class_B")):
            os.rmdir(os.path.join(dummy_data_root, "data/raw/dummy_class_B"))
        # Potentially remove data/raw if it was created solely for this test and is empty
        if os.path.exists(os.path.join(dummy_data_root, "data/raw")) and not os.listdir(os.path.join(dummy_data_root, "data/raw")):
            os.rmdir(os.path.join(dummy_data_root, "data/raw"))
        if os.path.exists(os.path.join(dummy_data_root, "data")) and not os.listdir(os.path.join(dummy_data_root, "data")):
            os.rmdir(os.path.join(dummy_data_root, "data"))

        print(f"Cleaned up dummy image files and directories for FewShotTaskDataset.")