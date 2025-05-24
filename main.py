# main.py

import argparse
import os
import sys

# Add src to Python path to allow for absolute imports
# This assumes main.py is in the project root and src is a subdirectory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import the main functions from your training and evaluation scripts
from training.train import main as train_main
from evaluation.evaluate import main as evaluate_main
from data_utils.preprocess_data import main as preprocess_main # If you want to run preprocessing too

def main():
    parser = argparse.ArgumentParser(description="RAC Model - Main Runner")
    parser.add_argument(
        'mode',
        type=str,
        choices=['train', 'evaluate', 'preprocess'],
        help="Mode to run: 'train' for training, 'evaluate' for evaluation, 'preprocess' for data preprocessing."
    )
    parser.add_argument(
        '--config',
        type=str,
        required=False, # Not required for preprocess if it has its own defaults or args
        help="Path to the experiment-specific YAML configuration file (e.g., src/configs/rac_resnet50_config.yaml). Required for 'train' and 'evaluate'."
    )
    parser.add_argument(
        '--base_config',
        type=str,
        default="src/configs/base_config.yaml",
        help="Path to the base YAML configuration file."
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=False, # Only required for 'evaluate' mode or resuming training
        help="Path to the model checkpoint (.pth.tar file). Required for 'evaluate' and optionally for 'train' (resume)."
    )

    # --- Arguments specific to 'preprocess' mode ---
    parser.add_argument(
        '--product_name',
        type=str,
        required=False, # Only relevant for preprocess mode
        help="Name of the product to preprocess (e.g., 'product_A')."
    )
    parser.add_argument(
        '--raw_data_root',
        type=str,
        default="./data/raw/", # Default if not specified
        help="Root directory of raw image data for preprocessing."
    )
    parser.add_argument(
        '--output_data_root',
        type=str,
        default="./data/processed/", # Default if not specified
        help="Directory to save the processed .pt file during preprocessing."
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=224, # Default if not specified
        help="Target image size for preprocessing."
    )
    parser.add_argument(
        '--split',
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Data split to process during preprocessing (e.g., 'train', 'val', 'test')."
    )
    # --- Add other arguments as needed for different modes ---
    # Example: --test_dataset_name for evaluation mode, if it's different from train config
    parser.add_argument(
        '--test_dataset_name',
        type=str,
        default=None,
        help="Specific name for the test dataset if different from training/val (e.g., product_A_ood). Used in 'evaluate' mode."
    )


    args = parser.parse_args()

    if args.mode == 'train':
        if not args.config:
            parser.error("--config is required for train mode.")
        # Construct a pseudo-sys.argv for train_main
        # train_main expects '--config' and '--base_config'
        # If resuming, it will pick up 'resume_checkpoint' from the config file itself.
        # Or you could add a --resume argument here.
        sys.argv = ['src/training/train.py', '--config', args.config, '--base_config', args.base_config]
        if args.checkpoint: # Allow overriding resume checkpoint from command line
            # This requires train.py to handle a --resume_checkpoint argument
            # or modifying the loaded config. For simplicity, assume config handles resume.
            # A simpler approach might be for train.py to look for 'resume_checkpoint' in its config.
             print(f"Note: For resuming training, ensure 'resume_checkpoint: {args.checkpoint}' is set in your YAML config or handled by train.py.")
        train_main()
    elif args.mode == 'evaluate':
        if not args.config or not args.checkpoint:
            parser.error("--config and --checkpoint are required for evaluate mode.")
        # Construct a pseudo-sys.argv for evaluate_main
        sys.argv = ['src/evaluation/evaluate.py', '--config', args.config, '--base_config', args.base_config, '--checkpoint', args.checkpoint]
        if args.test_dataset_name:
            sys.argv.extend(['--test_dataset_name', args.test_dataset_name])
        evaluate_main()
    elif args.mode == 'preprocess':
        if not args.product_name:
            parser.error("--product_name is required for preprocess mode.")
        # Construct a pseudo-sys.argv for preprocess_main
        sys.argv = [
            'src/data_utils/preprocess_data.py',
            '--product_name', args.product_name,
            '--raw_data_root', args.raw_data_root,
            '--output_data_root', args.output_data_root,
            '--image_size', str(args.image_size),
            '--split', args.split
        ]
        preprocess_main()
    else:
        print(f"Unknown mode: {args.mode}")
        parser.print_help()

if __name__ == '__main__':
    main()