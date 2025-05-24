# src/training/train.py

import os
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# Project-specific imports (adjust paths if your structure differs)
from src.data_utils.dataset import ProcessedProductDataset, get_dataloader # Assuming standard dataset loading
# from src.data_utils.dataset import FewShotTaskDataset # If doing few-shot episodic training
from src.data_utils.augmentations import get_train_transforms, get_val_test_transforms # Or get_clip_preprocess
from src.models.rac_model import RACModel 
from src.models.clip_backbone import load_clip_model
from src.models.utils import generate_text_features_for_prompts # A new utility you might need
from src.training.optimizers import create_optimizer, create_scheduler
from src.training.losses import compute_rac_loss # If you centralize loss computation, or compute in model
from src.utils.logging_utils import setup_logger, log_metrics
from src.utils.general_utils import save_config, save_checkpoint, load_checkpoint

# --- Configuration Loading ---
def load_config(base_cfg_path, exp_cfg_path):
    """Loads base and experiment-specific YAML configurations."""
    with open(base_cfg_path, 'r') as f:
        base_cfg = yaml.safe_load(f)
    with open(exp_cfg_path, 'r') as f:
        exp_cfg = yaml.safe_load(f)

    # Merge configs: experiment config overrides base config
    # Deep merge for nested dictionaries if necessary, or simple update
    config = {**base_cfg, **exp_cfg}
    # For nested dicts like 'rac_modules', ensure they are merged properly
    if 'rac_modules' in base_cfg and 'rac_modules' in exp_cfg:
        config['rac_modules'] = {**base_cfg['rac_modules'], **exp_cfg['rac_modules']}
    return config

# --- Main Training Loop ---
def run_training_loop(config, logger):
    """Orchestrates the main training and validation loop."""
    logger.info("Starting training loop...")
    logger.info(f"Configuration:\n{yaml.dump(config, indent=2)}")

    # --- Setup ---
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")

    # Seed for reproducibility
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Directories for saving
    experiment_dir = os.path.join(config.get('checkpoint_dir', './saved_models/'), config['experiment_name'])
    os.makedirs(experiment_dir, exist_ok=True)
    save_config(config, os.path.join(experiment_dir, 'config_used.yaml'))

    # --- Data Loading ---
    logger.info("Loading data...")
    # Load CLIP model and its preprocessing function (used for images and text prompts)
    clip_backbone, clip_preprocess_fn = load_clip_model(
        model_name=config['clip_model_name'],
        device=device,
        # Enable multi-level extraction if MFA needs it (this is a conceptual link)
        extract_multi_level=config['rac_modules']['mfa']['enabled'] and config['rac_modules']['mfa'].get('num_levels', 0) > 0,
        num_levels=config['rac_modules']['mfa'].get('num_levels', 4)
    )
    # Note: clip_preprocess_fn from load_clip_model should be used for consistency if images are loaded raw.
    # If ProcessedProductDataset uses .pt files with already CLIP-processed images,
    # then the transforms below might be for additional augmentation or identity.

    # Define image transforms (could use clip_preprocess_fn or custom ones)
    # If .pt files store raw-ish tensors, apply full transforms.
    # If .pt files store CLIP-preprocessed tensors, transforms might be minimal or None.
    train_transform = get_train_transforms(config['image_size']) # Or use clip_preprocess_fn directly if no further aug
    val_transform = get_val_test_transforms(config['image_size'])   # Or use clip_preprocess_fn

    # Create datasets
    # Assuming .pt files are structured as 'product_name_split.pt'
    train_pt_file = os.path.join(config['data_root'], 'processed', f"{config['dataset_name']}_train.pt")
    val_pt_file = os.path.join(config['data_root'], 'processed', f"{config['dataset_name']}_val.pt")

    train_dataset = ProcessedProductDataset(pt_file_path=train_pt_file) #, transform=train_transform if needed)
    val_dataset = ProcessedProductDataset(pt_file_path=val_pt_file) #, transform=val_transform if needed)

    train_loader = get_dataloader(train_dataset, config['batch_size'], shuffle=True, num_workers=config.get('num_workers', 4))
    val_loader = get_dataloader(val_dataset, config.get('eval_batch_size', config['batch_size']), shuffle=False, num_workers=config.get('num_workers', 4))

    # Generate text features for class prompts (e.g., "a photo of [class_name]")
    # This requires knowing your class names.
    # class_names = ["authentic_product_A", "counterfeit_product_A"] # Example
    class_names = config.get('class_names', [f"class_{i}" for i in range(config['num_classes'])])
    text_prompts = [f"a photo of a {name.replace('_', ' ')}" for name in class_names]
    text_features_T = generate_text_features_for_prompts(clip_backbone, text_prompts, device)
    logger.info(f"Generated text features for {len(class_names)} classes.")


    # --- Model Initialization ---
    logger.info("Initializing model...")
    model = RACModel(config, clip_backbone, num_classes=config['num_classes']).to(device)
    # clip_backbone parameters are already frozen by default in its wrapper

    # --- Optimizer and Scheduler ---
    learnable_params = model.get_learnable_parameters()
    if not learnable_params:
        logger.warning("Model has no learnable parameters! Check model.get_learnable_parameters() and module requires_grad settings.")
        # return # Or proceed if this is intentional (e.g., zero-shot eval)

    optimizer = create_optimizer(learnable_params, config)
    scheduler = create_scheduler(optimizer, config, steps_per_epoch=len(train_loader))
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scheduler: {scheduler.__class__.__name__ if scheduler else 'None'}")


    # --- Training ---
    best_val_metric = -float('inf') # Or float('inf') if monitoring loss
    epochs_no_improve = 0
    start_epoch = 0 # For resuming training

    # Optionally load from checkpoint
    if config.get('resume_checkpoint', None):
        checkpoint_path = config['resume_checkpoint']
        if os.path.isfile(checkpoint_path):
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            start_epoch, best_val_metric = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
            logger.info(f"Resumed. Start epoch: {start_epoch}, Best val metric: {best_val_metric:.4f}")
        else:
            logger.warning(f"Checkpoint not found at {checkpoint_path}. Starting from scratch.")


    logger.info(f"Learnable parameters: {sum(p.numel() for p in learnable_params if p.requires_grad)}")

    for epoch in range(start_epoch, config['epochs']):
        epoch_start_time = time.time()
        logger.info(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")

        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, device, text_features_T, config, logger, epoch)
        log_metrics(train_metrics, epoch + 1, 'Train', logger)

        val_metrics = validate_one_epoch(model, val_loader, device, text_features_T, config, logger)
        log_metrics(val_metrics, epoch + 1, 'Validation', logger)

        current_val_metric = val_metrics.get('val_accuracy', val_metrics.get('val_loss_total', 0)) # Prioritize accuracy

        is_best = current_val_metric > best_val_metric # Assuming higher is better for the metric
        if is_best:
            best_val_metric = current_val_metric
            epochs_no_improve = 0
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'best_val_metric': best_val_metric,
                'config': config
            }, True, experiment_dir, filename='best_model.pth.tar')
            logger.info(f"New best model saved with validation metric: {best_val_metric:.4f}")
        else:
            epochs_no_improve += 1

        # Save a checkpoint periodically
        if (epoch + 1) % config.get('save_interval', 5) == 0:
             save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'best_val_metric': best_val_metric, # Storing the current best, not necessarily this epoch's
                'config': config
            }, False, experiment_dir, filename=f'checkpoint_epoch_{epoch+1}.pth.tar')


        logger.info(f"Epoch {epoch+1} completed in {time.time() - epoch_start_time:.2f}s. Best Val Metric: {best_val_metric:.4f}")

        if config.get('early_stopping_patience', None) and epochs_no_improve >= config['early_stopping_patience']:
            logger.info(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
            break

    logger.info("Training completed.")
    logger.info(f"Best validation metric achieved: {best_val_metric:.4f}")


def train_one_epoch(model, data_loader, optimizer, scheduler, device, text_features_T, config, logger, epoch_num):
    model.train()
    total_loss_sum = 0.0
    all_preds = []
    all_labels = []
    loss_components_sum = {}

    progress_bar = tqdm(data_loader, desc=f"Training Epoch {epoch_num+1}", leave=False)
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        # For MFA, multi-level features might be needed.
        # If your clip_backbone.encode_image returns them:
        # global_image_features, multi_level_img_features = clip_backbone.encode_image(images)
        # Or if they are extracted inside RACModel using clip_backbone.
        # For simplicity, here we assume RACModel's forward handles feature extraction if needed.
        # The current RACModel forward expects `image_x` and optional `multi_level_features_f`.
        # If `multi_level_features_f` are to be extracted per batch:
        # with torch.no_grad(): # Assuming feature extraction part of CLIP is frozen
        #     _, multi_level_features = model.clip_backbone.encode_image(images) # If encode_image modified to return them

        optimizer.zero_grad()
        # Pass text_features_T to the model's forward method
        # Assuming multi_level_features_f might be None if not used or handled internally by RACModel
        final_logits, total_loss, loss_dict = model(images, multi_level_features_f=None, text_features_T=text_features_T, labels_y=labels)

        if total_loss is None:
            logger.error("Total loss is None. Check model's forward pass and loss calculation.")
            continue # or raise error

        total_loss.backward()
        # Gradient clipping (optional but often useful)
        if config.get('grad_clip_norm', None):
            torch.nn.utils.clip_grad_norm_(model.get_learnable_parameters(), config['grad_clip_norm'])
        optimizer.step()

        if scheduler and config.get('lr_scheduler_step_type', 'epoch') == 'batch':
            scheduler.step()

        total_loss_sum += total_loss.item()
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                loss_components_sum[k] = loss_components_sum.get(k, 0.0) + v.item()

        preds = torch.argmax(final_logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix(loss=total_loss.item())
        if batch_idx % config.get('log_interval', 50) == 0:
             logger.debug(f"Epoch {epoch_num+1}, Batch {batch_idx}/{len(data_loader)}, Batch Loss: {total_loss.item():.4f}")


    avg_loss = total_loss_sum / len(data_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    metrics = {'train_loss_total': avg_loss, 'train_accuracy': accuracy}
    for k, v_sum in loss_components_sum.items():
        metrics[f'train_loss_{k}'] = v_sum / len(data_loader)

    if scheduler and config.get('lr_scheduler_step_type', 'epoch') == 'epoch':
        scheduler.step()

    return metrics


def validate_one_epoch(model, data_loader, device, text_features_T, config, logger):
    model.eval()
    total_loss_sum = 0.0
    all_preds = []
    all_labels = []
    loss_components_sum = {}

    progress_bar = tqdm(data_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # As in training, handle multi-level features if needed by your RACModel
            # _, multi_level_features = model.clip_backbone.encode_image(images) if needed
            final_logits, total_loss, loss_dict = model(images, multi_level_features_f=None, text_features_T=text_features_T, labels_y=labels)

            if total_loss is not None:
                total_loss_sum += total_loss.item()
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        loss_components_sum[k] = loss_components_sum.get(k, 0.0) + v.item()


            preds = torch.argmax(final_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if total_loss is not None:
                progress_bar.set_postfix(loss=total_loss.item())


    avg_loss = total_loss_sum / len(data_loader) if len(data_loader) > 0 and total_loss is not None else 0.0
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100 if len(all_labels) > 0 else 0.0
    metrics = {'val_loss_total': avg_loss, 'val_accuracy': accuracy}

    for k, v_sum in loss_components_sum.items():
        metrics[f'val_loss_{k}'] = v_sum / len(data_loader) if len(data_loader) > 0 else 0.0

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train RAC Model")
    parser.add_argument('--config', type=str, required=True, help="Path to the experiment-specific YAML configuration file.")
    parser.add_argument('--base_config', type=str, default="src/configs/base_config.yaml", help="Path to the base YAML configuration file.")
    # Add other command-line overrides if needed, e.g., --device, --epochs
    args = parser.parse_args()

    config = load_config(args.base_config, args.config)

    # Setup logger
    log_dir = config.get('log_dir', './results/logs/')
    logger = setup_logger(config['experiment_name'], log_dir)

    try:
        run_training_loop(config, logger)
    except Exception as e:
        logger.exception(f"An error occurred during training: {e}")
        raise # Re-raise the exception after logging


if __name__ == '__main__':
    # Example of how to define generate_text_features_for_prompts in src.models.utils
    # This is a placeholder as it's not defined yet.
    if not hasattr(torch.nn.Module, 'generate_text_features_for_prompts'): # Avoid redefining if already added elsewhere
        def temp_generate_text_features(clip_model_wrapper, prompts, device):
            logger_dummy = setup_logger("dummy_text_feat_gen", "./") # dummy logger
            logger_dummy.info(f"Generating text features for {len(prompts)} prompts...")
            with torch.no_grad():
                text_features = clip_model_wrapper.encode_text(prompts) # Assumes encode_text handles tokenization
                text_features /= text_features.norm(dim=-1, keepdim=True)
            logger_dummy.info("Text features generated and normalized.")
            return text_features.to(device)
        # Monkey patch for this example run, ideally this util is properly defined
        from src.models import utils as model_utils # if utils.py exists
        if not hasattr(model_utils, 'generate_text_features_for_prompts'):
             model_utils.generate_text_features_for_prompts = temp_generate_text_features


    main()

    # Example Command:
    # python src/training/train.py --config src/configs/rac_resnet50_config.yaml