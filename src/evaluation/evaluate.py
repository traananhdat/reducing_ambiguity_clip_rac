import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sklearn.metrics as sk_metrics

# Project-specific imports
from src.data_utils.dataset import ProcessedProductDataset, get_dataloader # Assuming standard dataset loading
from src.data_utils.augmentations import get_val_test_transforms # Or get_clip_preprocess
from src.models.rac_model import RACModel
from src.models.clip_backbone import load_clip_model
from src.models.utils import generate_text_features_for_prompts # Utility from your project
from src.utils.logging_utils import setup_logger, log_metrics
from src.utils.general_utils import load_checkpoint # For loading the trained model
from src.evaluation.metrics import ( # Assuming these are defined in metrics.py
    calculate_accuracy,
    calculate_precision_recall_f1,
    plot_confusion_matrix
)


def load_config(base_cfg_path, exp_cfg_path):
    """Loads base and experiment-specific YAML configurations."""
    with open(base_cfg_path, 'r') as f:
        base_cfg = yaml.safe_load(f)
    # Experiment config can be loaded from the checkpoint or a separate file
    if os.path.exists(exp_cfg_path):
        with open(exp_cfg_path, 'r') as f:
            exp_cfg = yaml.safe_load(f)
        # Merge configs: experiment config overrides base config
        config = {**base_cfg, **exp_cfg}
        if 'rac_modules' in base_cfg and 'rac_modules' in exp_cfg: # Ensure nested dicts are merged
             config['rac_modules'] = {**base_cfg['rac_modules'], **exp_cfg['rac_modules']}
    else: # If no separate experiment config path, assume base_cfg is enough (or loaded from checkpoint)
        config = base_cfg
    return config

def evaluate_model(config, model_checkpoint_path, logger):
    """Loads a trained model and evaluates it on the test dataset."""
    logger.info("Starting evaluation...")
    logger.info(f"Loading model from checkpoint: {model_checkpoint_path}")
    logger.info(f"Using configuration:\n{yaml.dump(config, indent=2)}")

    # --- Setup ---
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")

    # --- Data Loading ---
    logger.info("Loading data...")
    clip_backbone, clip_preprocess_fn = load_clip_model(
        model_name=config['clip_model_name'],
        device=device,
        extract_multi_level=config['rac_modules']['mfa']['enabled'] and config['rac_modules']['mfa'].get('num_levels', 0) > 0,
        num_levels=config['rac_modules']['mfa'].get('num_levels', 4)
    )

    # Use validation/test transforms (typically no heavy augmentation)
    test_transform = get_val_test_transforms(config['image_size']) # Or clip_preprocess_fn

    test_pt_file = os.path.join(config['data_root'], 'processed', f"{config.get('test_dataset_name', config['dataset_name'])}_test.pt")
    logger.info(f"Loading test data from: {test_pt_file}")
    if not os.path.exists(test_pt_file):
        logger.error(f"Test data file not found: {test_pt_file}")
        return

    test_dataset = ProcessedProductDataset(pt_file_path=test_pt_file) #, transform=test_transform if needed)
    test_loader = get_dataloader(test_dataset, config.get('eval_batch_size', config['batch_size']), shuffle=False, num_workers=config.get('num_workers', 4))

    class_names = config.get('class_names', [f"class_{i}" for i in range(config['num_classes'])])
    text_prompts = [f"a photo of a {name.replace('_', ' ')}" for name in class_names]
    text_features_T = generate_text_features_for_prompts(clip_backbone, text_prompts, device)
    logger.info(f"Generated text features for {len(class_names)} classes.")

    # --- Model Initialization and Loading ---
    logger.info("Initializing model...")
    model = RACModel(config, clip_backbone, num_classes=config['num_classes']).to(device)

    # Load the trained weights
    # The load_checkpoint function might also return optimizer etc., but we only need the model state
    if os.path.isfile(model_checkpoint_path):
        # Pass model to load_checkpoint so it can load state_dict directly
        # Assuming load_checkpoint can handle loading only model state if optimizer is None
        _ = load_checkpoint(model_checkpoint_path, model, optimizer=None, scheduler=None, device=device)
        logger.info(f"Successfully loaded model weights from {model_checkpoint_path}")
    else:
        logger.error(f"Model checkpoint not found at {model_checkpoint_path}")
        return

    model.eval() # Set model to evaluation mode

    # --- Evaluation ---
    all_preds = []
    all_labels = []
    all_logits = [] # To store raw logits for more detailed analysis if needed

    progress_bar = tqdm(test_loader, desc="Evaluating on Test Set", leave=False)
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Handle multi-level features if your RACModel's forward expects them
            # For simplicity, assuming RACModel can handle multi_level_features_f=None
            # or that clip_backbone.encode_image in RACModel internally gets them if configured.
            final_logits, total_loss, loss_dict = model(images, multi_level_features_f=None, text_features_T=text_features_T, labels_y=labels)
            # Note: total_loss and loss_dict might be computed even in eval if labels are passed;
            # primarily interested in final_logits for predictions.

            preds = torch.argmax(final_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(final_logits.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)

    # --- Calculate Metrics ---
    logger.info("Calculating metrics...")
    accuracy = calculate_accuracy(all_preds, all_labels)
    precision, recall, f1, support = calculate_precision_recall_f1(all_labels, all_preds, average='weighted') # Or 'macro' or None for per-class
    classification_report_str = sk_metrics.classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)

    metrics = {
        'test_accuracy': accuracy,
        'test_precision_weighted': precision,
        'test_recall_weighted': recall,
        'test_f1_weighted': f1,
    }
    if support is not None: # Support is None if average is not None for precision_recall_fscore
        metrics['test_support_weighted'] = support


    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test F1 Score (Weighted): {f1:.4f}")
    logger.info(f"Test Precision (Weighted): {precision:.4f}")
    logger.info(f"Test Recall (Weighted): {recall:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report_str}")

    # --- Plot Confusion Matrix ---
    output_plots_dir = os.path.join(config.get('results_dir', './results/'), config['experiment_name'], 'plots')
    os.makedirs(output_plots_dir, exist_ok=True)
    cm_filename = os.path.join(output_plots_dir, f"confusion_matrix_test_{config['experiment_name']}.png")
    plot_confusion_matrix(all_labels, all_preds, class_names, filename=cm_filename, logger=logger)

    logger.info("Evaluation completed.")
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAC Model")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to the experiment-specific YAML configuration file "
                             "(can be the one saved with the checkpoint or a new one for evaluation).")
    parser.add_argument('--base_config', type=str, default="src/configs/base_config.yaml",
                        help="Path to the base YAML configuration file.")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the model checkpoint (.pth.tar file) to evaluate.")
    parser.add_argument('--test_dataset_name', type=str, default=None,
                        help="Specific name for the test dataset if different from training/val (e.g., product_A_ood). Defaults to config['dataset_name'].")

    args = parser.parse_args()

    # Load config from the YAML file specified
    config = load_config(args.base_config, args.config)

    # If a config was saved in the checkpoint, it often takes precedence for model architecture
    # For simplicity, we're using the passed config file, assuming it matches the model architecture.
    # A more robust way is to load config from checkpoint and merge.
    if os.path.exists(args.checkpoint):
        checkpoint_data = torch.load(args.checkpoint, map_location='cpu')
        if 'config' in checkpoint_data:
            logger_temp = setup_logger("temp_config_load", "./")
            logger_temp.info("Loading model architecture config from checkpoint and merging with evaluation config.")
            model_arch_config = checkpoint_data['config']
            # Merge: eval config can override data paths, batch sizes, etc.
            # but model-specific params should ideally come from training config
            config = {**model_arch_config, **config} # eval config can override
            # Re-merge rac_modules specifically to ensure deep merge if they exist in both
            if 'rac_modules' in model_arch_config and 'rac_modules' in yaml.safe_load(open(args.config, 'r')):
                exp_rac_modules = yaml.safe_load(open(args.config, 'r')).get('rac_modules', {})
                config['rac_modules'] = {**model_arch_config.get('rac_modules',{}), **exp_rac_modules}

    if args.test_dataset_name:
        config['test_dataset_name'] = args.test_dataset_name # Override for specific test set

    # Setup logger
    log_dir = config.get('log_dir', './results/logs/')
    # Use a different log file name for evaluation
    eval_log_name = f"eval_{config['experiment_name']}_{os.path.splitext(os.path.basename(args.checkpoint))[0]}"
    logger = setup_logger(eval_log_name, log_dir)

    try:
        evaluate_model(config, args.checkpoint, logger)
    except Exception as e:
        logger.exception(f"An error occurred during evaluation: {e}")
        raise


if __name__ == '__main__':
    # Placeholder for generate_text_features_for_prompts if not in utils
    if not hasattr(torch.nn.Module, 'generate_text_features_for_prompts_eval'):
        def temp_generate_text_features_eval(clip_model_wrapper, prompts, device):
            logger_dummy = setup_logger("dummy_text_feat_gen_eval", "./")
            logger_dummy.info(f"Generating text features for {len(prompts)} prompts (eval)...")
            with torch.no_grad():
                text_features = clip_model_wrapper.encode_text(prompts)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            logger_dummy.info("Text features generated and normalized (eval).")
            return text_features.to(device)

        from src.models import utils as model_utils_eval
        if not hasattr(model_utils_eval, 'generate_text_features_for_prompts'):
             model_utils_eval.generate_text_features_for_prompts = temp_generate_text_features_eval

    main()

    # Example Command:
    # python src/evaluation/evaluate.py \
    #   --config src/configs/rac_resnet50_config.yaml \
    #   --checkpoint ./saved_models/rac_resnet50_agricultural_product_A_16shot/best_model.pth.tar \
    #   --test_dataset_name product_A # Assumes product_A_test.pt exists