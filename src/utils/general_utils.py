# src/utils/general_utils.py

import os
import random
import numpy as np
import torch
import yaml
import shutil # For robustly saving checkpoints

def set_seed(seed_value=42):
    """
    Sets the seed for reproducibility in PyTorch, NumPy, and Python's random module.

    Args:
        seed_value (int): The seed value to use.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        # The following two lines are often used for deterministic behavior on CUDA,
        # but can impact performance. Enable if strict reproducibility is paramount.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed_value}")

def save_config(config_dict, file_path):
    """
    Saves a configuration dictionary to a YAML file.

    Args:
        config_dict (dict): The configuration dictionary to save.
        file_path (str): Path to the YAML file where the config will be saved.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, indent=2, sort_keys=False)
        print(f"Configuration saved to {file_path}")
    except Exception as e:
        print(f"Error saving configuration to {file_path}: {e}")

def load_yaml_config(file_path):
    """
    Loads a configuration dictionary from a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: The loaded configuration dictionary, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        print(f"Configuration loaded from {file_path}")
        return config_dict
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading configuration from {file_path}: {e}")
        return None

def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth.tar"):
    """
    Saves model and training parameters at checkpoint.

    Args:
        state (dict): Contains model's state_dict, optimizer's state_dict, epoch, etc.
        is_best (bool): True if this is the best model seen so far (e.g., based on validation metric).
        checkpoint_dir (str): Directory to save checkpoints.
        filename (str, optional): Filename for the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, "best_model.pth.tar")
        shutil.copyfile(filepath, best_filepath)
        print(f"Best model checkpoint updated to {best_filepath}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu', strict=True):
    """
    Loads model and training parameters from a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): Model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer instance to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler instance to load state into.
        device (str or torch.device, optional): Device to load the model onto.
        strict (bool, optional): Whether to strictly enforce that the keys in state_dict match the keys returned by this module’s state_dict(). Default True.

    Returns:
        tuple: (start_epoch, best_metric) loaded from checkpoint. Returns (0, -float('inf')) if keys not found.
    """
    if not os.path.isfile(checkpoint_path):
        print(f"Warning: Checkpoint file not found at {checkpoint_path}. No state loaded.")
        return 0, -float('inf') # Default values if checkpoint doesn't exist

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'state_dict' in checkpoint:
        # Handle potential DataParallel or DDP wrapped models
        state_dict = checkpoint['state_dict']
        # Create new OrderedDict that does not contain `module.` prefix
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        is_parallel_saved = False
        for k, v in state_dict.items():
            if k.startswith('module.'):
                is_parallel_saved = True
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        
        if is_parallel_saved:
            print("Checkpoint was saved from a DataParallel/DDP model. Removing 'module.' prefix.")
        
        # Check if the current model is wrapped (e.g. in DataParallel)
        is_current_model_parallel = isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))

        if is_current_model_parallel and not is_parallel_saved:
            print("Current model is DataParallel/DDP, but checkpoint is not. Adding 'module.' prefix.")
            # This case is less common for loading, usually it's the other way around.
            # For simplicity, this example doesn't add the prefix automatically when loading into a parallel model.
            # Usually, you'd unwrap the model before saving or wrap it after loading.
            # Or handle it like: model.module.load_state_dict(new_state_dict, strict=strict)
            # However, it's safer to load into the non-parallel model first.
            try:
                model.module.load_state_dict(new_state_dict, strict=strict)
            except AttributeError: # Not actually wrapped, or different wrapping
                 model.load_state_dict(new_state_dict, strict=strict)

        elif not is_current_model_parallel and is_parallel_saved:
            model.load_state_dict(new_state_dict, strict=strict) # Already handled by removing 'module.'
        else: # Both same (either both wrapped or both not, or current is not wrapped and save is not wrapped)
            model.load_state_dict(new_state_dict, strict=strict)
        
        print("Model weights loaded successfully.")
    else:
        print("Warning: 'state_dict' not found in checkpoint. Model weights not loaded.")

    if optimizer and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Optimizer state loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}. Optimizer state might be incompatible.")


    if scheduler and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("Scheduler state loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}. Scheduler state might be incompatible.")


    start_epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_val_metric', -float('inf')) # Default to -inf if not present

    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}, best metric so far: {best_metric:.4f}")
    return start_epoch, best_metric


def get_device(config_device_str="cuda"):
    """
    Determines and returns the appropriate torch device.
    Args:
        config_device_str (str): Desired device ("cuda" or "cpu").
    Returns:
        torch.device: The selected torch device.
    """
    if config_device_str.lower() == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        if config_device_str.lower() == "cuda":
            print("CUDA requested but not available. Using CPU.")
        device = torch.device("cpu")
        print("Using CPU device.")
    return device


# --- Example Usage (can be run directly for testing) ---
if __name__ == '__main__':
    print("--- Testing General Utilities ---")

    # Test set_seed
    set_seed(123)
    a = random.random()
    b = np.random.rand()
    c = torch.rand(1)
    set_seed(123)
    assert a == random.random(), "Python random seed consistency failed."
    assert b == np.random.rand(), "NumPy random seed consistency failed."
    assert c.item() == torch.rand(1).item(), "PyTorch random seed consistency failed."
    print("set_seed test passed.")

    # Test config save/load
    dummy_config_path = "./temp_test_config.yaml"
    dummy_config_data = {'param1': 10, 'model': {'type': 'ResNet', 'layers': 50}, 'lr': 0.001}
    save_config(dummy_config_data, dummy_config_path)
    loaded_config_data = load_yaml_config(dummy_config_path)
    assert dummy_config_data == loaded_config_data, "Config save/load consistency failed."
    print("Config save/load test passed.")
    if os.path.exists(dummy_config_path):
        os.remove(dummy_config_path)

    # Test checkpoint save/load (simplified model)
    print("\n--- Testing Checkpoint Save/Load ---")
    test_model = torch.nn.Linear(5, 2)
    test_optimizer = torch.optim.Adam(test_model.parameters(), lr=0.01)
    test_scheduler = torch.optim.lr_scheduler.StepLR(test_optimizer, step_size=1)
    checkpoint_test_dir = "./temp_checkpoints_general_utils/"
    
    # Save
    state_to_save = {
        'epoch': 5,
        'state_dict': test_model.state_dict(),
        'optimizer': test_optimizer.state_dict(),
        'scheduler': test_scheduler.state_dict(),
        'best_val_metric': 0.85,
        'config': dummy_config_data
    }
    save_checkpoint(state_to_save, True, checkpoint_test_dir, filename="test_checkpoint.pth.tar")

    # Load
    new_model = torch.nn.Linear(5, 2) # Fresh model
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.1) # Different LR initially
    new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=5)

    start_ep, best_m = load_checkpoint(
        os.path.join(checkpoint_test_dir, "test_checkpoint.pth.tar"),
        new_model, new_optimizer, new_scheduler
    )
    assert start_ep == 5, "Epoch loading failed."
    assert best_m == 0.85, "Best metric loading failed."
    # Check if optimizer LR changed (would be 0.01 after loading if successful and params match)
    assert new_optimizer.param_groups[0]['lr'] == 0.01, "Optimizer state loading failed."
    # Check scheduler step_size
    assert new_scheduler.step_size == 1, "Scheduler state loading failed."
    # Check model weights (a simple check)
    assert torch.allclose(test_model.weight, new_model.weight), "Model weight loading failed."
    print("Checkpoint save/load test passed.")

    if os.path.exists(checkpoint_test_dir):
        shutil.rmtree(checkpoint_test_dir) # Clean up

    # Test get_device
    print("\n--- Testing get_device ---")
    device_test = get_device("cuda") # Will fallback to CPU if no CUDA
    print(f"Device selected by get_device: {device_test}")


    print("\nGeneral utilities test completed.")