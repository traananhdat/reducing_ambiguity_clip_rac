# src/training/optimizers.py

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

def create_optimizer(learnable_params, config):
    """
    Creates an optimizer for the given model parameters based on the configuration.

    Args:
        learnable_params (iterable): Iterable of parameters to optimize.
                                     Typically model.parameters() or a filtered list.
        config (dict): Configuration dictionary, expected to contain:
                       config['optimizer'] (str): Optimizer type (e.g., "AdamW", "Adam", "SGD").
                       config['learning_rate'] (float): Initial learning rate.
                       config['weight_decay'] (float, optional): Weight decay.
                       config['optimizer_params'] (dict, optional): Additional params for optimizer.

    Returns:
        torch.optim.Optimizer: The created optimizer.
    """
    optimizer_name = config.get('optimizer', 'AdamW').lower()
    lr = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.01)
    optimizer_params = config.get('optimizer_params', {})

    if not list(learnable_params): # Check if the iterator is empty
        print("Warning: No learnable parameters provided to create_optimizer. Returning None.")
        return None

    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(learnable_params, lr=lr, weight_decay=weight_decay, **optimizer_params)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(learnable_params, lr=lr, weight_decay=weight_decay, **optimizer_params)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            learnable_params,
            lr=lr,
            momentum=optimizer_params.get('momentum', 0.9), # Common default for SGD
            weight_decay=weight_decay,
            nesterov=optimizer_params.get('nesterov', False),
            **{k:v for k,v in optimizer_params.items() if k not in ['momentum', 'nesterov']}
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return optimizer


def create_scheduler(optimizer, config, steps_per_epoch=None):
    """
    Creates a learning rate scheduler for the given optimizer based on the configuration.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to create the scheduler.
        config (dict): Configuration dictionary, expected to contain:
                       config['lr_scheduler'] (str, optional): Scheduler type
                                (e.g., "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau", "none").
                       config['lr_scheduler_params'] (dict, optional): Parameters for the scheduler.
        steps_per_epoch (int, optional): Number of steps (batches) per epoch.
                                         Required for schedulers like CosineAnnealingLR if T_max is not in params.

    Returns:
        torch.optim.lr_scheduler._LRScheduler or None: The created scheduler, or None.
    """
    if optimizer is None:
        return None

    scheduler_name = config.get('lr_scheduler', 'none').lower()
    scheduler_params = config.get('lr_scheduler_params', {})

    if scheduler_name == 'steplr':
        return StepLR(
            optimizer,
            step_size=scheduler_params.get('step_size', 30), # e.g., decay LR every 30 epochs
            gamma=scheduler_params.get('gamma', 0.1),       # e.g., multiply LR by 0.1
            **{k:v for k,v in scheduler_params.items() if k not in ['step_size', 'gamma']}
        )
    elif scheduler_name == 'cosineannealinglr':
        t_max = scheduler_params.get('T_max', None)
        if t_max is None:
            if steps_per_epoch is None:
                raise ValueError("steps_per_epoch must be provided for CosineAnnealingLR if T_max is not in params and total_epochs is used.")
            t_max = config.get('epochs', 50) * steps_per_epoch # T_max over all batches in training
        
        return CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=scheduler_params.get('eta_min', 0), # Minimum learning rate
            **{k:v for k,v in scheduler_params.items() if k not in ['T_max', 'eta_min']}
        )
    elif scheduler_name == 'reducelronplateau':
        return ReduceLROnPlateau(
            optimizer,
            mode=scheduler_params.get('mode', 'min'),      # 'min' for loss, 'max' for accuracy
            factor=scheduler_params.get('factor', 0.1),    # Factor by which the learning rate will be reduced
            patience=scheduler_params.get('patience', 10), # Number of epochs with no improvement after which LR will be reduced
            verbose=scheduler_params.get('verbose', True),
            **{k:v for k,v in scheduler_params.items() if k not in ['mode', 'factor', 'patience', 'verbose']}
        )
    elif scheduler_name == 'none':
        return None
    else:
        raise ValueError(f"Unsupported LR scheduler: {scheduler_name}")


# --- Example Usage (can be run directly for testing) ---
if __name__ == '__main__':
    # Create a dummy model and parameters for testing
    dummy_model = torch.nn.Linear(10, 2)
    dummy_params = list(dummy_model.parameters()) # Ensure it's an iterable list for the check

    print("--- Testing Optimizer Creation ---")
    # Test AdamW
    config_adamw = {'optimizer': 'AdamW', 'learning_rate': 1e-3, 'weight_decay': 1e-2}
    optimizer_adamw = create_optimizer(dummy_params, config_adamw)
    print(f"Created AdamW optimizer: {optimizer_adamw}")
    assert isinstance(optimizer_adamw, optim.AdamW)

    # Test SGD
    config_sgd = {
        'optimizer': 'SGD', 'learning_rate': 1e-2, 'weight_decay': 5e-4,
        'optimizer_params': {'momentum': 0.9, 'nesterov': True}
        }
    optimizer_sgd = create_optimizer(dummy_params, config_sgd)
    print(f"Created SGD optimizer: {optimizer_sgd}")
    assert isinstance(optimizer_sgd, optim.SGD)

    print("\n--- Testing Scheduler Creation ---")
    # Test StepLR
    config_steplr = {'lr_scheduler': 'StepLR', 'lr_scheduler_params': {'step_size': 10, 'gamma': 0.5}}
    scheduler_steplr = create_scheduler(optimizer_adamw, config_steplr)
    print(f"Created StepLR scheduler: {scheduler_steplr}")
    assert isinstance(scheduler_steplr, StepLR)

    # Test CosineAnnealingLR (needs steps_per_epoch or T_max in params)
    # Assuming 100 steps per epoch and 50 total epochs for T_max calculation if not provided
    config_cosinelr = {
        'lr_scheduler': 'CosineAnnealingLR',
        'epochs': 50, # Used if T_max not in lr_scheduler_params
        'lr_scheduler_params': {'eta_min': 1e-6}
        }
    scheduler_cosinelr = create_scheduler(optimizer_adamw, config_cosinelr, steps_per_epoch=100)
    print(f"Created CosineAnnealingLR scheduler: {scheduler_cosinelr}")
    assert isinstance(scheduler_cosinelr, CosineAnnealingLR)

    # Test ReduceLROnPlateau
    config_plateau = {
        'lr_scheduler': 'ReduceLROnPlateau',
        'lr_scheduler_params': {'mode': 'min', 'patience': 5, 'factor': 0.2}
        }
    scheduler_plateau = create_scheduler(optimizer_adamw, config_plateau)
    print(f"Created ReduceLROnPlateau scheduler: {scheduler_plateau}")
    assert isinstance(scheduler_plateau, ReduceLROnPlateau)

    # Test no scheduler
    config_none = {'lr_scheduler': 'none'}
    scheduler_none = create_scheduler(optimizer_adamw, config_none)
    print(f"Created None scheduler: {scheduler_none}")
    assert scheduler_none is None

    print("\nOptimizer and scheduler creation tests completed.")

    # Test with no learnable params
    print("\n--- Testing Optimizer Creation with No Learnable Params ---")
    no_params_optimizer = create_optimizer([], config_adamw) # Pass an empty list
    assert no_params_optimizer is None
    print("Test for no learnable params passed.")