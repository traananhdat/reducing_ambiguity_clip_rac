# src/utils/visualization_utils.py

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd # Optional, if you log metrics to CSV and want to plot from there

def plot_training_history(history_dict, plot_keys, title="Training History",
                          xlabel="Epoch", ylabel_primary="Loss", ylabel_secondary="Accuracy",
                          save_path=None, logger=None):
    """
    Plots training and validation loss and accuracy curves from a history dictionary.

    Args:
        history_dict (dict): A dictionary where keys are metric names (e.g., 'train_loss_total',
                             'val_loss_total', 'train_accuracy', 'val_accuracy') and
                             values are lists of metric values per epoch.
        plot_keys (dict): A dictionary specifying which keys from history_dict to plot.
                          Example: {
                              'loss': ['train_loss_total', 'val_loss_total'],
                              'accuracy': ['train_accuracy', 'val_accuracy']
                          }
        title (str): Title for the overall plot.
        xlabel (str): Label for the x-axis.
        ylabel_primary (str): Label for the primary y-axis (typically loss).
        ylabel_secondary (str): Label for the secondary y-axis (typically accuracy).
        save_path (str, optional): If provided, saves the plot to this file path.
        logger (logging.Logger, optional): Logger instance for messages.
    """
    epochs = range(1, len(history_dict.get(plot_keys['loss'][0], [])) + 1)
    if not epochs:
        if logger:
            logger.warning("No data found in history_dict to plot.")
        else:
            print("Warning: No data found in history_dict to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle(title, fontsize=16)

    # Plot losses on the primary y-axis (ax1)
    color_loss = 'tab:red'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel_primary, color=color_loss)
    for key in plot_keys.get('loss', []):
        if key in history_dict:
            ax1.plot(epochs, history_dict[key], color=color_loss if 'val' not in key else 'lightcoral',
                     linestyle='-' if 'train' in key else '--', label=key)
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.7)


    # Create a secondary y-axis for accuracies if accuracy keys are present
    if 'accuracy' in plot_keys and any(key in history_dict for key in plot_keys['accuracy']):
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color_acc = 'tab:blue'
        ax2.set_ylabel(ylabel_secondary, color=color_acc)
        for key in plot_keys.get('accuracy', []):
            if key in history_dict:
                ax2.plot(epochs, history_dict[key], color=color_acc if 'val' not in key else 'skyblue',
                         linestyle='-' if 'train' in key else '--', label=key)
        ax2.tick_params(axis='y', labelcolor=color_acc)
        ax2.legend(loc='upper right')

    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            if logger:
                logger.info(f"Training history plot saved to {save_path}")
            else:
                print(f"Training history plot saved to {save_path}")
        except Exception as e:
            if logger:
                logger.error(f"Failed to save training history plot: {e}")
            else:
                print(f"Failed to save training history plot: {e}")
    else:
        plt.show()
    plt.close()


def plot_sample_images_with_predictions(images, true_labels, pred_labels, class_names,
                                        max_samples=16,
                                        title="Sample Image Predictions",
                                        save_path=None,
                                        logger=None):
    """
    Plots a grid of sample images with their true and predicted labels.

    Args:
        images (torch.Tensor or list of np.ndarray/PIL.Image): Batch of images.
                                                               If tensors, expects (N, C, H, W) and normalized.
        true_labels (list or np.ndarray): True integer labels.
        pred_labels (list or np.ndarray): Predicted integer labels.
        class_names (list of str): List of class names corresponding to integer labels.
        max_samples (int): Maximum number of samples to plot.
        title (str): Title for the plot.
        save_path (str, optional): If provided, saves the plot to this file path.
        logger (logging.Logger, optional): Logger instance for messages.
    """
    if len(images) == 0:
        if logger: logger.warning("No images to plot for sample predictions.")
        return

    num_samples_to_plot = min(len(images), max_samples)
    cols = 4
    rows = (num_samples_to_plot + cols - 1) // cols  # Calculate rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle(title, fontsize=16)
    axes = axes.flatten() # Flatten to easily iterate

    for i in range(num_samples_to_plot):
        img = images[i]
        true_label_idx = true_labels[i]
        pred_label_idx = pred_labels[i]

        # If image is a PyTorch tensor (C, H, W), convert to H, W, C for plotting
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy().transpose((1, 2, 0))
            # Unnormalize if necessary (assuming standard ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        true_name = class_names[true_label_idx] if true_label_idx < len(class_names) else f"Label {true_label_idx}"
        pred_name = class_names[pred_label_idx] if pred_label_idx < len(class_names) else f"Label {pred_label_idx}"

        fontcolor = 'green' if true_label_idx == pred_label_idx else 'red'
        axes[i].set_title(f"True: {true_name}\nPred: {pred_name}", color=fontcolor, fontsize=10)
        axes[i].axis('off')

    # Hide any unused subplots
    for j in range(num_samples_to_plot, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            if logger:
                logger.info(f"Sample predictions plot saved to {save_path}")
            else:
                print(f"Sample predictions plot saved to {save_path}")
        except Exception as e:
            if logger:
                logger.error(f"Failed to save sample predictions plot: {e}")
            else:
                print(f"Failed to save sample predictions plot: {e}")
    else:
        plt.show()
    plt.close()


# --- Example Usage (can be run directly for testing) ---
if __name__ == '__main__':
    import logging
    test_logger = logging.getLogger("VisUtilTest")
    test_logger.setLevel(logging.INFO)
    test_logger.addHandler(logging.StreamHandler())
    test_plot_dir = "./temp_test_plots/"

    # Test plot_training_history
    print("--- Testing Training History Plot ---")
    dummy_history = {
        'train_loss_total': [0.9, 0.7, 0.5, 0.4, 0.35],
        'val_loss_total': [1.0, 0.8, 0.65, 0.6, 0.58],
        'train_accuracy': [60.0, 70.0, 80.0, 85.0, 87.0],
        'val_accuracy': [55.0, 65.0, 75.0, 78.0, 79.0]
    }
    plot_keys_config = {
        'loss': ['train_loss_total', 'val_loss_total'],
        'accuracy': ['train_accuracy', 'val_accuracy']
    }
    history_plot_path = os.path.join(test_plot_dir, "training_history.png")
    plot_training_history(dummy_history, plot_keys_config, save_path=history_plot_path, logger=test_logger)


    # Test plot_sample_images_with_predictions
    print("\n--- Testing Sample Predictions Plot ---")
    try:
        # Create dummy image data (normalized tensors)
        num_dummy_samples = 8
        dummy_images_tensor = torch.rand(num_dummy_samples, 3, 64, 64) # N, C, H, W
        # Simulate some unnormalization for display (usually done in the plot function)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        dummy_images_tensor = dummy_images_tensor * std + mean # This makes them look more like images
        dummy_images_tensor = torch.clamp(dummy_images_tensor, 0, 1)

        dummy_true_labels = np.random.randint(0, 2, num_dummy_samples)
        dummy_pred_labels = np.random.randint(0, 2, num_dummy_samples)
        # Correct some predictions to show green/red
        for k in range(num_dummy_samples // 2):
            dummy_pred_labels[k] = dummy_true_labels[k]

        dummy_class_names = ["Authentic", "Counterfeit"]
        samples_plot_path = os.path.join(test_plot_dir, "sample_predictions.png")
        plot_sample_images_with_predictions(
            dummy_images_tensor, dummy_true_labels, dummy_pred_labels,
            dummy_class_names, max_samples=8, save_path=samples_plot_path, logger=test_logger
        )
    except Exception as e:
        test_logger.error(f"Error in sample predictions plot test: {e}")
        import traceback
        traceback.print_exc()


    # Clean up temp plot directory
    if os.path.exists(test_plot_dir):
        for f in os.listdir(test_plot_dir):
            os.remove(os.path.join(test_plot_dir, f))
        os.rmdir(test_plot_dir)
        print(f"\nCleaned up {test_plot_dir} directory.")

    print("\nVisualization utilities test completed.")