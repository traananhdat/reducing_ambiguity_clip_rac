# src/evaluation/metrics.py

import numpy as np
import sklearn.metrics as sk_metrics
import matplotlib.pyplot as plt
import seaborn as sns
import itertools # For iterating over confusion matrix

def calculate_accuracy(y_pred_np, y_true_np):
    """
    Calculates the accuracy.

    Args:
        y_pred_np (np.ndarray): Predicted labels (integers).
        y_true_np (np.ndarray): True labels (integers).

    Returns:
        float: Accuracy score.
    """
    if y_pred_np.shape != y_true_np.shape:
        raise ValueError(f"Shape mismatch: y_pred {y_pred_np.shape} vs y_true {y_true_np.shape}")
    return np.mean(y_pred_np == y_true_np) * 100.0


def calculate_precision_recall_f1(y_true_np, y_pred_np, average='weighted', zero_division=0):
    """
    Calculates precision, recall, and F1-score.

    Args:
        y_true_np (np.ndarray): True labels.
        y_pred_np (np.ndarray): Predicted labels.
        average (str, optional): Type of averaging to perform on the data.
                                 Options: 'micro', 'macro', 'samples', 'weighted', None.
                                 If None, the scores for each class are returned.
        zero_division (int or str, optional): Value to return when there is a zero division.
                                              See sklearn.metrics.precision_recall_fscore_support.

    Returns:
        tuple: (precision, recall, f1_score, support)
               If average is None, these are arrays per class.
               Otherwise, they are scalar values.
    """
    precision, recall, f1, support = sk_metrics.precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        average=average,
        zero_division=zero_division,
        labels=np.unique(np.concatenate((y_true_np, y_pred_np))) # Ensure all present labels are considered
    )
    return precision, recall, f1, support


def generate_confusion_matrix(y_true_np, y_pred_np, class_names=None):
    """
    Generates a confusion matrix.

    Args:
        y_true_np (np.ndarray): True labels.
        y_pred_np (np.ndarray): Predicted labels.
        class_names (list of str, optional): List of class names for display.
                                             If None, integer labels will be used.

    Returns:
        np.ndarray: The confusion matrix.
    """
    labels = np.unique(np.concatenate((y_true_np, y_pred_np))) # Get all unique labels present
    cm = sk_metrics.confusion_matrix(y_true_np, y_pred_np, labels=labels)

    if class_names is not None:
        if len(class_names) != len(labels):
            print(f"Warning: Number of class_names ({len(class_names)}) does not match "
                  f"number of unique labels found ({len(labels)}). Displaying with integer labels.")
    return cm, labels # Return labels used for CM construction for consistent plotting


def plot_confusion_matrix(cm, classes, labels_for_cm_ticks,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          filename=None,
                          logger=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args:
        cm (np.ndarray): Confusion matrix.
        classes (list of str): List of class names for tick labels.
        labels_for_cm_ticks (np.ndarray): The actual label values corresponding to CM rows/cols.
                                          Used to map `classes` correctly if `classes` is a subset or reordered.
        normalize (bool, optional): Whether to normalize the CM. Defaults to False.
        title (str, optional): Title for the plot.
        cmap (matplotlib.colors.Colormap, optional): Colormap for the plot.
        filename (str, optional): If provided, saves the plot to this file.
        logger (logging.Logger, optional): Logger instance for messages.
    """
    if normalize:
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.zeros_like(cm, dtype=float)
        # Handle cases where a row sum is 0 to avoid division by zero
        for i in range(cm.shape[0]):
            if cm_sum[i, 0] > 0:
                cm_normalized[i, :] = cm[i, :] / cm_sum[i, 0]
        cm_to_plot = cm_normalized
        if logger: logger.info("Normalized confusion matrix shown.")
    else:
        cm_to_plot = cm
        if logger: logger.info('Confusion matrix, without normalization shown.')

    if logger: logger.info(f"\n{cm_to_plot}")


    plt.figure(figsize=(max(8, len(classes) * 0.8), max(6, len(classes) * 0.6)))
    plt.imshow(cm_to_plot, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if classes is not None and labels_for_cm_ticks is not None:
        # Ensure ticks match the order and values of labels_for_cm_ticks
        tick_marks = np.arange(len(labels_for_cm_ticks))
        # Create tick labels based on the order of labels_for_cm_ticks
        tick_labels = [classes[i] if i < len(classes) else str(labels_for_cm_ticks[i]) for i in range(len(labels_for_cm_ticks))]
        plt.xticks(tick_marks, tick_labels, rotation=45, ha="right")
        plt.yticks(tick_marks, tick_labels)
    else:
        tick_marks = np.arange(cm.shape[0])
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)


    fmt = '.2f' if normalize else 'd'
    thresh = cm_to_plot.max() / 2.
    for i, j in itertools.product(range(cm_to_plot.shape[0]), range(cm_to_plot.shape[1])):
        plt.text(j, i, format(cm_to_plot[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_to_plot[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if filename:
        try:
            plt.savefig(filename, dpi=300)
            if logger: logger.info(f"Confusion matrix saved to {filename}")
        except Exception as e:
            if logger: logger.error(f"Failed to save confusion matrix: {e}")
    else:
        plt.show()
    plt.close()


# --- Example Usage (can be run directly for testing) ---
if __name__ == '__main__':
    print("--- Testing Metrics Functions ---")

    # Dummy data
    y_true = np.array([0, 1, 0, 1, 2, 0, 1, 2, 2, 0])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 0, 2, 2, 1])
    class_names_test = ['Class A (0)', 'Class B (1)', 'Class C (2)'] # Display names
    actual_class_labels_test = [0, 1, 2] # The actual integer labels corresponding to class_names

    # Test accuracy
    acc = calculate_accuracy(y_pred, y_true)
    print(f"Accuracy: {acc:.2f}%")
    assert acc == 70.0

    # Test precision, recall, F1 (weighted)
    precision_w, recall_w, f1_w, support_w = calculate_precision_recall_f1(y_true, y_pred, average='weighted')
    print(f"Weighted Precision: {precision_w:.4f}")
    print(f"Weighted Recall: {recall_w:.4f}")
    print(f"Weighted F1-Score: {f1_w:.4f}")
    # print(f"Weighted Support: {support_w}") # support_w is None for weighted if labels are passed

    # Test precision, recall, F1 (per class)
    print("\nPer-class metrics:")
    precision_pc, recall_pc, f1_pc, support_pc = calculate_precision_recall_f1(y_true, y_pred, average=None)
    for i, class_name in enumerate(class_names_test):
        print(f"  Class: {class_name}")
        print(f"    Precision: {precision_pc[i]:.4f}")
        print(f"    Recall: {recall_pc[i]:.4f}")
        print(f"    F1-Score: {f1_pc[i]:.4f}")
        print(f"    Support: {support_pc[i]}")

    # Test confusion matrix generation and plotting
    print("\nGenerating and plotting confusion matrix...")
    cm, cm_labels = generate_confusion_matrix(y_true, y_pred, class_names=class_names_test)
    print("Confusion Matrix (numeric labels used for CM construction):")
    print(f"Labels for CM ticks: {cm_labels}") # These are the unique sorted labels from data
    print(cm)

    # Map class_names_test to the order of cm_labels for plotting
    # This is a bit more robust if class_names_test isn't perfectly [0, 1, 2, ...]
    # or if some classes aren't present in y_true/y_pred.
    # However, for this example, cm_labels will be [0, 1, 2]
    
    # For plotting, we need class names corresponding to `cm_labels`
    # If class_names_test = ['A', 'B', 'C'] and cm_labels = [0,1,2] then it's direct.
    # If cm_labels could be sparse, e.g. [0, 2] because class 1 was not in y_true/y_pred,
    # then class_names mapping needs care.
    # Our current generate_confusion_matrix uses labels=np.unique(...) ensuring all present classes are covered.
    plot_confusion_matrix(cm, classes=class_names_test, labels_for_cm_ticks=cm_labels,
                          title='Test Confusion Matrix', filename='test_cm.png')
    plot_confusion_matrix(cm, classes=class_names_test, labels_for_cm_ticks=cm_labels, normalize=True,
                          title='Test Normalized Confusion Matrix', filename='test_cm_normalized.png')
    print("Confusion matrix plots saved as test_cm.png and test_cm_normalized.png (if matplotlib is working).")
    # Clean up test files
    if os.path.exists('test_cm.png'): os.remove('test_cm.png')
    if os.path.exists('test_cm_normalized.png'): os.remove('test_cm_normalized.png')

    print("\nMetrics functions test completed.")