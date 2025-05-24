# src/utils/logging_utils.py

import logging
import os
import sys
import datetime

def setup_logger(name="RAC_Project", log_dir="./results/logs/", level=logging.INFO, console_level=None, file_level=None):
    """
    Sets up a logger that outputs to both console and a file.

    Args:
        name (str): Name of the logger.
        log_dir (str): Directory to save log files.
        level (int): Overall minimum logging level for the logger object.
        console_level (int, optional): Logging level for console output. Defaults to `level`.
        file_level (int, optional): Logging level for file output. Defaults to `level`.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if console_level is None:
        console_level = level
    if file_level is None:
        file_level = level

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = [] # Remove any existing handlers

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # File handler
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name.replace(' ', '_')}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # Simpler for console
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.propagate = False # Prevent logging from propagating to the root logger

    logger.info(f"Logger '{name}' initialized. Logging to console and file: {log_file}")
    return logger

def log_metrics(metrics_dict, epoch, stage="Train", logger=None):
    """
    Logs a dictionary of metrics in a structured way.

    Args:
        metrics_dict (dict): Dictionary of metrics (e.g., {'loss': 0.5, 'accuracy': 0.8}).
        epoch (int): Current epoch number.
        stage (str): Stage of the process (e.g., "Train", "Validation", "Test").
        logger (logging.Logger, optional): Logger instance. If None, prints to stdout.
    """
    log_message = f"Epoch {epoch} - {stage} Metrics: "
    metrics_str_list = []
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            metrics_str_list.append(f"{key}: {value:.4f}")
        else:
            metrics_str_list.append(f"{key}: {value}")
    log_message += ", ".join(metrics_str_list)

    if logger:
        logger.info(log_message)
    else:
        print(log_message)

# --- Example Usage (can be run directly for testing) ---
if __name__ == '__main__':
    # Test setup_logger
    print("--- Testing Logger Setup ---")
    test_logger_dir = "./temp_test_logs/"
    try:
        test_logger = setup_logger(name="TestApp", log_dir=test_logger_dir, level=logging.DEBUG)
        test_logger.debug("This is a debug message for TestApp.")
        test_logger.info("This is an info message for TestApp.")
        test_logger.warning("This is a warning message for TestApp.")
        test_logger.error("This is an error message for TestApp.")
        print(f"Log file should be created in: {test_logger_dir}")

        # Check if log file was created
        log_files = [f for f in os.listdir(test_logger_dir) if f.startswith("TestApp") and f.endswith(".log")]
        if not log_files:
            print("Error: Log file not found in temp_test_logs.")
        else:
            print(f"Found log file: {log_files[0]}")

    except Exception as e:
        print(f"Error during logger setup test: {e}")
    finally:
        # Clean up temp log directory
        if os.path.exists(test_logger_dir):
            for f in os.listdir(test_logger_dir):
                os.remove(os.path.join(test_logger_dir, f))
            os.rmdir(test_logger_dir)
        print("Cleaned up temp_test_logs directory.")


    # Test log_metrics
    print("\n--- Testing Metric Logging ---")
    dummy_metrics = {
        'loss_total': 0.789123,
        'accuracy': 85.12345,
        'lr': 0.0001,
        'some_other_metric': 'value_abc'
    }
    print("\nLogging with default print:")
    log_metrics(metrics_dict=dummy_metrics, epoch=5, stage="Validation")

    print("\nLogging with test_logger (if initialized):")
    if 'test_logger' in locals() and test_logger is not None:
        log_metrics(metrics_dict=dummy_metrics, epoch=5, stage="Validation", logger=test_logger)
    else:
        print("test_logger was not initialized, skipping this part of the test.")

    print("\nLogging utilities test completed.")