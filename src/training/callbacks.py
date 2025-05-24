# src/training/callbacks.py

import os
import torch
import numpy as np
import shutil # For saving checkpoints

class Callback:
    """Base class for all callbacks."""
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.logger = None

    def set_trainer(self, trainer_instance):
        # In a more integrated system, a "Trainer" class might manage these.
        # For now, we can pass them individually or as part of a context.
        pass

    def set_logger(self, logger):
        self.logger = logger

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass


class ModelCheckpoint(Callback):
    """
    Saves the model after every epoch or only the best model.
    """
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            raise ValueError(f"Mode '{mode}' is unknown. Choose 'min' or 'max'.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_metric = logs.get(self.monitor)

        if current_metric is None:
            if self.logger:
                self.logger.warning(f"ModelCheckpoint: Metric '{self.monitor}' not found in logs. Skipping.")
            return

        if self.save_best_only:
            if self.monitor_op(current_metric, self.best):
                if self.verbose > 0 and self.logger:
                    self.logger.info(f"Epoch {epoch+1}: {self.monitor} improved from {self.best:.4f} to {current_metric:.4f}. Saving model to {self.filepath}")
                self.best = current_metric
                self._save_model(epoch, logs)
            elif self.verbose > 0 and self.logger:
                 self.logger.debug(f"Epoch {epoch+1}: {self.monitor} ({current_metric:.4f}) did not improve from {self.best:.4f}.")

        else: # Save every epoch
            filepath = self.filepath.format(epoch=epoch+1, **logs)
            if self.verbose > 0 and self.logger:
                self.logger.info(f"Epoch {epoch+1}: saving model to {filepath}")
            self._save_model(epoch, logs, filepath_override=filepath)

    def _save_model(self, epoch, logs, filepath_override=None):
        filepath_to_save = filepath_override if filepath_override else self.filepath
        try:
            save_dict = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(), # Requires self.model to be set
                'optimizer': self.optimizer.state_dict() if self.optimizer else None,
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                'best_val_metric': self.best, # Storing the best value of the monitored metric
                # 'config': self.config # If you pass config to the callback
            }
            if self.save_weights_only:
                torch.save(self.model.state_dict(), filepath_to_save)
            else:
                torch.save(save_dict, filepath_to_save)

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving model: {e}")


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving.
    """
    def __init__(self, monitor='val_loss', min_delta=0, patience=10, mode='min', verbose=1, restore_best_weights=False):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

        if self.mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif self.mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            raise ValueError(f"Mode '{mode}' is unknown. Choose 'min' or 'max'.")

        if self.min_delta < 0:
             self.min_delta *= -1 # Ensure min_delta is positive

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.mode == 'min' else -np.Inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_metric = logs.get(self.monitor)

        if current_metric is None:
            if self.logger:
                self.logger.warning(f"EarlyStopping: Metric '{self.monitor}' not found in logs. Skipping.")
            return False # Indicate training should not stop

        # Check if the current metric is an improvement
        if self.mode == 'min':
            improvement = self.best - current_metric
        else: # mode == 'max'
            improvement = current_metric - self.best

        if improvement > self.min_delta: # Significant improvement
            self.best = current_metric
            self.wait = 0
            if self.restore_best_weights and self.model:
                self.best_weights = {k: v.cpu() for k, v in self.model.state_dict().items()}
        else: # No significant improvement
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose > 0 and self.logger:
                    self.logger.info(f"Epoch {epoch+1}: Early stopping triggered. Monitored metric '{self.monitor}' did not improve significantly for {self.patience} epochs.")
                if self.restore_best_weights and self.best_weights and self.model:
                    if self.logger: self.logger.info(f"Restoring model weights from the end of the best epoch: {epoch - self.wait + 1}.")
                    self.model.load_state_dict(self.best_weights)
                return True # Indicate training should stop
        return False # Indicate training should continue

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0 and self.logger:
            self.logger.info(f"Training stopped early at epoch {self.stopped_epoch + 1}.")


class LearningRateLogger(Callback):
    """Logs the current learning rate at the end of each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        if self.optimizer and self.logger:
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = param_group['lr']
                self.logger.info(f"Epoch {epoch+1}: Learning rate for group {i} is {lr:.2e}")


# --- Example of how to integrate callbacks into a simplified training loop ---
# This is illustrative; your `train.py` would manage this.
if __name__ == '__main__':
    import logging
    logger = logging.getLogger("CallbackTest")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    # Dummy model, optimizer
    model = torch.nn.Linear(10,1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Initialize Callbacks
    checkpoint_cb = ModelCheckpoint(filepath='./temp_checkpoints/best_model_cb.pth', monitor='val_accuracy', mode='max', verbose=1)
    early_stop_cb = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1, restore_best_weights=True)
    lr_logger_cb = LearningRateLogger()

    callbacks = [checkpoint_cb, early_stop_cb, lr_logger_cb]

    # Set model, optimizer, logger for callbacks
    for cb in callbacks:
        cb.model = model
        cb.optimizer = optimizer
        # cb.scheduler = scheduler # if you have one
        cb.set_logger(logger)


    logger.info("--- Simulating Training with Callbacks ---")
    num_epochs = 10
    simulated_val_accuracies = [0.5, 0.6, 0.7, 0.68, 0.69, 0.71, 0.70, 0.69, 0.68, 0.67] # Example accuracies

    for cb in callbacks: cb.on_train_begin()

    for epoch in range(num_epochs):
        logger.info(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        for cb in callbacks: cb.on_epoch_begin(epoch)

        # --- Simulate training step ---
        # model.train() ... optimizer.step() ...
        time.sleep(0.1) # Simulate work
        train_logs = {'loss': 1.0 - simulated_val_accuracies[epoch] * 0.5, 'accuracy': simulated_val_accuracies[epoch] - 0.05} # Dummy train logs

        # --- Simulate validation step ---
        # model.eval() ...
        val_logs = {'val_loss': 1.0 - simulated_val_accuracies[epoch], 'val_accuracy': simulated_val_accuracies[epoch]}
        logger.info(f"Simulated Val Accuracy: {val_logs['val_accuracy']:.4f}")

        stop_training = False
        for cb in callbacks:
            # Pass combined logs for callbacks to access
            epoch_logs = {**train_logs, **val_logs}
            if cb.on_epoch_end(epoch, logs=epoch_logs): # EarlyStopping returns True if it wants to stop
                stop_training = True
        
        if stop_training:
            break

    for cb in callbacks: cb.on_train_end()

    # Cleanup dummy checkpoint directory
    if os.path.exists('./temp_checkpoints'):
        shutil.rmtree('./temp_checkpoints')
    logger.info("Callback simulation finished.")