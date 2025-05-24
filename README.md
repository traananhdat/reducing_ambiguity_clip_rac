```markdown
# RAC-CLIP: Reducing Ambiguity with CLIP for Counterfeit Agricultural Product Identification

## Overview

This project implements the "RAC-CLIP" method for few-shot learning, specifically applied to the task of identifying counterfeit agricultural products. The core idea is to enhance the robustness and discriminative power of CLIP's features and logits by explicitly modeling and reducing inter-class ambiguity.

This repository provides the code for:
* Data preprocessing.
* Training the RAC-CLIP model.
* Evaluating model performance.
* Core model components including Zero-Shot CLIP, Multi-Feature Aggregation (MFA), Inter-class Ambiguity Reduction (IRA), and Adaptive Logits Fusion (ALF).

## Project Structure

```
reducing_ambiguity_clip_rac/
├── data/                     # Datasets (raw, processed, few-shot splits)
│   ├── processed/
│   └── raw/
├── notebooks/                # Jupyter notebooks for exploration and visualization
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_testing.ipynb
│   └── 03_results_visualization.ipynb
├── results/                  # Output directory for logs, plots, and predictions
│   ├── logs/
│   └── plots/
├── saved_models/             # Saved model checkpoints
├── src/                      # Source code
│   ├── configs/              # YAML configuration files for experiments
│   ├── data_utils/           # Data loading, preprocessing, augmentation
│   ├── evaluation/           # Evaluation scripts and metrics
│   ├── models/               # Model architecture definitions (RACModel, sub-modules)
│   ├── prompts/              # Text prompt generation for CLIP
│   ├── training/             # Training scripts, optimizers, losses
│   └── utils/                # General utilities (logging, general helpers, visualization)
├── main.py                   # Main script to run training, evaluation, preprocessing
├── requirements.txt          # Python package dependencies
└── README.md                 # This file
```

## Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd reducing_ambiguity_clip_rac
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Data Preparation

* Place your raw image data into the `data/raw/` directory. The expected structure for a product (e.g., "product\_A") is:
    ```
    data/
    └── raw/
        ├── authentic_product_A/
        │   ├── image001.jpg
        │   └── ...
        └── counterfeit_product_A/
            ├── image001.jpg
            └── ...
    ```
* Run the preprocessing script to convert raw images into `.pt` files for efficient loading. This step typically involves applying CLIP's preprocessing.
    ```bash
    python main.py preprocess --product_name product_A --split train
    python main.py preprocess --product_name product_A --split val
    python main.py preprocess --product_name product_A --split test
    ```
    This will create files like `data/processed/product_A_train.pt`.

## Usage

All operations are run through `main.py`.

### Configuration

Experiments are defined by YAML configuration files located in `src/configs/`.
* `base_config.yaml`: Contains common settings.
* `rac_resnet50_config.yaml`, `rac_vit_b_16_config.yaml`: Experiment-specific configurations that override or add to the base config. These define model architecture (e.g., CLIP backbone), dataset details, training hyperparameters, etc.

**Key fields to set in your experiment config (e.g., `src/configs/my_experiment_config.yaml`):**
* `experiment_name`: A unique name for your experiment. Outputs will be saved under this name.
* `dataset_name`: Corresponds to the prefix of your processed `.pt` files (e.g., `product_A` if your files are `product_A_train.pt`).
* `num_classes`: Number of classes for your specific task (e.g., 2 for authentic/counterfeit).
* `class_names`: A list of string names for your classes in the order they correspond to integer labels (e.g., `["authentic_product_A", "counterfeit_product_A"]`).
* `clip_model_name`: The CLIP backbone to use (e.g., "RN50", "ViT-B/32").
* `rac_modules`: Configuration for the MFA, IRA, and ALF modules (enable/disable, specific parameters).
* Training hyperparameters like `epochs`, `batch_size`, `learning_rate`.

### Training

To start a training run:

```bash
python main.py train --config src/configs/your_experiment_config.yaml
```

* Logs will be saved to `results/logs/`.
* Model checkpoints will be saved to `saved_models/<experiment_name>/`. The best model (based on validation performance) will be saved as `best_model.pth.tar`.

To resume training from a checkpoint, ensure the `resume_checkpoint` field in your configuration YAML points to the desired `.pth.tar` file, or modify `train.py` to accept a command-line argument for resuming.

### Evaluation

To evaluate a trained model on a test set:

```bash
python main.py evaluate --config src/configs/your_experiment_config.yaml --checkpoint saved_models/<experiment_name>/best_model.pth.tar
```

* You can specify a different test set using the `--test_dataset_name` argument if your processed test file has a different prefix (e.g., `product_A_ood_test.pt`).
* Evaluation results, including metrics and plots like the confusion matrix, will be saved to `results/<experiment_name>/`.

## Model Architecture (RAC-CLIP)

The RAC-CLIP model enhances standard CLIP by incorporating several modules:

1.  **Zero-Shot CLIP Module**: Computes initial image features ($z_i$) and zero-shot logits ($s_i^{ZS}$) using the pre-trained CLIP model and text prompts for each class.
2.  **Multi-Feature Aggregation (MFA) Module**: (Conceptually similar to MAF in RAC)
    * Extracts features from multiple levels of the CLIP image encoder.
    * Applies `Adapter` layers to these features.
    * Fuses the adapted features (Weighted Fusion or Learnable Fusion).
    * A `Projector` (frozen) generates an enhanced feature representation ($z_i^e$).
    * An `MLP` then produces logits ($s_i^{MFA}$) from $z_i^e$.
3.  **Inter-class Ambiguity Reduction (IRA) Module**: (Conceptually similar to ICD in RAC)
    * Takes $s_i^{ZS}$ and $z_i^e$ as input.
    * Uses a series of `Adapter` layers to explicitly model the inter-class ambiguity/confusion pattern ($\Delta s$).
    * Subtracts this learned pattern from the original $s_i^{ZS}$ to produce deconfused logits ($s_i^{IRA}$).
4.  **Adaptive Logits Fusion (ALF) Module**:
    * An `AlphaGenerator` (taking $z_i^e$ as input) produces an adaptive weight $\alpha$.
    * Fuses $s_i^{MFA}$ and $s_i^{IRA}$ using this weight: $s_i^{ALF} = \alpha \cdot s_i^{MFA} + (1-\alpha) \cdot s_i^{IRA}$.
    * $s_i^{ALF}$ are the final logits used for classification.

### Training Objective

The model is trained by minimizing a combined loss function:
$\mathcal{L} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{Sim}$
* $\mathcal{L}_{CE}$: Sum of cross-entropy losses for $s_i^{MFA}$, $s_i^{IRA}$, and $s_i^{ALF}$.
* $\mathcal{L}_{Sim}$: Sum of $L_1$ similarity losses between the softmax probabilities of $s_i^{MFA}$ (and $s_i^{IRA}$) and the detached $s_i^{ZS}$. This acts as a regularization term to prevent over-deconfusion and retain knowledge from the original CLIP model.
* $\lambda$: A trade-off parameter.

Only the parameters of the MFA, IRA, ALF modules, and their constituent Adapters/MLPs are trained. The pre-trained CLIP backbone remains frozen.

## Customization

* **Prompts**: Modify `src/prompts/agricultural_prompts.py` to define different text prompt strategies for your classes. You'll also need to implement `src/models/utils.py:generate_text_features_from_prompt_dict` to process these prompts into text features for CLIP.
* **Backbones**: Experiment with different CLIP backbones (e.g., "RN101", "ViT-L/14") by changing `clip_model_name` in the config. Ensure `feature_dim` and multi-level feature extraction logic are compatible.
* **Module Configuration**: Adjust `adapter_bottleneck_ratio`, `fusion_type` (for MFA), `num_levels`, etc., in the experiment config files.
* **Data Augmentation**: Modify `src/data_utils/augmentations.py` to change data augmentation strategies.

## Future Work / To-Do

* Implement more sophisticated multi-level feature extraction for ViT backbones.
* Add support for few-shot episodic training and evaluation.
* Integrate with experiment tracking tools (e.g., Weights & Biases, TensorBoard).
* More comprehensive error analysis in notebooks.

