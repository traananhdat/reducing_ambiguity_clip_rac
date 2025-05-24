# src/models/rac_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming these modules are defined in their respective files
# and RAC uses a similar modular structure to RAC [cite: 71]
from .zs_clip_module import ZeroShotCLIPModule
from .mfa_module import MFAModule # Conceptually like Multi-level Adapter Fusion (MAF) [cite: 3, 26]
from .ira_module import IRAModule # Conceptually like Inter-Class Deconfusion (ICD) [cite: 3, 26]
from .alf_module import ALFModule # Conceptually like Adaptive Logits Fusion (ALF) [cite: 71]


class RACModel(nn.Module):
    """
    Main RAC (Reducing Ambiguity in CLIP) model.
    This model integrates components inspired by the RAC paper's methodology
    to enhance few-shot learning performance by addressing inter-class ambiguity/confusion.
    It uses:
    1. ZeroShotCLIPModule: To get initial zero-shot logits from CLIP[cite: 77].
    2. MFAModule: To get enhanced visual features and MAF-like logits[cite: 78, 79].
    3. IRAModule: To learn and remove inter-class ambiguity patterns from ZS logits[cite: 80].
    4. ALFModule: To adaptively fuse the logits from MFA-like and IRA-like modules[cite: 81].
    """
    def __init__(self, config, clip_model, num_classes):
        """
        Args:
            config (dict): Configuration dictionary (e.g., loaded from YAML).
            clip_model (nn.Module): The pre-trained CLIP model (or its relevant parts).
                                    This RACModel will not own the CLIP model's weights
                                    if they are meant to be frozen, but will use its encoders.
            num_classes (int): Number of output classes for the classification task.
        """
        super().__init__()
        self.config = config
        self.clip_model = clip_model # For ZS-CLIP and potentially image encoding for MFA
        self.num_classes = num_classes

        # --- Instantiate ZeroShotCLIPModule ---
        # This module will use the clip_model's image and text encoders.
        self.zs_clip_module = ZeroShotCLIPModule(self.clip_model)

        # --- Instantiate MFA (Multi-Feature Aggregation) Module ---
        # MFA needs info about image encoder channels if adapting intermediate layers,
        # and the target feature dimension (CLIP's embedding dim).
        if config['rac_modules']['mfa']['enabled']:
            self.mfa_module = MFAModule(
                image_encoder_channels=config.get('image_encoder_channels', None), # e.g., [256, 512, 1024, 2048] for RN50
                feature_dim=config['feature_dim'], # CLIP's output embedding dimension
                num_levels=config['rac_modules']['mfa'].get('num_levels', 4),
                fusion_type=config['rac_modules']['mfa'].get('fusion_type', 'WF'),
                projector_type=config['rac_modules']['mfa'].get('projector_type', 'attention_pool'),
                adapter_bottleneck_ratio=config['rac_modules']['mfa'].get('adapter_bottleneck_ratio', 0.5),
                num_classes=self.num_classes # For the MLP within MFA that produces s_MFA
            )
        else:
            self.mfa_module = None

        # --- Instantiate IRA (Inter-class Ambiguity Reduction) Module ---
        # IRA needs the number of classes (for logit-based adapters) and feature dimension.
        if config['rac_modules']['ira']['enabled']:
            self.ira_module = IRAModule(
                num_classes=self.num_classes,
                feature_dim=config['feature_dim'],
                adapter_bottleneck_ratio=config['rac_modules']['ira'].get('adapter_bottleneck_ratio', 0.5)
            )
        else:
            self.ira_module = None

        # --- Instantiate ALF (Adaptive Logits Fusion) Module ---
        # ALF needs the feature dimension for its alpha generator.
        if config['rac_modules']['alf']['enabled']:
            self.alf_module = ALFModule(
                feature_dim=config['feature_dim'],
                # num_classes is implicitly handled by input logits
                alpha_generator_bottleneck_ratio=config['rac_modules']['alf'].get('adapter_bottleneck_ratio', 0.5)
            )
        else:
            self.alf_module = None

        self.lambda_tradeoff_similarity = config.get('lambda_tradeoff_similarity', 1.0)

    def forward(self, image_x, multi_level_features_f=None, text_features_T=None, labels_y=None):
        """
        Forward pass of the RACModel.

        Args:
            image_x (torch.Tensor): Input images (batch_size, C, H, W).
            multi_level_features_f (list of torch.Tensor, optional):
                Pre-extracted multi-level features from the image encoder for MFA.
                If None, MFA might need to extract them using self.clip_model.image_encoder.
            text_features_T (torch.Tensor): Pre-computed text features for C classes for ZS-CLIP.
                                          Shape: (num_classes, feature_dim).
            labels_y (torch.Tensor, optional): Ground truth labels for loss calculation.

        Returns:
            tuple:
                - final_logits (torch.Tensor): The final output logits after fusion.
                - total_loss (torch.Tensor, optional): Total computed loss if labels_y is provided.
                - loss_dict (dict, optional): Dictionary of individual loss components.
        """
        outputs = {}

        # 1. Zero-Shot CLIP Logits (s_ZS)
        # ZS-CLIP module computes logits s_i^ZS and image features z_i [cite: 77]
        # The RAC paper shows text features z_T and image features z_i go into ZS-CLIP [cite: 70]
        logits_s_ZS, image_feature_z_i = self.zs_clip_module(image_x, text_features_T)
        outputs['s_ZS'] = logits_s_ZS
        outputs['z_i'] = image_feature_z_i # CLIP's global image feature

        # 2. Multi-Feature Aggregation (MFA) Logits (s_MFA)
        # MFA module uses multi-level features f_i to produce enhanced feature z_i^e and logits s_i^MAF [cite: 78, 79]
        if self.mfa_module:
            # If multi_level_features_f are not provided, they need to be extracted.
            # This might require hooking into the clip_model's image encoder.
            # For simplicity, assume they are provided or MFA handles it internally.
            if multi_level_features_f is None and hasattr(self.clip_model, 'get_multi_level_features'):
                # This is a placeholder for actual multi-level feature extraction
                multi_level_features_f = self.clip_model.get_multi_level_features(image_x)

            enhanced_feature_z_e, logits_s_MFA = self.mfa_module(multi_level_features_f, image_feature_z_i)
            outputs['s_MFA'] = logits_s_MFA
            outputs['z_e'] = enhanced_feature_z_e
        else: # Fallback if MFA is disabled
            logits_s_MFA = logits_s_ZS # Or some other baseline
            enhanced_feature_z_e = image_feature_z_i # Use global CLIP feature as z_e
            outputs['s_MFA'] = logits_s_MFA
            outputs['z_e'] = enhanced_feature_z_e


        # 3. Inter-class Ambiguity Reduction (IRA) Logits (s_IRA)
        # IRA module uses s_i^ZS and z_i^e to produce s_i^ICD [cite: 80]
        if self.ira_module:
            logits_s_IRA = self.ira_module(logits_s_ZS, enhanced_feature_z_e)
            outputs['s_IRA'] = logits_s_IRA
        else: # Fallback if IRA is disabled
            logits_s_IRA = logits_s_ZS
            outputs['s_IRA'] = logits_s_IRA

        # 4. Adaptive Logits Fusion (ALF)
        # ALF module fuses s_i^MAF and s_i^ICD using z_i^e to generate alpha [cite: 81, 99]
        # Note: The RAC paper's Eq 11: s_ALF = alpha * s_MAF + (1-alpha) * s_ICD [cite: 99]
        # Figure 2 shows alpha * s_ICD + (1-alpha) * s_MAF [cite: 70]
        # We'll follow Eq 11.
        if self.alf_module and self.mfa_module and self.ira_module:
            final_logits, alpha_weight = self.alf_module(logits_s_MFA, logits_s_IRA, enhanced_feature_z_e)
            outputs['alpha'] = alpha_weight
        elif self.mfa_module: # If only MFA is active
            final_logits = logits_s_MFA
        elif self.ira_module: # If only IRA is active
            final_logits = logits_s_IRA
        else: # If neither, use ZS logits
            final_logits = logits_s_ZS
        outputs['s_ALF'] = final_logits


        # --- Loss Calculation ---
        total_loss = None
        loss_dict = {}

        if labels_y is not None:
            # Cross-entropy losses [cite: 102]
            if 's_MFA' in outputs:
                loss_dict['ce_mfa'] = F.cross_entropy(outputs['s_MFA'], labels_y)
            if 's_IRA' in outputs:
                loss_dict['ce_ira'] = F.cross_entropy(outputs['s_IRA'], labels_y)
            loss_dict['ce_alf'] = F.cross_entropy(final_logits, labels_y) # Always on final logits

            total_ce_loss = sum(loss_dict[k] for k in loss_dict if k.startswith('ce_'))

            # Similarity losses (L1 between softmax probabilities) [cite: 103, 104]
            # L_sim ensures output logits maintain similarity to original ZS logits [cite: 69]
            loss_sim_mfa = 0.0
            loss_sim_ira = 0.0

            if 's_MFA' in outputs and self.mfa_module: # Only if MFA module is active and producing its own logits
                prob_s_MFA = F.softmax(outputs['s_MFA'], dim=-1)
                prob_s_ZS_detached = F.softmax(logits_s_ZS.detach(), dim=-1)
                loss_sim_mfa = F.l1_loss(prob_s_MFA, prob_s_ZS_detached)
                loss_dict['sim_mfa'] = loss_sim_mfa

            if 's_IRA' in outputs and self.ira_module: # Only if IRA module is active
                prob_s_IRA = F.softmax(outputs['s_IRA'], dim=-1)
                prob_s_ZS_detached = F.softmax(logits_s_ZS.detach(), dim=-1)
                loss_sim_ira = F.l1_loss(prob_s_IRA, prob_s_ZS_detached)
                loss_dict['sim_ira'] = loss_sim_ira

            total_sim_loss = loss_sim_mfa + loss_sim_ira # As in RAC Eq 13 [cite: 105]

            # Total loss [cite: 106]
            total_loss = total_ce_loss + self.lambda_tradeoff_similarity * total_sim_loss
            loss_dict['total_ce'] = total_ce_loss
            loss_dict['total_sim'] = total_sim_loss
            loss_dict['total_loss'] = total_loss

            if 'alpha' in outputs:
                loss_dict['alpha_mean'] = outputs['alpha'].mean()

        return final_logits, total_loss, loss_dict

    def get_learnable_parameters(self):
        """Helper function to get parameters that need to be trained."""
        params = []
        if self.mfa_module:
            params.extend(list(self.mfa_module.parameters()))
        if self.ira_module:
            params.extend(list(self.ira_module.parameters()))
        if self.alf_module:
            params.extend(list(self.alf_module.parameters()))
        return params

# Example of how you might instantiate this in your training script:
# from src.models.clip_backbone import load_clip_model # Assuming you have a helper for this
#
# config = load_yaml_config(...) # Your config loading
# clip_model, clip_preprocess = load_clip_model(config['clip_model_name'], device)
#
# # Freeze CLIP parameters
# for param in clip_model.parameters():
#     param.requires_grad = False
#
# rac_model_instance = RACModel(config, clip_model, num_classes=config['num_classes']).to(device)
#
# # Optimizer would then use:
# optimizer = torch.optim.AdamW(rac_model_instance.get_learnable_parameters(), lr=config['learning_rate'])