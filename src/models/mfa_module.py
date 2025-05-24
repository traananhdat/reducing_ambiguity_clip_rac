# src/models/mfa_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .adapters import Adapter, MLP # Assuming Adapter and MLP are defined here

class MFAModule(nn.Module):
    """
    Multi-Feature Aggregation (MFA) Module.
    Conceptually similar to the Multi-level Adapter Fusion (MAF) module from the RAC paper.
    It extracts features from different levels of an image encoder,
    fuses them using adapters, and then generates an enhanced feature representation (z_e)
    and logits (s_MFA) from this enhanced feature.
    """
    def __init__(self, image_encoder_channels, feature_dim, num_levels=4,
                 fusion_type='WF', projector_type='attention_pool',
                 adapter_bottleneck_ratio=0.5, num_classes=1000):
        """
        Args:
            image_encoder_channels (list of int): List of channel dimensions for features
                                                 from different levels of the image encoder
                                                 (e.g., [256, 512, 1024, 2048] for ResNet-50 blocks).
                                                 This is needed if adapters are applied to raw encoder outputs.
                                                 If adapters are applied to CLIP's multi-level features
                                                 (which might all be projected to 'feature_dim' already),
                                                 this might be a list like [feature_dim, feature_dim, ...].
            feature_dim (int): The target dimension after adapters and for the enhanced feature z_e
                               (typically CLIP's embedding dimension).
            num_levels (int): Number of feature levels to fuse. Should match len(image_encoder_channels).
            fusion_type (str): 'WF' (Weighted Fusion) or 'LF' (Learnable Fusion). [cite: 91]
            projector_type (str): Type of projector to get z_e from fused features.
                                  'attention_pool' for ResNet-like, 'linear' for ViT-like,
                                  or 'identity' if no explicit projection is needed. [cite: 90]
            adapter_bottleneck_ratio (float): Bottleneck ratio for the Adapters.
            num_classes (int): Number of classes for the output logits s_MFA.
        """
        super().__init__()
        self.num_levels = num_levels
        self.fusion_type = fusion_type.upper()
        self.projector_type = projector_type
        self.feature_dim = feature_dim

        if image_encoder_channels is not None and len(image_encoder_channels) != num_levels:
            raise ValueError(f"Length of image_encoder_channels ({len(image_encoder_channels)}) "
                             f"must match num_levels ({num_levels}).")

        # --- Adapters for each feature level ---
        # These transform f_i^k to z_i^k (from RAC paper's notation for MAF)
        self.adapters = nn.ModuleList()
        for k in range(self.num_levels):
            # Input dimension for adapter k
            # If image_encoder_channels is None, assume input features are already at feature_dim
            adapter_input_dim = image_encoder_channels[k] if image_encoder_channels else self.feature_dim
            self.adapters.append(
                Adapter(adapter_input_dim, self.feature_dim, bottleneck_dim_ratio=adapter_bottleneck_ratio)
            )

        # --- Fusion Mechanism ---
        if self.fusion_type == 'WF':
            # Weighted Fusion: preset weights beta_k. [cite: 92]
            # Example weights from RAC paper for 4 levels: 0.1, 0.2, 0.3, 0.4. [cite: 114]
            if self.num_levels == 4:
                self.weights_beta = nn.Parameter(torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32), requires_grad=False)
            else: # Default to equal weights if not 4 levels
                self.weights_beta = nn.Parameter(torch.ones(self.num_levels, dtype=torch.float32) / self.num_levels, requires_grad=False)
        elif self.fusion_type == 'LF':
            # Learnable Fusion: concatenate then use an Adapter. [cite: 93]
            # Input to fusion_adapter is concatenated adapted features (num_levels * feature_dim)
            self.fusion_adapter = Adapter(self.feature_dim * self.num_levels, self.feature_dim,
                                          bottleneck_dim_ratio=adapter_bottleneck_ratio)
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}. Choose 'WF' or 'LF'.")

        # --- Projector ---
        # This transforms the fused feature (z_hat_e) into the enhanced feature (z_e). [cite: 89]
        # The RAC paper states the Projector is frozen. [cite: 89]
        if self.projector_type == 'attention_pool' and self.feature_dim == 2048: # Typical for ResNet final block output
             # This is a simplification. CLIP's ResNet uses a specific AttentionPool2d.
             # If input features are already pooled, this might be an Identity or a Linear layer.
             # For MAF, this projector takes the FUSED feature of `self.feature_dim`.
             # If `self.feature_dim` is already the final CLIP embedding dim (e.g., 1024 for RN50),
             # an additional attention pool might not be what's intended unless it's a custom one.
             # Often, `z_hat_e` (fused feature) is already at the target `feature_dim`.
             # Let's assume for now it's a linear projection if specified, or identity.
            self.projector = nn.Linear(self.feature_dim, self.feature_dim) # Placeholder, might be more complex
        elif self.projector_type == 'linear':
            self.projector = nn.Linear(self.feature_dim, self.feature_dim)
        elif self.projector_type == 'identity':
            self.projector = nn.Identity()
        else: # Default to identity if not specified or understood
            print(f"Warning: Projector type '{self.projector_type}' not explicitly handled, using Identity.")
            self.projector = nn.Identity()

        # Freeze projector parameters as per RAC paper [cite: 89]
        for param in self.projector.parameters():
            param.requires_grad = False

        # --- MLP for s_MFA logits ---
        # This MLP takes the enhanced feature z_e and outputs logits s_MFA. [cite: 79]
        self.mlp_for_logits = MLP(self.feature_dim, num_classes)


    def forward(self, multi_level_features_f, global_clip_feature_z_i=None):
        """
        Forward pass of the MFA module.

        Args:
            multi_level_features_f (list of torch.Tensor):
                List of features [f_i^1, f_i^2, ..., f_i^L] from different levels
                of the image encoder. Each tensor f_i^k should have a shape
                compatible with the k-th adapter (e.g., (batch, channels_k, H_k, W_k)
                or (batch, num_tokens_k, channels_k) for ViT, or already pooled features).
                Adapters might expect (batch, feature_dim_k).
            global_clip_feature_z_i (torch.Tensor, optional):
                The global image feature from CLIP's image encoder.
                This might be used if one of the "levels" in multi_level_features_f
                is intended to be this global feature, or if MFA's design is extended.
                The RAC paper's MAF primarily focuses on intermediate layer features.

        Returns:
            tuple:
                - enhanced_feature_z_e (torch.Tensor): The enhanced image feature.
                                                     Shape: (batch_size, feature_dim).
                - logits_s_MFA (torch.Tensor): Logits generated from z_e.
                                              Shape: (batch_size, num_classes).
        """
        if len(multi_level_features_f) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} feature levels, but got {len(multi_level_features_f)}.")

        adapted_features_z_k_list = []
        for k in range(self.num_levels):
            f_k = multi_level_features_f[k]
            # Adapters might expect 2D input (batch, features).
            # If f_k is (batch, C, H, W), apply global average pooling or flatten.
            if f_k.ndim == 4: # Assuming (B, C, H, W)
                # Global Average Pooling before adapter, common for conv features
                f_k_pooled = F.adaptive_avg_pool2d(f_k, (1, 1)).squeeze(-1).squeeze(-1) # (B, C)
            elif f_k.ndim == 3: # Assuming (B, NumTokens, C) for ViT features
                # Take the CLS token feature if available, or average token features
                f_k_pooled = f_k[:, 0, :] # Assuming CLS token is at index 0
                # Or: f_k_pooled = f_k.mean(dim=1)
            elif f_k.ndim == 2: # Assuming (B, C) - already pooled/flattened
                f_k_pooled = f_k
            else:
                raise ValueError(f"Unsupported feature dimension {f_k.ndim} for level {k}.")

            adapted_features_z_k_list.append(self.adapters[k](f_k_pooled)) # z_i^k in RAC paper

        # --- Fuse adapted features ---
        if self.fusion_type == 'WF':
            # Stack features: (num_levels, batch_size, feature_dim)
            stacked_features = torch.stack(adapted_features_z_k_list, dim=0)
            # Weighted sum: (batch_size, feature_dim)
            # self.weights_beta needs to be on the same device
            self.weights_beta = self.weights_beta.to(stacked_features.device)
            # Reshape weights_beta for broadcasting: (num_levels, 1, 1)
            fused_feature_z_hat_e = torch.sum(stacked_features * self.weights_beta.view(-1, 1, 1), dim=0)
            # The RAC paper mentions dividing by sum of betas if they don't sum to 1. [cite: 92]
            # Our example weights sum to 1.0. If not, normalization is needed.
            # fused_feature_z_hat_e = fused_feature_z_hat_e / torch.sum(self.weights_beta)
        elif self.fusion_type == 'LF':
            # Concatenate features along the feature dimension: (batch_size, num_levels * feature_dim)
            concatenated_features = torch.cat(adapted_features_z_k_list, dim=-1)
            fused_feature_z_hat_e = self.fusion_adapter(concatenated_features)
        else:
            # Should not happen due to __init__ check, but as a safeguard:
            raise RuntimeError("Invalid fusion type encountered in forward pass.")

        # --- Project to get enhanced feature z_e ---
        # The RAC paper calls output of Fusion z_hat_e, and output of Projector z_e. [cite: 88, 89]
        enhanced_feature_z_e = self.projector(fused_feature_z_hat_e)

        # --- Generate logits s_MFA from z_e ---
        logits_s_MFA = self.mlp_for_logits(enhanced_feature_z_e)

        return enhanced_feature_z_e, logits_s_MFA


# --- Example Usage (can be run directly for testing) ---
if __name__ == '__main__':
    print("--- Testing MFAModule ---")
    batch_s = 4
    num_cls = 10
    feat_dim = 512 # Target feature dimension (like CLIP's output)

    # Example 1: ResNet-like features
    print("\nTesting with ResNet-like features (image_encoder_channels provided):")
    img_enc_channels_rn = [64, 128, 256, 256] # Dummy channel numbers for 4 levels
    num_lvls_rn = 4
    # Dummy multi-level features (Batch, Channels, H, W) - H,W can vary
    dummy_features_rn = [
        torch.randn(batch_s, img_enc_channels_rn[0], 56, 56),
        torch.randn(batch_s, img_enc_channels_rn[1], 28, 28),
        torch.randn(batch_s, img_enc_channels_rn[2], 14, 14),
        torch.randn(batch_s, img_enc_channels_rn[3], 7, 7),
    ]
    try:
        mfa_module_rn_wf = MFAModule(
            image_encoder_channels=img_enc_channels_rn, feature_dim=feat_dim,
            num_levels=num_lvls_rn, fusion_type='WF', projector_type='identity',
            adapter_bottleneck_ratio=0.5, num_classes=num_cls
        )
        z_e_rn_wf, s_mfa_rn_wf = mfa_module_rn_wf(dummy_features_rn)
        print(f"WF ResNet-like: z_e shape: {z_e_rn_wf.shape}, s_MFA shape: {s_mfa_rn_wf.shape}")
        assert z_e_rn_wf.shape == (batch_s, feat_dim)
        assert s_mfa_rn_wf.shape == (batch_s, num_cls)

        mfa_module_rn_lf = MFAModule(
            image_encoder_channels=img_enc_channels_rn, feature_dim=feat_dim,
            num_levels=num_lvls_rn, fusion_type='LF', projector_type='linear',
            adapter_bottleneck_ratio=0.5, num_classes=num_cls
        )
        z_e_rn_lf, s_mfa_rn_lf = mfa_module_rn_lf(dummy_features_rn)
        print(f"LF ResNet-like: z_e shape: {z_e_rn_lf.shape}, s_MFA shape: {s_mfa_rn_lf.shape}")
        assert z_e_rn_lf.shape == (batch_s, feat_dim)
        assert s_mfa_rn_lf.shape == (batch_s, num_cls)
        print("ResNet-like tests passed.")
    except Exception as e:
        print(f"Error in ResNet-like test: {e}")
        import traceback
        traceback.print_exc()


    # Example 2: ViT-like features (already at feature_dim from different blocks)
    print("\nTesting with ViT-like features (image_encoder_channels is None, features are (B,Tokens,C)):")
    num_lvls_vit = 3
    num_tokens_vit = 197 # e.g., 1 CLS + 196 patch tokens for 224x224 ViT-B/16
    # Dummy multi-level features (Batch, NumTokens, Channels=feat_dim)
    dummy_features_vit = [
        torch.randn(batch_s, num_tokens_vit, feat_dim), # Output of some ViT block
        torch.randn(batch_s, num_tokens_vit, feat_dim), # Output of another ViT block
        torch.randn(batch_s, num_tokens_vit, feat_dim), # Output of a third ViT block
    ]
    try:
        mfa_module_vit_wf = MFAModule(
            image_encoder_channels=None, # Or [feat_dim, feat_dim, feat_dim]
            feature_dim=feat_dim,
            num_levels=num_lvls_vit, fusion_type='WF', projector_type='identity',
            adapter_bottleneck_ratio=0.5, num_classes=num_cls
        )
        # If image_encoder_channels is None, adapters expect input_dim = feature_dim
        # We need to ensure the input to adapters is (B, feature_dim)
        # The current forward pass handles f_k.ndim == 3 by taking f_k[:, 0, :] (CLS token)
        z_e_vit_wf, s_mfa_vit_wf = mfa_module_vit_wf(dummy_features_vit)
        print(f"WF ViT-like: z_e shape: {z_e_vit_wf.shape}, s_MFA shape: {s_mfa_vit_wf.shape}")
        assert z_e_vit_wf.shape == (batch_s, feat_dim)
        assert s_mfa_vit_wf.shape == (batch_s, num_cls)
        print("ViT-like tests passed.")
    except Exception as e:
        print(f"Error in ViT-like test: {e}")
        import traceback
        traceback.print_exc()