# src/models/alf_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .adapters import AlphaGenerator # Assuming AlphaGenerator is defined in adapters.py

class ALFModule(nn.Module):
    """
    Adaptive Logits Fusion (ALF) Module.
    This module adaptively fuses logits from two sources (e.g., MFA-like and IRA-like modules)
    using a dynamically generated weight (alpha). The alpha weight is produced by an
    AlphaGenerator that takes an enhanced visual feature (z_e) as input.
    """
    def __init__(self, feature_dim, alpha_generator_bottleneck_ratio=0.5):
        """
        Args:
            feature_dim (int): The dimension of the enhanced feature z_e,
                               which is input to the AlphaGenerator.
            alpha_generator_bottleneck_ratio (float): Bottleneck ratio for the
                                                      internal adapter of the AlphaGenerator.
        """
        super().__init__()
        self.feature_dim = feature_dim

        # Instantiate the AlphaGenerator
        # The RAC paper's AlphaGenerator (Figure 2) takes z_i^e and outputs alpha.
        # It typically has a structure like: Linear -> BatchNorm -> ReLU -> Linear -> Sigmoid
        self.alpha_generator = AlphaGenerator(
            input_dim=feature_dim,
            bottleneck_dim_ratio=alpha_generator_bottleneck_ratio
        )

    def forward(self, logits_source1, logits_source2, enhanced_feature_z_e):
        """
        Forward pass of the ALF module.

        Args:
            logits_source1 (torch.Tensor): Logits from the first source (e.g., s_MFA).
                                          Shape: (batch_size, num_classes).
            logits_source2 (torch.Tensor): Logits from the second source (e.g., s_IRA).
                                          Shape: (batch_size, num_classes).
            enhanced_feature_z_e (torch.Tensor): Enhanced visual feature from MFA/MAF module.
                                               Shape: (batch_size, feature_dim).

        Returns:
            tuple:
                - fused_logits (torch.Tensor): The adaptively fused logits.
                                             Shape: (batch_size, num_classes).
                - alpha (torch.Tensor): The generated alpha weight.
                                        Shape: (batch_size, 1).
        """
        if logits_source1.shape != logits_source2.shape:
            raise ValueError(f"Shapes of logits_source1 ({logits_source1.shape}) and "
                             f"logits_source2 ({logits_source2.shape}) must match.")

        # Generate the adaptive weight alpha using the enhanced feature z_e
        # alpha = E_G_alpha(z_i^e) in RAC paper
        alpha = self.alpha_generator(enhanced_feature_z_e) # Shape: (batch_size, 1)

        # Fuse the logits
        # RAC Paper Eq. 11: s_i^ALF = alpha * s_i^MAF + (1 - alpha) * s_i^ICD
        # Note: Figure 2 in RAC paper shows alpha * s_ICD + (1-alpha) * s_MAF.
        # The text and Eq. 11 are usually followed. We will use s_MFA as source1, s_IRA as source2.
        fused_logits = alpha * logits_source1 + (1 - alpha) * logits_source2

        return fused_logits, alpha

# --- Example Usage (can be run directly for testing) ---
if __name__ == '__main__':
    # Assuming `AlphaGenerator` is defined in `src.models.adapters`
    # Mocking AlphaGenerator for this test:
    class MockAlphaGenerator(nn.Module):
        def __init__(self, input_dim, bottleneck_dim_ratio=0.5):
            super().__init__()
            bottleneck_dim = int(input_dim * bottleneck_dim_ratio)
            if bottleneck_dim == 0 and input_dim > 0 : bottleneck_dim = 1 # Ensure not zero if input > 0
            
            self.fc1 = nn.Linear(input_dim, bottleneck_dim) if bottleneck_dim > 0 else nn.Identity()
            # No BatchNorm in mock for simplicity, real one would have it
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(bottleneck_dim, 1) if bottleneck_dim > 0 else nn.Linear(input_dim, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            if not isinstance(self.fc1, nn.Identity):
                x = self.fc1(x)
                x = self.relu(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            return x

    if 'AlphaGenerator' not in globals(): # If the import from .adapters failed
        AlphaGenerator = MockAlphaGenerator
        print("Using MockAlphaGenerator for testing ALFModule.")

    print("--- Testing ALFModule ---")
    batch_s = 4
    num_cls = 10    # Example number of classes
    feat_dim = 512  # Example feature dimension for z_e

    # Dummy input tensors
    dummy_s_mfa = torch.randn(batch_s, num_cls)  # Logits from MFA-like module
    dummy_s_ira = torch.randn(batch_s, num_cls)  # Logits from IRA-like module
    dummy_z_e = torch.randn(batch_s, feat_dim)   # Enhanced feature

    try:
        # Instantiate the ALFModule
        alf_module_test = ALFModule(
            feature_dim=feat_dim,
            alpha_generator_bottleneck_ratio=0.5
        )

        # Perform a forward pass
        fused_logits_output, alpha_output = alf_module_test(dummy_s_mfa, dummy_s_ira, dummy_z_e)

        print(f"Input s_MFA shape: {dummy_s_mfa.shape}")
        print(f"Input s_IRA shape: {dummy_s_ira.shape}")
        print(f"Input z_e shape: {dummy_z_e.shape}")
        print(f"Output fused_logits shape: {fused_logits_output.shape}")
        print(f"Output alpha shape: {alpha_output.shape}")
        print(f"Sample alpha values: {alpha_output.squeeze().tolist()}")

        # Verify output shapes
        assert fused_logits_output.shape == (batch_s, num_cls)
        assert alpha_output.shape == (batch_s, 1)
        # Verify alpha values are between 0 and 1 (due to Sigmoid in AlphaGenerator)
        assert torch.all(alpha_output >= 0) and torch.all(alpha_output <= 1)
        print("\nALFModule test passed successfully!")

    except Exception as e:
        print(f"\nAn error occurred during ALFModule testing: {e}")
        import traceback
        traceback.print_exc()