# src/models/ira_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .adapters import Adapter # Assuming Adapter is defined here

class IRAModule(nn.Module):
    """
    Inter-class Ambiguity Reduction (IRA) Module.
    Conceptually similar to the Inter-Class Deconfusion (ICD) module from the RAC.
    It learns inter-class confusion/ambiguity patterns from zero-shot logits (s_ZS)
    using an enhanced visual feature (z_e) as a prior, and then removes these
    patterns via a residual structure to produce cleaner logits (s_IRA).
    """
    def __init__(self, num_classes, feature_dim, adapter_bottleneck_ratio=0.5):
        """
        Args:
            num_classes (int): The number of classes (dimension of the logits).
            feature_dim (int): The dimension of the enhanced feature z_e.
            adapter_bottleneck_ratio (float): Bottleneck ratio for the Adapters.
        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Input: s_i^ZS (num_classes), Output: (num_classes)
        self.adapter_A1 = Adapter(num_classes, num_classes, bottleneck_dim_ratio=adapter_bottleneck_ratio)

        # Input: z_i^e (feature_dim), Output: (num_classes) - to match dimension for addition
        self.adapter_A2 = Adapter(feature_dim, num_classes, bottleneck_dim_ratio=adapter_bottleneck_ratio)

        # Input: sum of A1 and A2 outputs (num_classes), Output: (num_classes) - representing the learned confusion delta_s
        self.adapter_A3 = Adapter(num_classes, num_classes, bottleneck_dim_ratio=adapter_bottleneck_ratio)

    def forward(self, logits_s_ZS, enhanced_feature_z_e):
        """
        Forward pass of the IRA module.

        Args:
            logits_s_ZS (torch.Tensor): Zero-shot logits from ZS-CLIP.
                                      Shape: (batch_size, num_classes).
            enhanced_feature_z_e (torch.Tensor): Enhanced visual feature from MFA/MAF module.
                                               Shape: (batch_size, feature_dim).

        Returns:
            torch.Tensor: Deconfused/ambiguity-reduced logits s_IRA.
                          Shape: (batch_size, num_classes).
        """
        # Output of Adapter A1, learning from s_ZS [cite: 94]
        out_A1 = self.adapter_A1(logits_s_ZS) # E_A1_ICD(s_i^ZS)

        # Output of Adapter A2, learning from z_e [cite: 94]
        out_A2 = self.adapter_A2(enhanced_feature_z_e) # E_A2_ICD(z_i^e)

        # Combine the outputs of A1 and A2 by element-wise addition [cite: 94]
        # This forms the input to Adapter A3
        combined_A1_A2 = out_A1 + out_A2

        # Or, more specifically, E_A3_ICD(E_A1_ICD(s_i^ZS) + E_A2_ICD(z_i^e))
        learned_ambiguity_pattern_delta_s = self.adapter_A3(combined_A1_A2)

        # Remove the learned ambiguity pattern from the original zero-shot logits
        # The text explanation ("removes them through a residual structure", "remove the learned inter-class confusion pattern") [cite: 28, 95]
        # strongly suggests subtraction, aligning with Eq. 10.
        # We will follow Eq. 10.
        logits_s_IRA = logits_s_ZS - learned_ambiguity_pattern_delta_s

        return logits_s_IRA

# --- Example Usage (can be run directly for testing) ---
if __name__ == '__main__':
    # Assuming `Adapter` is defined in `src.models.adapters`
    # For standalone testing, we might need to mock it or ensure the import works.
    # Mocking Adapter for this test:
    class MockAdapter(nn.Module):
        def __init__(self, input_dim, output_dim, bottleneck_dim_ratio=0.5):
            super().__init__()
            bottleneck_dim = int(input_dim * bottleneck_dim_ratio)
            if bottleneck_dim == 0 and input_dim > 0 : bottleneck_dim = output_dim # Ensure bottleneck is not 0 if input > 0
            self.fc1 = nn.Linear(input_dim, bottleneck_dim) if bottleneck_dim > 0 else nn.Identity()
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(bottleneck_dim, output_dim) if bottleneck_dim > 0 else nn.Linear(input_dim, output_dim)


        def forward(self, x):
            if not isinstance(self.fc1, nn.Identity):
                x = self.fc1(x)
                x = self.relu(x)
            x = self.fc2(x)
            return x

    # Replace the import if Adapter is in a different relative path for direct execution
    # For this example, let's assume we are overriding the import for testing
    if 'Adapter' not in globals(): # If the import from .adapters failed (e.g. running standalone)
        Adapter = MockAdapter
        print("Using MockAdapter for testing IRAModule.")


    print("--- Testing IRAModule ---")
    batch_s = 4
    num_cls = 10  # Example number of classes
    feat_dim = 512 # Example feature dimension for z_e

    # Dummy input tensors
    dummy_s_ZS = torch.randn(batch_s, num_cls)
    dummy_z_e = torch.randn(batch_s, feat_dim)

    try:
        # Instantiate the IRAModule
        ira_module_test = IRAModule(
            num_classes=num_cls,
            feature_dim=feat_dim,
            adapter_bottleneck_ratio=0.5
        )

        # Perform a forward pass
        output_logits_s_IRA = ira_module_test(dummy_s_ZS, dummy_z_e)

        print(f"Input s_ZS shape: {dummy_s_ZS.shape}")
        print(f"Input z_e shape: {dummy_z_e.shape}")
        print(f"Output s_IRA shape: {output_logits_s_IRA.shape}")

        # Verify output shape
        assert output_logits_s_IRA.shape == (batch_s, num_cls)
        print("\nIRAModule test passed successfully!")

    except Exception as e:
        print(f"\nAn error occurred during IRAModule testing: {e}")
        import traceback
        traceback.print_exc()