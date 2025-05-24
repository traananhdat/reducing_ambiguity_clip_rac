# src/models/adapters.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    """
    A simple adapter module, typically used to adapt features from one dimension
    to another with a bottleneck structure.
    Structure: Linear -> LayerNorm/BatchNorm -> ReLU -> Linear
    As per RAC paper Figure 2 (bottom left), it shows Linear -> BatchNorm -> ReLU -> Linear.
    We'll use BatchNorm1d by default, suitable for (Batch, Features) inputs.
    """
    def __init__(self, input_dim, output_dim, bottleneck_dim_ratio=0.5, norm_layer='batchnorm'):
        """
        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            bottleneck_dim_ratio (float): Ratio to determine the bottleneck dimension
                                          (bottleneck_dim = input_dim * bottleneck_dim_ratio).
            norm_layer (str): 'batchnorm' or 'layernorm'.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        bottleneck_dim = int(input_dim * bottleneck_dim_ratio)
        # Ensure bottleneck_dim is at least 1 if input_dim > 0, or if output_dim is very small
        if input_dim > 0 and bottleneck_dim == 0:
            bottleneck_dim = max(1, min(input_dim // 2, output_dim // 2 if output_dim > 1 else 1))


        if bottleneck_dim <= 0 : # No bottleneck, or input_dim is 0 (should not happen if input_dim > 0)
            self.fc1 = nn.Identity()
            self.norm = nn.Identity()
            self.relu = nn.Identity()
            self.fc2 = nn.Linear(input_dim, output_dim)
        else:
            self.fc1 = nn.Linear(input_dim, bottleneck_dim)
            if norm_layer == 'batchnorm':
                self.norm = nn.BatchNorm1d(bottleneck_dim)
            elif norm_layer == 'layernorm':
                self.norm = nn.LayerNorm(bottleneck_dim)
            else:
                self.norm = nn.Identity() # No normalization
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(bottleneck_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, input_dim) or (batch_size, sequence_length, input_dim)
        original_shape = x.shape
        
        if not isinstance(self.fc1, nn.Identity):
            if x.ndim == 3 and original_shape[-1] == self.input_dim: # (B, S, D)
                # Temporarily reshape for BatchNorm1d or LayerNorm if they expect (B, D) or (B, S, D) respectively
                x_reshaped = x.contiguous().view(-1, self.input_dim) # (B*S, D)
                out = self.fc1(x_reshaped)
                if isinstance(self.norm, nn.BatchNorm1d):
                    out = self.norm(out) # BatchNorm1d expects (N, C)
                elif isinstance(self.norm, nn.LayerNorm):
                    # LayerNorm can handle (..., D), so apply it before reshaping back or after fc1 on (B,S,D)
                    # For simplicity with current structure, we apply to (B*S,D)
                    out = self.norm(out)
                out = self.relu(out)
                out = self.fc2(out)
                out = out.view(original_shape[0], original_shape[1], self.output_dim) # Reshape back to (B, S, D_out)
            elif x.ndim == 2 and original_shape[-1] == self.input_dim: # (B, D)
                out = self.fc1(x)
                if not isinstance(self.norm, nn.Identity):
                    out = self.norm(out)
                out = self.relu(out)
                out = self.fc2(out)
            else:
                raise ValueError(f"Adapter input shape {original_shape} not compatible or input_dim mismatch.")
        else: # No bottleneck path (Identity fc1)
            out = self.fc2(x) # Direct linear transformation

        return out


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP).
    Structure: Linear -> LayerNorm/BatchNorm -> ReLU -> Linear
    Similar to Adapter, but often used as a classifier head.
    RAC paper (Figure 2, MAF module) shows an MLP after z_e to get s_MAF.
    It also describes the Adapter/MLP structure as Linear -> BatchNorm -> ReLU -> Linear.
    """
    def __init__(self, input_dim, output_dim, hidden_dim_ratio=2.0, norm_layer='batchnorm'):
        """
        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features (e.g., num_classes).
            hidden_dim_ratio (float): Ratio to determine the hidden dimension
                                     (hidden_dim = input_dim * hidden_dim_ratio).
            norm_layer (str): 'batchnorm' or 'layernorm'.
        """
        super().__init__()
        hidden_dim = int(input_dim * hidden_dim_ratio)
        if hidden_dim == 0 and input_dim > 0: hidden_dim = max(1, input_dim)


        if hidden_dim <= 0: # No hidden layer
            self.fc1 = nn.Identity()
            self.norm = nn.Identity()
            self.relu = nn.Identity()
            self.fc2 = nn.Linear(input_dim, output_dim)
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            if norm_layer == 'batchnorm':
                self.norm = nn.BatchNorm1d(hidden_dim)
            elif norm_layer == 'layernorm':
                self.norm = nn.LayerNorm(hidden_dim)
            else:
                self.norm = nn.Identity()
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Assuming x shape is (batch_size, input_dim)
        if not isinstance(self.fc1, nn.Identity):
            x = self.fc1(x)
            if not isinstance(self.norm, nn.Identity):
                x = self.norm(x)
            x = self.relu(x)
        x = self.fc2(x)
        return x


class AlphaGenerator(nn.Module):
    """
    Generates the alpha weight for the Adaptive Logits Fusion (ALF) module.
    Structure from RAC paper (Figure 2, bottom center):
    Input z_e, then Linear -> BatchNorm -> ReLU -> Linear -> Sigmoid to output alpha.
    Alpha is a scalar weight per sample.
    """
    def __init__(self, input_dim, bottleneck_dim_ratio=0.5, norm_layer='batchnorm'):
        """
        Args:
            input_dim (int): Dimension of the input feature (e.g., z_e).
            bottleneck_dim_ratio (float): Ratio for the hidden layer dimension.
            norm_layer (str): 'batchnorm' or 'layernorm'.
        """
        super().__init__()
        # Using "d" as a placeholder for bottleneck dimension from Figure 2
        bottleneck_dim_d = int(input_dim * bottleneck_dim_ratio)
        if bottleneck_dim_d == 0 and input_dim > 0: bottleneck_dim_d = max(1, input_dim // 2)

        if bottleneck_dim_d <=0: # No bottleneck
            self.fc1 = nn.Identity()
            self.norm = nn.Identity()
            self.relu = nn.Identity()
            self.fc2 = nn.Linear(input_dim, 1) # Output a single alpha value
        else:
            self.fc1 = nn.Linear(input_dim, bottleneck_dim_d)
            if norm_layer == 'batchnorm':
                self.norm = nn.BatchNorm1d(bottleneck_dim_d)
            elif norm_layer == 'layernorm':
                self.norm = nn.LayerNorm(bottleneck_dim_d)
            else:
                self.norm = nn.Identity()
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(bottleneck_dim_d, 1) # Output a single alpha value

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Assuming x shape is (batch_size, input_dim)
        if not isinstance(self.fc1, nn.Identity):
            x = self.fc1(x)
            if not isinstance(self.norm, nn.Identity):
                x = self.norm(x)
            x = self.relu(x)
        x = self.fc2(x)
        alpha = self.sigmoid(x) # Ensure alpha is between 0 and 1
        return alpha


# --- Example Usage (can be run directly for testing) ---
if __name__ == '__main__':
    batch_s = 4
    input_d = 512
    output_d = 256
    num_cls = 10

    print("--- Testing Adapter ---")
    try:
        adapter_module = Adapter(input_dim=input_d, output_dim=output_d, bottleneck_dim_ratio=0.25)
        dummy_input_2d = torch.randn(batch_s, input_d)
        output_2d = adapter_module(dummy_input_2d)
        print(f"Adapter 2D Input shape: {dummy_input_2d.shape}, Output shape: {output_2d.shape}")
        assert output_2d.shape == (batch_s, output_d)

        dummy_input_3d = torch.randn(batch_s, 50, input_d) # B, SeqLen, Dim
        output_3d = adapter_module(dummy_input_3d)
        print(f"Adapter 3D Input shape: {dummy_input_3d.shape}, Output shape: {output_3d.shape}")
        assert output_3d.shape == (batch_s, 50, output_d)
        print("Adapter tests passed.")
    except Exception as e:
        print(f"Error in Adapter test: {e}")
        import traceback
        traceback.print_exc()


    print("\n--- Testing MLP ---")
    try:
        mlp_module = MLP(input_dim=input_d, output_dim=num_cls, hidden_dim_ratio=1.0)
        dummy_input_mlp = torch.randn(batch_s, input_d)
        output_mlp = mlp_module(dummy_input_mlp)
        print(f"MLP Input shape: {dummy_input_mlp.shape}, Output shape: {output_mlp.shape}")
        assert output_mlp.shape == (batch_s, num_cls)
        print("MLP test passed.")
    except Exception as e:
        print(f"Error in MLP test: {e}")
        import traceback
        traceback.print_exc()


    print("\n--- Testing AlphaGenerator ---")
    try:
        alpha_gen_module = AlphaGenerator(input_dim=input_d, bottleneck_dim_ratio=0.3)
        dummy_input_alpha = torch.randn(batch_s, input_d)
        alpha_output = alpha_gen_module(dummy_input_alpha)
        print(f"AlphaGenerator Input shape: {dummy_input_alpha.shape}, Output alpha shape: {alpha_output.shape}")
        print(f"Sample alpha values: {alpha_output.squeeze().tolist()}")
        assert alpha_output.shape == (batch_s, 1)
        assert torch.all(alpha_output >= 0) and torch.all(alpha_output <= 1)
        print("AlphaGenerator test passed.")
    except Exception as e:
        print(f"Error in AlphaGenerator test: {e}")
        import traceback
        traceback.print_exc()