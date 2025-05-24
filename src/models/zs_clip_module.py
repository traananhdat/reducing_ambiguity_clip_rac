# src/models/zs_clip_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ZeroShotCLIPModule(nn.Module):
    """
    Computes Zero-Shot classification logits using a pre-trained CLIP model.
    It takes an image and pre-computed text features for class prompts
    and outputs the cosine similarity scores (logits) between the image embedding
    and each class's text embedding.
    """
    def __init__(self, clip_model):
        """
        Args:
            clip_model (nn.Module): The pre-trained CLIP model, which should have
                                    `encode_image` and `encode_text` methods,
                                    or a unified interface if it's a custom wrapper.
                                    This module does not train the clip_model.
        """
        super().__init__()
        self.clip_model = clip_model
        # Ensure CLIP model parameters are frozen if this module is part of a larger trainable network
        # This is typically done outside this module, when the main RACModel is initialized.
        # for param in self.clip_model.parameters():
        #     param.requires_grad = False

    def forward(self, image_x, text_features_T):
        """
        Performs the zero-shot classification.

        Args:
            image_x (torch.Tensor): Input images (batch_size, C, H, W).
                                    These should be preprocessed according to CLIP's requirements.
            text_features_T (torch.Tensor): Pre-computed and normalized text features for C classes.
                                          Shape: (num_classes, embedding_dim).

        Returns:
            tuple:
                - logits_s_ZS (torch.Tensor): Zero-shot logits (cosine similarities).
                                            Shape: (batch_size, num_classes).
                - image_features_z_i (torch.Tensor): Normalized image features from CLIP.
                                                  Shape: (batch_size, embedding_dim).
        """
        # Encode the image
        # The RAC paper refers to the image feature as z_i [cite: 70]
        image_features_z_i = self.clip_model.encode_image(image_x) # [batch_size, embedding_dim]

        # Normalize image features (CLIP standard practice)
        image_features_z_i = image_features_z_i / image_features_z_i.norm(dim=-1, keepdim=True)

        # Text features (text_features_T) are assumed to be already normalized
        # as they are pre-computed from text prompts like "a photo of a [class]". [cite: 70]

        # Calculate cosine similarity (dot product of normalized features)
        # This produces the zero-shot logits s_i^ZS [cite: 70, 77]
        # The RAC paper also mentions a temperature parameter (logit_scale) often learned by CLIP.
        # If your clip_model has it, you might apply it here.
        # logits_s_ZS = (image_features_z_i @ text_features_T.T) * self.clip_model.logit_scale.exp()
        # For simplicity, if logit_scale is not directly accessible or handled by encode_image/text:
        logits_s_ZS = image_features_z_i @ text_features_T.T # [batch_size, num_classes]

        # Note: Eq. 4 in the RAC paper applies softmax to get probabilities. [cite: 61]
        # However, for deconfusion and fusion, raw logits *before* softmax are typically used.
        # The term "logits s_i^ZS" in Figure 2 appears before any explicit softmax for classification output. [cite: 70]
        # The final classification probability P(y|x) would involve softmax(logits_s_ZS).

        return logits_s_ZS, image_features_z_i

# --- Example Usage (can be run directly for testing) ---
if __name__ == '__main__':
    # This is a simplified mock of a CLIP model for testing purposes.
    # In a real scenario, you would load an actual pre-trained CLIP model.
    class MockCLIP(nn.Module):
        def __init__(self, embedding_dim=512):
            super().__init__()
            self.embedding_dim = embedding_dim
            # Mock image encoder (e.g., a simple convnet or linear layer)
            self.visual = nn.Linear(3 * 224 * 224, embedding_dim)
            # Mock text encoder (e.g., a simple linear layer)
            # self.text_encoder = nn.Linear(77, embedding_dim) # if input were token IDs
            self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592) # Mock logit_scale

        def encode_image(self, image):
            batch_size = image.shape[0]
            flat_image = image.view(batch_size, -1) # Flatten
            return self.visual(flat_image)

        def encode_text(self, text_prompts_features): # For this mock, assume text_prompts_features are already embeddings
            return text_prompts_features # Pass through

    # Configuration
    embedding_dimension = 512
    num_test_classes = 5
    batch_size_test = 4
    image_height, image_width = 224, 224

    # Instantiate mock CLIP and the module
    mock_clip_model = MockCLIP(embedding_dim=embedding_dimension)
    zs_clip_module_test = ZeroShotCLIPModule(mock_clip_model)

    # Create dummy image input (already preprocessed)
    dummy_images = torch.randn(batch_size_test, 3, image_height, image_width)

    # Create dummy pre-computed text features (normalized)
    dummy_text_features = torch.randn(num_test_classes, embedding_dimension)
    dummy_text_features = dummy_text_features / dummy_text_features.norm(dim=-1, keepdim=True)

    print("--- Testing ZeroShotCLIPModule ---")
    # Perform forward pass
    try:
        logits, image_feats = zs_clip_module_test(dummy_images, dummy_text_features)

        print(f"Output logits shape: {logits.shape}") # Expected: [batch_size, num_classes]
        print(f"Output image features shape: {image_feats.shape}") # Expected: [batch_size, embedding_dim]
        print("Logits sample:\n", logits[0])
        print("Image features sample (norm):", image_feats[0].norm().item())

        # Verify shapes
        assert logits.shape == (batch_size_test, num_test_classes)
        assert image_feats.shape == (batch_size_test, embedding_dimension)
        print("\nTest passed successfully!")

    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
        import traceback
        traceback.print_exc()