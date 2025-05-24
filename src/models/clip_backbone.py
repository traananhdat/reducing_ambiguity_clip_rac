# src/models/clip_backbone.py

import torch
import torch.nn as nn
import clip # OpenAI's CLIP library: pip install git+https://github.com/openai/CLIP.git
# Or, if using open_clip:
# import open_clip

class CLIPVisionTower(nn.Module):
    """
    A wrapper for the CLIP vision model to allow for easier extraction
    of multi-level features if needed, or just the final global feature.
    """
    def __init__(self, clip_vision_model, extract_multi_level=False, num_levels=4):
        super().__init__()
        self.visual = clip_vision_model
        self.extract_multi_level = extract_multi_level
        self.num_levels = num_levels # Number of levels to extract if multi-level

        # Identify if it's a ResNet or ViT to determine how to get multi-level features
        self.is_resnet = "resnet" in str(type(self.visual)).lower() # Heuristic
        self.is_vit = "vit" in str(type(self.visual)).lower() or "visiontransformer" in str(type(self.visual)).lower() # Heuristic

        if self.extract_multi_level and self.is_resnet:
            # For ResNet, typically hook into layer1, layer2, layer3, layer4
            # The actual feature extraction needs to be handled by hooking or modifying forward
            print("Multi-level feature extraction for ResNet needs custom hooking or forward modification.")
            # Storing features from hooks
            self.multi_level_features = {}
            self._register_hooks()

        elif self.extract_multi_level and self.is_vit:
            # For ViT, features can be taken from different transformer blocks
            print("Multi-level feature extraction for ViT can be done by processing specific blocks.")
            # We might need to access self.visual.transformer.resblocks
            # or modify ViT's forward pass slightly.

    def _get_hook(self, name):
        def hook(model, input, output):
            self.multi_level_features[name] = output
        return hook

    def _register_hooks(self):
        if self.is_resnet and self.extract_multi_level:
            # These are common names, but might vary with specific CLIP ResNet implementations
            # You'd need to inspect your CLIP model's `visual` part.
            try:
                self.visual.layer1.register_forward_hook(self._get_hook('layer1'))
                self.visual.layer2.register_forward_hook(self._get_hook('layer2'))
                self.visual.layer3.register_forward_hook(self._get_hook('layer3'))
                self.visual.layer4.register_forward_hook(self._get_hook('layer4'))
                print("Registered hooks for ResNet layers 1-4.")
            except AttributeError:
                print("Could not register hooks for ResNet. Layer names might be different.")


    def forward(self, x: torch.Tensor):
        """
        Forward pass for the vision tower.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor or tuple:
                If extract_multi_level is False, returns global image feature.
                If extract_multi_level is True, returns (global_image_feature, list_of_multi_level_features).
        """
        self.multi_level_features.clear() # Clear previous features

        if self.is_vit:
            # Standard ViT processing for global feature (often the [CLS] token embedding)
            # For multi-level, we might need to modify how the ViT processes blocks
            # or iterate through blocks and store their outputs.
            # This is a simplified example; actual ViT multi-level extraction is more involved.
            # Official OpenAI ViT model's forward method:
            # x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
            # x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            # x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            # x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            # x = x + self.visual.positional_embedding.to(x.dtype)
            # x = self.visual.ln_pre(x)
            # x = x.permute(1, 0, 2)  # NLD -> LND
            # intermediate_outputs = []
            # for i, blk in enumerate(self.visual.transformer.resblocks):
            #     x = blk(x)
            #     if self.extract_multi_level and (len(self.visual.transformer.resblocks) - i <= self.num_levels) :
            #         intermediate_outputs.append(x.permute(1,0,2)[:,1:,:].permute(0,2,1).contiguous().view(x.size(1), -1, int(x.size(0)**0.5) ,int(x.size(0)**0.5))) # Example, needs actual patch reconstruction
            # x = x.permute(1, 0, 2)  # LND -> NLD
            # global_feature = self.visual.ln_post(x[:, 0, :]) # CLS token

            # Simpler: just use the standard encode_image for global
            global_feature = self.visual(x) # Assuming self.visual IS the CLIP vision model

            multi_level_feats = []
            if self.extract_multi_level:
                # Placeholder: Actual ViT multi-level extraction is complex
                # You might pass x through blocks sequentially and store outputs.
                # For now, returning duplicates of global as placeholders if ViT multi-level is complex to implement here
                print("Warning: ViT multi-level feature extraction is simplified in this example.")
                for _ in range(self.num_levels):
                    multi_level_feats.append(global_feature.clone().detach()) # Placeholder

        elif self.is_resnet:
            global_feature = self.visual(x) # Run forward to trigger hooks and get global feature
            multi_level_feats = []
            if self.extract_multi_level:
                # Order them from earlier to later layers
                # The exact feature processing (e.g., pooling) might be needed here or in MFA
                if 'layer1' in self.multi_level_features: multi_level_feats.append(self.multi_level_features['layer1'])
                if 'layer2' in self.multi_level_features: multi_level_feats.append(self.multi_level_features['layer2'])
                if 'layer3' in self.multi_level_features: multi_level_feats.append(self.multi_level_features['layer3'])
                if 'layer4' in self.multi_level_features: multi_level_feats.append(self.multi_level_features['layer4'])
                # Ensure we have the correct number of levels, pad with global if not enough
                while len(multi_level_feats) < self.num_levels and len(multi_level_feats) > 0 :
                    multi_level_feats.append(multi_level_feats[-1]) # duplicate last available
                if not multi_level_feats: # if no hooks worked
                    for _ in range(self.num_levels): multi_level_feats.append(global_feature.clone().detach())


        else: # Fallback or other architectures
            global_feature = self.visual(x)
            multi_level_feats = []
            if self.extract_multi_level:
                for _ in range(self.num_levels):
                    multi_level_feats.append(global_feature.clone().detach()) # Placeholder

        if self.extract_multi_level:
            return global_feature, multi_level_feats
        else:
            return global_feature


class CLIPWithMultiLevelFeatures(nn.Module):
    """
    A wrapper around a CLIP model to provide access to both global features
    and multi-level visual features (if configured).
    """
    def __init__(self, clip_model_name="ViT-B/32", device="cpu", extract_multi_level=False, num_levels=4):
        super().__init__()
        self.device = device
        try:
            self.model, self.preprocess = clip.load(clip_model_name, device=self.device, jit=False)
            print(f"Loaded OpenAI CLIP model: {clip_model_name}")
        except Exception as e_openai:
            print(f"Failed to load OpenAI CLIP model {clip_model_name}: {e_openai}")
            try:
                # Fallback to open_clip if OpenAI fails or for more models
                # import open_clip
                # self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                #     clip_model_name.replace('/', '-'), # open_clip uses '-' e.g., 'ViT-B-32'
                #     pretrained='laion2b_s34b_b79k', # Example pretrained weights
                #     device=self.device
                # )
                # print(f"Loaded open_clip model: {clip_model_name}")
                raise NotImplementedError("open_clip loading part is commented out. Please install and uncomment if needed.")
            except Exception as e_openclip:
                print(f"Failed to load open_clip model {clip_model_name}: {e_openclip}")
                raise RuntimeError(f"Could not load CLIP model '{clip_model_name}' from OpenAI or open_clip.")

        self.vision_tower = CLIPVisionTower(self.model.visual, extract_multi_level, num_levels)
        self.extract_multi_level = extract_multi_level

        # Ensure original CLIP model parameters are frozen by default
        for param in self.model.parameters():
            param.requires_grad = False

    def encode_image(self, image_x):
        """Encodes an image, potentially returning multi-level features."""
        return self.vision_tower(image_x) # This will handle multi-level logic

    def encode_text(self, text_tokens_or_prompts):
        """
        Encodes text.
        Input can be tokenized text or a list of string prompts.
        """
        if isinstance(text_tokens_or_prompts, list) and isinstance(text_tokens_or_prompts[0], str):
            text_tokens = clip.tokenize(text_tokens_or_prompts).to(self.device)
        elif isinstance(text_tokens_or_prompts, torch.Tensor):
            text_tokens = text_tokens_or_prompts.to(self.device)
        else:
            raise TypeError("Input to encode_text must be a list of strings or a tensor of tokens.")
        return self.model.encode_text(text_tokens)

    @property
    def logit_scale(self):
        return self.model.logit_scale

    def forward(self, image_x, text_tokens_or_prompts):
        """
        Standard CLIP forward pass for image-text similarity.
        For RAC model, you'll likely use encode_image and encode_text separately.
        """
        image_features = self.encode_image(image_x)
        if self.extract_multi_level:
            image_features = image_features[0] # Take global feature for this standard forward

        text_features = self.encode_text(text_tokens_or_prompts)

        # Normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


def load_clip_model(model_name, device="cpu", extract_multi_level=False, num_levels=4):
    """
    Loads a CLIP model and its preprocessing function.
    """
    wrapped_model = CLIPWithMultiLevelFeatures(
        clip_model_name=model_name,
        device=device,
        extract_multi_level=extract_multi_level,
        num_levels=num_levels
    )
    return wrapped_model, wrapped_model.preprocess


# --- Example Usage ---
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Test 1: Load CLIP with global features only
    print("\n--- Test 1: CLIP with global features ---")
    try:
        clip_global, preprocess_global = load_clip_model("ViT-B/32", device=device, extract_multi_level=False)
        dummy_image_global = preprocess_global(Image.new('RGB', (224, 224))).unsqueeze(0).to(device)
        dummy_text_global = ["a photo of a cat", "a photo of a dog"]

        with torch.no_grad():
            image_feat_global = clip_global.encode_image(dummy_image_global)
            text_feat_global = clip_global.encode_text(dummy_text_global)

            print(f"Global Image feature shape: {image_feat_global.shape}")
            print(f"Global Text feature shape: {text_feat_global.shape}")
            assert image_feat_global.ndim == 2 # Should be (batch, dim)

            logits_img, _ = clip_global(dummy_image_global, dummy_text_global)
            print(f"Logits per image shape: {logits_img.shape}")

    except Exception as e:
        print(f"Error in Test 1: {e}")
        import traceback
        traceback.print_exc()


    # Test 2: Load CLIP with multi-level features (ResNet example, if RN50 is available)
    print("\n--- Test 2: CLIP with multi-level features (RN50) ---")
    # Note: Multi-level feature extraction for RN50 via hooks is experimental here.
    try:
        clip_multi_rn, preprocess_multi_rn = load_clip_model("RN50", device=device, extract_multi_level=True, num_levels=4)
        dummy_image_multi_rn = preprocess_multi_rn(Image.new('RGB', (224, 224))).unsqueeze(0).to(device)

        with torch.no_grad():
            global_feat_rn, multi_feats_rn = clip_multi_rn.encode_image(dummy_image_multi_rn)
            print(f"RN50 Global Image feature shape: {global_feat_rn.shape}")
            if multi_feats_rn:
                print(f"RN50 Extracted {len(multi_feats_rn)} multi-level features:")
                for i, feat in enumerate(multi_feats_rn):
                    print(f"  Level {i+1} shape: {feat.shape}")
                assert len(multi_feats_rn) == 4 # Expecting 4 levels from ResNet layers
            else:
                print("RN50 Multi-level features not extracted (hooks might have failed or not implemented for this specific model version).")

    except Exception as e:
        print(f"Error in Test 2 (RN50): {e}. This might happen if RN50 is not available or hook registration fails.")
        import traceback
        traceback.print_exc()

    # Test 3: Load CLIP with multi-level features (ViT example)
    print("\n--- Test 3: CLIP with multi-level features (ViT-B/32) ---")
    # Note: Multi-level feature extraction for ViT here is a simplified placeholder.
    try:
        clip_multi_vit, preprocess_multi_vit = load_clip_model("ViT-B/32", device=device, extract_multi_level=True, num_levels=3)
        dummy_image_multi_vit = preprocess_multi_vit(Image.new('RGB', (224, 224))).unsqueeze(0).to(device)

        with torch.no_grad():
            global_feat_vit, multi_feats_vit = clip_multi_vit.encode_image(dummy_image_multi_vit)
            print(f"ViT Global Image feature shape: {global_feat_vit.shape}")
            if multi_feats_vit:
                print(f"ViT Extracted {len(multi_feats_vit)} multi-level features (placeholder):")
                for i, feat in enumerate(multi_feats_vit):
                    print(f"  Level {i+1} shape: {feat.shape}") # Will be same as global in this simplified example
            else:
                print("ViT Multi-level features not extracted.")
    except Exception as e:
        print(f"Error in Test 3 (ViT): {e}")