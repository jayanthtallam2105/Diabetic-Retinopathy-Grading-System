import asyncio
import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image

from utils import NUM_CLASSES


@dataclass(frozen=True)
class ModelSpec:
    key: str
    weights_filename: str
    arch: str


def get_device() -> torch.device:
    """
    Pick the best available torch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_image_bytes(image_bytes: bytes, device: torch.device) -> torch.Tensor:
    """
    EXACT match to training preprocessing:

      resize 224x224
      ToTensor()  -> float32 in [0,1], CHW
      Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

    Returns: tensor shape [1, 3, 224, 224] on `device`.
    """
    if not image_bytes:
        raise ValueError("Empty image bytes.")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))

    # ToTensor(): HWC uint8 -> float32 [0,1] CHW
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
    arr = np.transpose(arr, (2, 0, 1))  # CHW

    # Normalize(mean=0.5,std=0.5) per channel
    arr = (arr - 0.5) / 0.5

    x = torch.from_numpy(arr).unsqueeze(0).to(device)  # [1,3,224,224]
    return x


def _load_state_dict_strict(model: nn.Module, weights_path: Path, device: torch.device) -> None:
    """
    Load state dict with strict=True and helpful error messages.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")

    obj = torch.load(weights_path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise ValueError(
            f"Unsupported checkpoint format in {weights_path}. "
            "Expected a state_dict dict (or dict with 'state_dict')."
        )

    model.load_state_dict(state_dict, strict=True)


def _make_single_model(arch: str) -> nn.Module:
    """
    Single-backbone model: timm.create_model(..., num_classes=5)
    """
    return timm.create_model(arch, pretrained=False, num_classes=NUM_CLASSES)


class HybridEffViT(nn.Module):
    """
    HybridEffViT feature-fusion architecture (MUST match training):

      EfficientNet-B0 reset_classifier(0) -> 1280-d features
      ViT-base reset_classifier(0)        -> 768-d features
      concat -> 2048
      Linear(2048->512) -> BN -> ReLU -> Dropout -> Linear(512->5)
    """

    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
        self.eff = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)
        self.eff.reset_classifier(0)  # features: 1280

        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES)
        self.vit.reset_classifier(0)  # features: 768

        # Classifier head: MUST match training (nn.Sequential -> fc.0, fc.1, fc.4)
        self.fc = nn.Sequential(
            nn.Linear(1280 + 768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eff_feat = self.eff(x)  # [B,1280]
        vit_feat = self.vit(x)  # [B,768]
        feat = torch.cat([eff_feat, vit_feat], dim=1)  # [B,2048]
        return self.fc(feat)


class HybridResViT(nn.Module):
    """
    HybridResViT feature-fusion architecture (MUST match training):

      ResNet50 reset_classifier(0) -> 2048-d features
      ViT-base reset_classifier(0) -> 768-d features
      concat -> 2816
      Linear(2816->512) -> BN -> ReLU -> Dropout -> Linear(512->5)
    """

    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
        self.res = timm.create_model("resnet50", pretrained=False, num_classes=NUM_CLASSES)
        self.res.reset_classifier(0)  # features: 2048

        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES)
        self.vit.reset_classifier(0)  # features: 768

        # Classifier head: MUST match training (nn.Sequential -> fc.0, fc.1, fc.4)
        self.fc = nn.Sequential(
            nn.Linear(2048 + 768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res_feat = self.res(x)  # [B,2048]
        vit_feat = self.vit(x)  # [B,768]
        feat = torch.cat([res_feat, vit_feat], dim=1)  # [B,2816]
        return self.fc(feat)


def build_models(models_dir: Path, device: torch.device) -> Dict[str, nn.Module]:
    """
    Load all models ONCE at startup (strict=True).
    If some weights are missing, those models are skipped and
    will later report an error payload in /predict.
    """
    specs = [
        # Allow either "efficientnet.pt" or "efficientnet_final.pt"
        ModelSpec(key="efficientnet", weights_filename="efficientnet.pt", arch="efficientnet_b0"),
        ModelSpec(key="resnet50", weights_filename="resnet50.pt", arch="resnet50"),
        ModelSpec(key="vit", weights_filename="vit.pt", arch="vit_base_patch16_224"),
    ]

    models: Dict[str, nn.Module] = {}

    # Single models
    for spec in specs:
        try:
            m = _make_single_model(spec.arch)
            weights_path = models_dir / spec.weights_filename

            # Special-case efficientnet: fall back to efficientnet_final.pt if that exists.
            if spec.key == "efficientnet" and not weights_path.exists():
                alt = models_dir / "efficientnet_final.pt"
                if alt.exists():
                    weights_path = alt

            _load_state_dict_strict(m, weights_path, device)
            m.to(device)
            m.eval()
            models[spec.key] = m
            print(f"[Models] Loaded {spec.key} from {weights_path}")
        except FileNotFoundError as e:
            print(f"[Models] Skipping {spec.key}: {e}")
        except Exception as e:
            print(f"[Models] Failed to load {spec.key}: {e}")

    # Hybrid models (feature fusion) – load if weights are present
    try:
        effvit = HybridEffViT()
        _load_state_dict_strict(effvit, models_dir / "hybrid_effvit.pt", device)
        effvit.to(device)
        effvit.eval()
        models["hybrid_effvit"] = effvit
        print("[Models] Loaded hybrid_effvit")
    except FileNotFoundError as e:
        print(f"[Models] Skipping hybrid_effvit: {e}")
    except Exception as e:
        print(f"[Models] Failed to load hybrid_effvit: {e}")

    try:
        resvit = HybridResViT()
        _load_state_dict_strict(resvit, models_dir / "hybrid_resvit.pt", device)
        resvit.to(device)
        resvit.eval()
        models["hybrid_resvit"] = resvit
        print("[Models] Loaded hybrid_resvit")
    except FileNotFoundError as e:
        print(f"[Models] Skipping hybrid_resvit: {e}")
    except Exception as e:
        print(f"[Models] Failed to load hybrid_resvit: {e}")

    return models


def _find_target_conv_layer(target: nn.Module, use_earlier: bool = True) -> Optional[nn.Module]:
    """
    Find convolutional layer for CAM generation.
    For better localization, uses earlier layers (not final block).
    """
    target_layer = None
    
    try:
        if hasattr(target, "blocks"):
            # EfficientNet: use second-to-last block for better spatial resolution
            if use_earlier and len(target.blocks) >= 2:
                target_block = target.blocks[-2]  # Earlier layer
            else:
                target_block = target.blocks[-1]
            
            # Find last conv in selected block
            for module in reversed(list(target_block.modules())):
                if isinstance(module, nn.Conv2d):
                    target_layer = module
                    break
        elif hasattr(target, "layer4"):
            # ResNet: use layer3 for better spatial resolution (earlier than layer4)
            if use_earlier and hasattr(target, "layer3"):
                target_layer_group = target.layer3
            else:
                target_layer_group = target.layer4
            
            # Find last conv in selected layer
            for module in reversed(list(target_layer_group.modules())):
                if isinstance(module, nn.Conv2d):
                    target_layer = module
                    break
        else:
            # Generic: find second-to-last Conv2d for better resolution
            all_convs = [(name, module) for name, module in target.named_modules() if isinstance(module, nn.Conv2d)]
            if use_earlier and len(all_convs) >= 2:
                # Use second-to-last conv
                target_layer = all_convs[-2][1]
            elif all_convs:
                target_layer = all_convs[-1][1]
    except Exception:
        pass
    
    # Fallback: find last Conv2d anywhere
    if target_layer is None:
        for name, module in reversed(list(target.named_modules())):
            if isinstance(module, nn.Conv2d):
                target_layer = module
                break
    
    return target_layer


def _generate_gradcam_plusplus(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    cnn_module: Optional[nn.Module] = None,
) -> Optional[np.ndarray]:
    """
    GradCAM++ implementation for better localization.
    Uses weighted combination of positive gradients for more precise heatmaps.
    Returns a 2D numpy array in [0,1] or None on failure.
    """
    target = cnn_module if cnn_module is not None else model
    
    activations = None
    gradients = None
    hook_handles = []

    def fwd_hook(_module, _inp, out):
        nonlocal activations
        activations = out

    def bwd_hook(_module, _grad_in, grad_out):
        nonlocal gradients
        if grad_out[0] is not None:
            gradients = grad_out[0]

    target_layer = _find_target_conv_layer(target, use_earlier=True)
    if target_layer is None:
        return None

    handle_f = target_layer.register_forward_hook(fwd_hook)
    handle_b = target_layer.register_full_backward_hook(bwd_hook)
    hook_handles = [handle_f, handle_b]

    try:
        model.zero_grad(set_to_none=True)
        if not x.requires_grad:
            x.requires_grad_(True)
        
        logits = model(x)  # [1, num_classes]
        score = logits[0, target_class]
        score.backward(retain_graph=False)

        if activations is None or gradients is None:
            return None

        if len(activations.shape) != 4 or len(gradients.shape) != 4:
            return None

        # GradCAM++: weighted combination using positive gradients
        # More localized than standard Grad-CAM
        B, C, H, W = activations.shape
        
        # Compute alpha (importance weights) using positive gradients
        alpha = torch.relu(gradients) / (torch.sum(torch.relu(gradients), dim=(2, 3), keepdim=True) + 1e-8)
        
        # Weighted combination
        weights = torch.sum(alpha * torch.relu(gradients), dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [B, 1, H, W]
        cam = torch.relu(cam)
        cam = cam[0, 0]  # [H, W]
        
        # Normalize to [0, 1] with better contrast
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
            # Apply power transform for better localization (emphasize high activations)
            cam = torch.pow(cam, 0.8)
        else:
            cam = torch.zeros_like(cam)
        
        return cam.detach().cpu().numpy()
    except Exception:
        return None
    finally:
        for handle in hook_handles:
            handle.remove()


def _generate_scorecam(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    cnn_module: Optional[nn.Module] = None,
    num_samples: int = 16,
) -> Optional[np.ndarray]:
    """
    ScoreCAM implementation: uses forward passes with masked inputs.
    More stable but slower than GradCAM++.
    Returns a 2D numpy array in [0,1] or None on failure.
    """
    target = cnn_module if cnn_module is not None else model
    
    activations = None
    hook_handles = []

    def fwd_hook(_module, _inp, out):
        nonlocal activations
        activations = out

    target_layer = _find_target_conv_layer(target, use_earlier=True)
    if target_layer is None:
        return None

    handle_f = target_layer.register_forward_hook(fwd_hook)
    hook_handles = [handle_f]

    try:
        # Get baseline activation
        with torch.no_grad():
            _ = model(x)
        
        if activations is None or len(activations.shape) != 4:
            return None
        
        B, C, H, W = activations.shape
        
        # Normalize activations to [0, 1] for masking
        act_norm = activations[0]  # [C, H, W]
        act_min = act_norm.view(C, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)  # [C, 1, 1]
        act_max = act_norm.view(C, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)  # [C, 1, 1]
        act_range = act_max - act_min + 1e-8
        act_norm = (act_norm - act_min) / act_range  # [C, H, W]
        
        # Upsample activations to input size for masking
        act_upsampled = torch.nn.functional.interpolate(
            act_norm.unsqueeze(0), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False
        )[0]  # [C, H_in, W_in]
        
        # Sample channels (use top-k by variance for efficiency)
        channel_var = act_norm.view(C, -1).var(dim=1)  # [C]
        top_k = min(num_samples, C)
        top_channels = torch.topk(channel_var, top_k).indices
        
        scores = []
        masks = []
        
        for ch_idx in top_channels:
            mask = act_upsampled[ch_idx:ch_idx+1]  # [1, H, W]
            masks.append(mask)
            
            # Mask input and forward pass
            masked_input = x * mask.unsqueeze(0)  # [1, 1, H, W] * [1, 3, H, W]
            with torch.no_grad():
                logits = model(masked_input)
                score = logits[0, target_class].item()
            scores.append(max(0.0, score))  # Only positive contributions
        
        if not scores or max(scores) == 0:
            return None
        
        # Weighted combination
        scores_t = torch.tensor(scores, device=x.device)
        scores_t = scores_t / (scores_t.sum() + 1e-8)
        
        cam = torch.zeros(H, W, device=x.device)
        for i, ch_idx in enumerate(top_channels):
            mask = act_norm[ch_idx]  # [H, W]
            cam += scores_t[i] * mask
        
        cam = torch.relu(cam)
        
        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
            cam = torch.pow(cam, 0.8)  # Emphasize high activations
        else:
            cam = torch.zeros_like(cam)
        
        return cam.detach().cpu().numpy()
    except Exception:
        return None
    finally:
        for handle in hook_handles:
            handle.remove()


def _generate_vit_attention_rollout(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
) -> Optional[np.ndarray]:
    """
    Attention Rollout for Vision Transformer.
    Aggregates attention across all transformer blocks to show global context.
    Returns a 2D numpy array in [0,1] or None on failure.
    This is NOT lesion localization - it shows global attention patterns.
    """
    all_attentions = []
    hook_handles = []

    def attn_fwd_hook(_module, _inp, out):
        # timm ViT attention: out is tuple (attn_output, attn_weights)
        if isinstance(out, tuple) and len(out) >= 2:
            attn = out[1]  # [B, num_heads, num_patches+1, num_patches+1]
            if attn is not None:
                all_attentions.append(attn.detach())

    try:
        # Hook all transformer blocks
        if not hasattr(model, "blocks") or len(model.blocks) == 0:
            return None
        
        for block in model.blocks:
            if hasattr(block, "attn"):
                handle = block.attn.register_forward_hook(attn_fwd_hook)
                hook_handles.append(handle)

        # Forward pass (no gradients needed for attention rollout)
        with torch.no_grad():
            _ = model(x)

        if not all_attentions:
            return None

        # Get number of patches
        num_patches = None
        if hasattr(model, "patch_embed"):
            img_size = 224
            patch_size = model.patch_embed.patch_size[0] if hasattr(model.patch_embed, 'patch_size') else 16
            num_patches = (img_size // patch_size) ** 2
        
        if num_patches is None:
            num_patches = 196  # Default for ViT-B/16

        # Attention Rollout: multiply attention matrices across layers
        # Start with identity matrix
        rollout = torch.eye(num_patches + 1, device=x.device).unsqueeze(0)  # [1, num_patches+1, num_patches+1]
        
        for attn in all_attentions:
            # attn: [B, num_heads, num_patches+1, num_patches+1]
            # Average across heads
            attn_avg = attn[0].mean(dim=0)  # [num_patches+1, num_patches+1]
            # Add residual connection (identity) and multiply
            attn_residual = attn_avg + torch.eye(num_patches + 1, device=x.device)
            rollout = torch.matmul(rollout, attn_residual)

        # Extract attention from CLS token (index 0) to all patches
        attn_to_patches = rollout[0, 0, 1:]  # [num_patches]
        cam_1d = attn_to_patches.cpu()

        # Normalize
        cam_1d = torch.relu(cam_1d)
        cam_min = cam_1d.min()
        cam_max = cam_1d.max()
        if cam_max - cam_min > 1e-8:
            cam_1d = (cam_1d - cam_min) / (cam_max - cam_min)
        else:
            cam_1d = torch.zeros_like(cam_1d)

        # Reshape to spatial dimensions
        patch_size_side = int(np.sqrt(num_patches))
        if patch_size_side * patch_size_side == num_patches:
            cam_2d = cam_1d.reshape(patch_size_side, patch_size_side)
        else:
            patch_size_side = int(np.sqrt(num_patches))
            cam_2d = cam_1d[:patch_size_side * patch_size_side].reshape(patch_size_side, patch_size_side)
        
        # Upsample to image size (224x224) with smooth interpolation
        cam_2d = torch.nn.functional.interpolate(
            cam_2d.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )[0, 0]  # [224, 224]
        
        # Soft normalization (no sharp thresholding for ViT - it's global context)
        cam_2d = torch.relu(cam_2d)
        cam_min = cam_2d.min()
        cam_max = cam_2d.max()
        if cam_max - cam_min > 1e-8:
            cam_2d = (cam_2d - cam_min) / (cam_max - cam_min)
            # Softer power transform for ViT (less aggressive than CNN)
            cam_2d = torch.pow(cam_2d, 0.6)
        else:
            cam_2d = torch.zeros_like(cam_2d)
        
        return cam_2d.detach().cpu().numpy()
    except Exception as e:
        print(f"[ViT Attention Rollout] Error: {e}")
        return None
    finally:
        for handle in hook_handles:
            handle.remove()


def _generate_cam_for_module(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    cnn_module: Optional[nn.Module] = None,
    method: str = "gradcam++",
) -> Optional[np.ndarray]:
    """
    Generate CAM using specified method (gradcam++ or scorecam).
    Defaults to GradCAM++ for better localization.
    For ViT models, uses attention-based GradCAM.
    """
    if method == "scorecam":
        return _generate_scorecam(model, x, target_class, cnn_module)
    else:
        return _generate_gradcam_plusplus(model, x, target_class, cnn_module)


def _apply_colormap(cam_arr: np.ndarray) -> np.ndarray:
    """
    Apply TURBO colormap (medical standard) to CAM array using OpenCV.
    Returns RGB heatmap [H, W, 3] in [0, 1].
    """
    # Convert to uint8 [0, 255] for OpenCV
    cam_uint8 = (cam_arr * 255.0).astype(np.uint8)
    
    # Apply TURBO colormap (medical standard for heatmaps)
    heat_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_TURBO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    return heat_rgb.astype(np.float32) / 255.0


def _create_retina_mask(img: np.ndarray) -> np.ndarray:
    """
    Create a circular retinal mask to zero out background.
    Uses brightness thresholding and largest contour detection.
    Returns binary mask [H, W] with 1.0 inside retina, 0.0 outside.
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (img * 255.0).astype(np.uint8)
    
    # Threshold to find bright regions (retina is typically brighter than background)
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    
    # Find largest contour (should be the retina)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback: return full mask if no contour found
        return np.ones_like(gray, dtype=np.float32)
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create mask from largest contour
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.fillPoly(mask, [largest_contour], 255)
    
    # Optional: fit ellipse for smoother circular mask
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        mask_ellipse = np.zeros_like(gray, dtype=np.uint8)
        cv2.ellipse(mask_ellipse, ellipse, 255, -1)
        # Use ellipse if it's reasonable size
        if cv2.countNonZero(mask_ellipse) > 0.1 * mask.size:
            mask = mask_ellipse
    
    return (mask / 255.0).astype(np.float32)


def _suppress_optic_disc(cam: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    Detect and suppress the optic disc (brightest circular region) in CAM.
    Prevents optic disc from dominating GradCAM visualization.
    Returns CAM with suppressed optic disc region.
    """
    # Convert image to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (img * 255.0).astype(np.uint8)
    
    # Detect circular regions (optic disc is typically circular and bright)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=min(gray.shape) // 4,
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # Suppress the brightest/largest circle (likely optic disc)
        for (cx, cy, r) in circles:
            # Create circular mask
            y, x = np.ogrid[:cam.shape[0], :cam.shape[1]]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
            
            # Suppress CAM in this region (reduce to 30% of original)
            cam[mask] *= 0.3
    
    return cam


def _extract_bounding_boxes_medical(
    cam_arr: np.ndarray,
    threshold: float = 0.6,
    min_area_ratio: float = 0.01,
    top_k: int = 3,
) -> list:
    """
    Medical-grade bounding box extraction:
    - CAM must be resized to image size BEFORE calling this
    - Gaussian blur for noise reduction
    - Area filtering (ignore small speckles)
    - Top-K boxes by mean CAM activation
    
    Returns list of {x, y, w, h} dicts for high-activation regions.
    """
    # Apply Gaussian blur to reduce noise
    cam_blurred = cv2.GaussianBlur(cam_arr, (11, 11), 0)
    
    # Threshold CAM to binary mask
    cam_uint8 = (cam_blurred * 255.0).astype(np.uint8)
    threshold_val = int(threshold * 255)
    _, binary = cv2.threshold(cam_uint8, threshold_val, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # Calculate minimum area (1% of image area)
    image_area = cam_arr.shape[0] * cam_arr.shape[1]
    min_area = min_area_ratio * image_area
    
    # Filter contours by area and compute mean CAM activation per box
    box_candidates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract CAM region for this box
        box_cam = cam_arr[y:y+h, x:x+w]
        mean_activation = np.mean(box_cam)
        
        box_candidates.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "mean_activation": float(mean_activation),
        })
    
    if not box_candidates:
        return []
    
    # Sort by mean activation (descending) and keep top-K
    box_candidates.sort(key=lambda b: b["mean_activation"], reverse=True)
    top_boxes = box_candidates[:top_k]
    
    # Return only coordinates (remove mean_activation from output)
    return [{"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]} for b in top_boxes]


def _cam_overlay_to_base64(
    cam: np.ndarray,
    x: torch.Tensor,
    alpha: float = 0.6,
    draw_boxes: bool = False,
) -> Tuple[str, list]:
    """
    Medical-grade CAM overlay with retina masking, optic disc suppression, and clean bounding boxes.
    Uses TURBO colormap and yellow bounding boxes.
    
    CRITICAL: CAM is resized to image size (224x224) BEFORE finding contours.
    
    Args:
        cam: 2D numpy array [H, W] in [0, 1] (may be smaller than 224x224)
        x: Normalized tensor [1, 3, 224, 224]
        alpha: Overlay opacity (0-1)
        draw_boxes: If True, draw bounding boxes on overlay (CNN models only)
    
    Returns:
        (base64 PNG string, list of bounding boxes)
    """
    # Denormalize input tensor back to [0,1] and HWC
    img_t = x[0].detach().cpu().numpy()  # [3,224,224]
    img_t = (img_t * 0.5) + 0.5
    img_t = np.clip(img_t, 0.0, 1.0)
    img = np.transpose(img_t, (1, 2, 0))  # [224,224,3]

    # STEP 1: Resize CAM to image size (224x224) BEFORE any processing
    # This ensures correct coordinate mapping for bounding boxes
    if cam.shape != img.shape[:2]:
        cam_img = Image.fromarray((cam * 255.0).astype("uint8")).resize(
            (img.shape[1], img.shape[0]), resample=Image.BILINEAR
        )
        cam_arr = np.asarray(cam_img, dtype=np.float32) / 255.0  # [224,224]
    else:
        cam_arr = cam.copy()

    # STEP 2: Create retina mask and apply to CAM (zero out background)
    retina_mask = _create_retina_mask(img)
    cam_arr = cam_arr * retina_mask

    # STEP 3: Suppress optic disc to prevent dominance
    cam_arr = _suppress_optic_disc(cam_arr, img)

    # STEP 4: Extract bounding boxes AFTER resizing and masking
    # This ensures boxes are in correct coordinates
    boxes = []
    if draw_boxes:
        boxes = _extract_bounding_boxes_medical(
            cam_arr, threshold=0.6, min_area_ratio=0.01, top_k=3
        )

    # STEP 5: Apply threshold to focus on high-activation regions
    threshold = 0.2
    cam_arr_thresh = np.clip((cam_arr - threshold) / (1.0 - threshold + 1e-8), 0.0, 1.0)

    # STEP 6: Apply TURBO colormap (medical standard)
    heat = _apply_colormap(cam_arr_thresh)

    # STEP 7: Blend original + heat
    overlay = (1.0 - alpha) * img + alpha * heat
    overlay = np.clip(overlay, 0.0, 1.0)

    # STEP 8: Convert to uint8 for OpenCV drawing
    overlay_uint8 = (overlay * 255.0).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR)

    # STEP 9: Draw yellow bounding boxes (medical standard)
    if draw_boxes and boxes:
        for box in boxes:
            # Yellow boxes (BGR: 0, 255, 255) for medical visualization
            cv2.rectangle(
                overlay_bgr,
                (box["x"], box["y"]),
                (box["x"] + box["w"], box["y"] + box["h"]),
                (0, 255, 255),  # Yellow in BGR
                2,  # 2px thickness
            )

    # Convert back to RGB and encode
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    out_img = Image.fromarray(overlay_rgb)
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii"), boxes


def _generate_heatmap_only_base64(cam: np.ndarray, img: Optional[np.ndarray] = None) -> str:
    """
    Generate heatmap-only visualization (no overlay) with TURBO colormap.
    Applies retina masking and optic disc suppression for medical-grade visualization.
    Returns base64 PNG string.
    """
    # Resize to 224x224 if needed
    if cam.shape[0] < 224 or cam.shape[1] < 224:
        cam_img = Image.fromarray((cam * 255.0).astype("uint8")).resize(
            (224, 224), resample=Image.BILINEAR
        )
        cam_arr = np.asarray(cam_img, dtype=np.float32) / 255.0
    else:
        cam_arr = cam.copy()

    # Apply retina masking and optic disc suppression if image provided
    if img is not None:
        retina_mask = _create_retina_mask(img)
        cam_arr = cam_arr * retina_mask
        cam_arr = _suppress_optic_disc(cam_arr, img)

    # Apply threshold
    threshold = 0.2
    cam_arr_thresh = np.clip((cam_arr - threshold) / (1.0 - threshold + 1e-8), 0.0, 1.0)

    # Apply TURBO colormap
    heat = _apply_colormap(cam_arr_thresh)

    # Convert to PIL and encode
    out_img = Image.fromarray((heat * 255.0).astype("uint8"))
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def predict_with_model(
    name: str,
    model: nn.Module,
    x_in: torch.Tensor,
) -> Tuple[int, float, list, Optional[Dict[str, Any]]]:
    """
    Returns:
      grade: int [0..4]
      confidence: float (max softmax prob)
      probs: list[5 floats]
      gradcam: Optional[dict] with overlays for different colormaps and heatmap
    """
    # First, get predictions without gradients (faster)
    with torch.no_grad():
        logits = model(x_in)  # [1,5]
        probs_t = torch.softmax(logits, dim=1)[0]  # [5]
        conf_t, grade_t = torch.max(probs_t, dim=0)

    grade = int(grade_t.item())
    confidence = float(conf_t.item())
    probs = [float(v) for v in probs_t.tolist()]

    # GradCAM support for all models (requires gradients)
    gradcam_data: Optional[Dict[str, Any]] = None
    try:
        cam: Optional[np.ndarray] = None
        method_name = "gradcam++"
        
        is_cnn = False
        if name == "vit":
            # ViT: GradCAM not available (attention rollout will be added later)
            cam = None
            method_name = None
        else:
            # CNN models: use GradCAM++ with bounding boxes
            is_cnn = True
            cnn_module: Optional[nn.Module] = None
            if name == "efficientnet":
                cnn_module = model
            elif name == "resnet50":
                cnn_module = model
            elif name == "hybrid_effvit" and hasattr(model, "eff"):
                cnn_module = model.eff
            elif name == "hybrid_resvit" and hasattr(model, "res"):
                cnn_module = model.res

            if cnn_module is not None:
                x_grad = x_in.clone().detach().requires_grad_(True)
                cam = _generate_cam_for_module(model, x_grad, grade, cnn_module=cnn_module, method="gradcam++")

        if cam is not None:
            # Denormalize image for heatmap processing
            img_t = x_in[0].detach().cpu().numpy()  # [3,224,224]
            img_t = (img_t * 0.5) + 0.5
            img_t = np.clip(img_t, 0.0, 1.0)
            img_np = np.transpose(img_t, (1, 2, 0))  # [224,224,3]
            
            # Generate overlay with TURBO colormap (medical standard)
            overlay_b64, boxes = _cam_overlay_to_base64(
                cam, x_in, alpha=0.6, draw_boxes=is_cnn
            )
            
            # Generate heatmap-only visualization (TURBO colormap, with masking)
            heatmap = _generate_heatmap_only_base64(cam, img=img_np)
            
            gradcam_data = {
                "overlay": overlay_b64,  # Single overlay with TURBO colormap
                "heatmap": heatmap,  # Heatmap-only (no overlay)
                "method": method_name,
                "colormap": "turbo",  # Always TURBO (medical standard)
                "alpha": 0.6,
            }
            
            # Add bounding boxes for CNN models only
            if is_cnn:
                gradcam_data["bounding_boxes"] = boxes
                gradcam_data["note"] = "Highlighted regions indicate model attention — not clinical diagnosis."
            # ViT: gradcam_data stays None (no GradCAM for ViT)
    except Exception:
        # If GradCAM fails we still return predictions; gradcam stays None
        gradcam_data = None

    return grade, confidence, probs, gradcam_data


async def _predict_one_safe(name: str, model: nn.Module, x: torch.Tensor) -> Tuple[str, Dict[str, Any]]:
    """
    Per-model isolation: if one model fails, return error for that model only.
    """
    try:
        # to_thread keeps the event loop responsive while running CPU-bound inference
        grade, confidence, probs, gradcam_data = await asyncio.to_thread(
            predict_with_model, name, model, x
        )
        result = {
            "grade": grade,
            "confidence": confidence,
            "probs": probs,
        }
        if gradcam_data is not None:
            result["gradcam"] = gradcam_data
        return name, result
    except Exception as exc:
        return name, {"grade": None, "confidence": None, "probs": None, "error": str(exc)}


async def run_inference_all(models: Dict[str, nn.Module], x: torch.Tensor) -> Dict[str, Any]:
    """
    Run ALL 5 models "in parallel" and return dict keyed by:
      efficientnet, resnet50, vit, hybrid_effvit, hybrid_resvit
    """
    tasks = [_predict_one_safe(name, model, x) for name, model in models.items()]
    pairs = await asyncio.gather(*tasks)
    return {name: payload for (name, payload) in pairs}

