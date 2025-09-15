"""
ObjAware-based mask generator wrapper (FastSAM-style) for RADIOSAM.

This module provides a simple callable wrapper that uses MobileSAMv2's
ObjectAwareModel to propose boxes and MobileSAM(v2) predictor to generate
instance masks. The return is compatible with the FastSAM/Ultralytics-like
result object consumed by format_result in data/tools/load_scannet_mv_data_fast.py.

Example:
    from third_party.RADIOSAM.RADIOSAM import ObjAwareMaskGenerator
    gen = ObjAwareMaskGenerator(
        checkpoint="/path/to/mobile_sam_checkpoint.pt",
        obj_model_path="/hdd/caoyuhao/VLN_ws/MobileSAM/MobileSAMv2/PromptGuidedDecoder/ObjectAwareModel.pt",
        prompt_decoder_path="/hdd/caoyuhao/VLN_ws/MobileSAM/MobileSAMv2/PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt",
        device="cuda",
        imgsz=1024,
        conf=0.4,
        iou=0.9,
        retina=True,
        mobilesamv2_path="/hdd/caoyuhao/VLN_ws/MobileSAM/MobileSAMv2"
    )
    results = gen("/path/to/image.jpg")  # list with one YOLO-like result
"""
from __future__ import annotations

import os
import sys
sys.path.append('/hdd/caoyuhao/VLN_ws/MobileSAM/MobileSAMv2')
sys.path.append('/hdd/caoyuhao/VLN_ws/MTU3D/hm3d-online')
sys.path.append('/hdd/caoyuhao/VLN_ws/MTU3D/hm3d-online/FastSAM')

from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import nn
from einops import rearrange
from omegaconf import OmegaConf
from mobile_sam.modeling.image_encoder import ImageEncoderViT
import inspect
#torch fixup
def _ensure_hub_weights_only_compat() -> None:
    """Patch torch.hub.load_state_dict_from_url to ignore 'weights_only' on old PyTorch."""
    try:
        fn = torch.hub.load_state_dict_from_url
    except Exception:
        return
    try:
        sig = inspect.signature(fn)
        if 'weights_only' in sig.parameters:
            return  # already supports
    except Exception:
        pass
    orig_fn = fn

    def patched(url, *args, **kwargs):
        kwargs.pop('weights_only', None)
        return orig_fn(url, *args, **kwargs)

    torch.hub.load_state_dict_from_url = patched
_ensure_hub_weights_only_compat()
# -----------------------------
# Try import MobileSAM and MobileSAMv2
# -----------------------------

def _append_path(p: Optional[str]):
    if p and os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)


class _ImportGuard:
    def __init__(self):
        self.ok = True
        self.err: Optional[Exception] = None

    def fail(self, e: Exception):
        self.ok = False
        self.err = e


def _import_mobilesam_modules(mobilesamv2_path: Optional[str] = None):
    _append_path(mobilesamv2_path)
    try:
        from mobile_sam import sam_model_registry as sam_v1_registry  # type: ignore
        from mobilesamv2 import sam_model_registry as sam_v2_registry  # type: ignore
        from mobilesamv2 import SamPredictor as V2SamPredictor  # type: ignore
        from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel  # type: ignore
        return sam_v1_registry, sam_v2_registry, V2SamPredictor, ObjectAwareModel
    except Exception as e:
        raise ImportError(
            f"Failed to import MobileSAM/MobileSAMv2 modules. Ensure mobilesamv2_path is correct and dependencies installed. Details: {e}"
        )


# -----------------------------
# Lightweight YOLO-like results wrappers
# -----------------------------

class _Boxes:
    def __init__(self, boxes_xyxy: torch.Tensor, conf: torch.Tensor):
        # boxes_xyxy: (N, 4), conf: (N,)
        self.data = boxes_xyxy  # torch.Tensor
        self.conf = conf        # torch.Tensor


class _Masks:
    def __init__(self, masks: List[torch.Tensor]):
        # list of (H, W) float tensors with values in {0., 1.}
        self.data = masks


class _YoloLikeResult:
    def __init__(self, boxes: _Boxes, masks: _Masks):
        self.boxes = boxes
        self.masks = masks


# -----------------------------
# RADIO wrapper encoder and adapter (from my_Inference.py)
# -----------------------------

class DimAdapter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layer_num: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_num = layer_num

        layers = []
        hidden_dim = max(in_channels, out_channels)
        for i in range(layer_num):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                layers.append(nn.LayerNorm([hidden_dim, 64, 64]))
                layers.append(nn.GELU())
            elif i == layer_num - 1:
                layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1))
            else:
                layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))
                layers.append(nn.LayerNorm([hidden_dim, 64, 64]))
                layers.append(nn.GELU())

        self.layers = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class RADIOVenc(nn.Module):
    def __init__(self, radio: nn.Module, img_enc: ImageEncoderViT, img_size: int = 1024,cfg=None):
        super().__init__()
        self.radio = radio
        self.neck = img_enc.neck
        self.img_size = img_size
        self.dtype = radio.input_conditioner.dtype
        self.tiny_feat = None
        self.dim_adapter = None
        self.radio_output=None

        # 使用高参数量的维度适配器：从1280维降到320维
        if cfg is not None:
            self.dim_adapter = DimAdapter(1280, 320, layer_num=cfg.get('adapter_layer_num', 3))

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]

        if self.dtype is not None:
            x = x.to(dtype=self.dtype)

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.dtype is None):
            output = self.radio(x)
        self.radio_output=output
        features = output["sam"].features
        print("output backbone shape:",output["backbone"].features.shape)
        rows = h // 16  
        cols = w // 16

        features = rearrange(features, 'b (h w) c -> b c h w', h=rows, w=cols)
        # 添加维度适配
        adapted_feature = self.dim_adapter(features)

        self.tiny_feat = adapted_feature

        output_features = self.neck(adapted_feature)

        return output_features


# -----------------------------
# Main wrapper
# -----------------------------

class ObjAwareMaskGenerator:
    """FastSAM-style callable mask generator based on MobileSAMv2 ObjectAwareModel.

    It first detects boxes via ObjectAwareModel, then uses MobileSAMv2 SamPredictor
    to generate instance masks for those boxes.
    """

    def __init__(
        self,
        checkpoint: str,
        obj_model_path: str,
        prompt_decoder_path: str,
        device: str = "cuda",
        imgsz: int = 1024,
        conf: float = 0.4,
        iou: float = 0.9,
        retina: bool = True,
        mobilesamv2_path: Optional[str] = None,
        # --- radio specific args ---
        model_type: str = "radio",
        radio_config: Optional[str] = "/hdd/caoyuhao/VLN_ws/MTU3D/third_party/RADIOSAM/config/eradio_enc_ms_dec_config.yaml",
        radio_adapter_ckpt: Optional[str] = "/hdd/caoyuhao/VLN_ws/MTU3D/third_party/RADIOSAM/checkpoints/eradio_3/best.pth",
    ) -> None:
        self.device = device
        self.imgsz = imgsz
        self.det_conf = conf
        self.det_iou = iou
        self.retina = retina

        sam_v1_registry, sam_v2_registry, V2SamPredictor, ObjectAwareModel = _import_mobilesam_modules(mobilesamv2_path)

        # Build MobileSAM base and MobileSAMv2 shell
        base_sam = sam_v1_registry["vit_t"](checkpoint=checkpoint)
        mobile_v2 = sam_v2_registry["vit_h"]()  # backbone shell, will reuse v1 encoder

        if model_type == "radio":
            # Load RADIO model according to config
            cfg = None
            if radio_config and os.path.exists(radio_config):
                try:
                    cfg = OmegaConf.load(radio_config)
                except Exception:
                    cfg = None
            model_version = cfg.get("radio_model_version") if cfg is not None else None
            # Load radio from hub
            radio_model = torch.hub.load(
                'NVlabs/RADIO', 'radio_model', version=model_version,
                adaptor_names='sam', progress=True, skip_validation=True
            )
            # Wrap the SAM v1 encoder with RADIOVenc
            radio_enc = RADIOVenc(radio_model, base_sam.image_encoder, cfg=cfg)
            # Optionally load adapter weights
            if radio_adapter_ckpt and os.path.exists(radio_adapter_ckpt):
                ckpt = torch.load(radio_adapter_ckpt, map_location=(self.device if isinstance(self.device, str) else 'cpu'))
                state = ckpt.get('model_state_dict', ckpt)
                radio_enc.dim_adapter.load_state_dict(state, strict=False)
            # Configure normalization from RADIO preprocessor
            try:
                preproc = radio_model.make_preprocessor_external()
                mobile_v2.pixel_mean = preproc.norm_mean * 255
                mobile_v2.pixel_std = preproc.norm_std * 255
            except Exception:
                pass
            mobile_v2.image_encoder = radio_enc
        else:
            # Default: reuse MobileSAM v1 image encoder directly
            mobile_v2.image_encoder = base_sam.image_encoder
       
        # Load prompt-guided decoder and wire into v2
        PromptGuidedDecoder = sam_v2_registry["PromptGuidedDecoder"](prompt_decoder_path)
        mobile_v2.prompt_encoder = PromptGuidedDecoder["PromtEncoder"]
        mobile_v2.mask_decoder = PromptGuidedDecoder["MaskDecoder"]

        # Build detector
        self.obj_model = ObjectAwareModel(obj_model_path)

        # Predictor
        self.predictor = V2SamPredictor(mobile_v2.to(device=self.device).eval())

    # ---- Internal helpers ----
    @staticmethod
    def _load_image(image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image_or_path, str):
            bgr = cv2.imread(image_or_path)
            if bgr is None:
                raise FileNotFoundError(f"Cannot load image: {image_or_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb
        img = image_or_path
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("image must be HxWx3 RGB array")
        return img

    def _detect_boxes(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        # Run ObjAware detector, returns yolov8-like results list
        results = self.obj_model(
            image,
            device=self.device,
            retina_masks=self.retina,
            imgsz=self.imgsz,
            conf=self.det_conf,
            iou=self.det_iou,
        )
        if not results:
            return torch.zeros((0, 4), device="cpu"), torch.zeros((0,), device="cpu")
        r0 = results[0]
        # Expect .boxes.xyxy and .boxes.conf living on torch device
        boxes_xyxy = r0.boxes.xyxy
        conf = r0.boxes.conf if hasattr(r0.boxes, "conf") else torch.ones((boxes_xyxy.size(0),), device=boxes_xyxy.device)
        return boxes_xyxy, conf

    def _segment_with_sam(self, image: np.ndarray, boxes_xyxy: torch.Tensor) -> List[torch.Tensor]:
        # Set image for predictor and transform boxes
        self.predictor.set_image(image)
        if boxes_xyxy.numel() == 0:
            return []
        if boxes_xyxy.device.type != ("cuda" if torch.cuda.is_available() and isinstance(self.device, str) and self.device.startswith("cuda") else "cpu"):
            boxes_xyxy = boxes_xyxy.to(self.predictor.features.device)
        input_boxes = self.predictor.transform.apply_boxes_torch(boxes_xyxy, self.predictor.original_size)
        # Use predict_torch to get masks; select the best mask per box
        # API: masks, scores, logits = predictor.predict_torch(boxes=..., multimask_output=True)
        masks, scores, _ = self.predictor.predict_torch(
            boxes=input_boxes,
            point_coords=None,
            point_labels=None,
            multimask_output=True,
        )
        # masks: (B, M, H, W), scores: (B, M)
        best_idx = scores.argmax(dim=1)
        out_masks: List[torch.Tensor] = []
        for i in range(masks.shape[0]):
            m = masks[i, best_idx[i]]  # (H, W), bool/float
            out_masks.append(m.float().cpu())
        return out_masks

    # ---- Public APIs ----
    def generate(self, image_or_path: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Return list of dicts with keys: segmentation (np.bool_), bbox (xyxy), predicted_iou (det conf), area."""
        image = self._load_image(image_or_path)
        boxes_xyxy, conf = self._detect_boxes(image)
        masks = self._segment_with_sam(image, boxes_xyxy)
        masks_np = [m.numpy() > 0.5 for m in masks]
        boxes_cpu = boxes_xyxy.detach().cpu() if boxes_xyxy.is_cuda else boxes_xyxy.cpu()
        conf_cpu = conf.detach().cpu() if conf.is_cuda else conf.cpu()
        out: List[Dict[str, Any]] = []
        for i in range(len(masks_np)):
            seg = masks_np[i]
            x0, y0, x1, y1 = boxes_cpu[i].tolist()
            bbox_xywh = [x0, y0, x1 - x0, y1 - y0]
            area = int(seg.sum())
            out.append({
                "id": i,
                "segmentation": seg,
                "bbox": bbox_xywh,
                "predicted_iou": float(conf_cpu[i].item()),
                "stability_score": float(conf_cpu[i].item()),  # reuse det conf
                "area": area,
            })
        return out

    def __call__(
        self,
        image_or_path: Union[str, np.ndarray],
        device: Optional[str] = None,
        retina_masks: Optional[bool] = None,
        imgsz: Optional[int] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        keep_all: bool = False,
    ) -> List['_YoloLikeResult']:
        # Allow per-call override (optional)
        if device:
            self.device = device
        if retina_masks is not None:
            self.retina = retina_masks
        if imgsz is not None:
            self.imgsz = imgsz
        if conf is not None:
            self.det_conf = conf
        if iou is not None:
            self.det_iou = iou

        image = self._load_image(image_or_path)
        boxes_xyxy, det_conf = self._detect_boxes(image)
        masks = self._segment_with_sam(image, boxes_xyxy)

        # Wrap into YOLO-like result
        if len(masks) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            conf_tensor = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes_tensor = boxes_xyxy.detach().cpu().float()
            conf_tensor = det_conf.detach().cpu().float()
        mask_tensors = [m.cpu().float() for m in masks]
        return [_YoloLikeResult(_Boxes(boxes_tensor, conf_tensor), _Masks(mask_tensors))]


__all__ = ["ObjAwareMaskGenerator"]
