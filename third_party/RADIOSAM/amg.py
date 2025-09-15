# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore

from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

import argparse
import json
import os
from typing import Any, Dict, List
import torch
from torch import nn
from mobile_sam.modeling.image_encoder import ImageEncoderViT
from einops import rearrange
import inspect  # add for hub compat


parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)
parser.add_argument(
    "--config",
    type=str,
    default="third_party//RADIOSAM/config/eradio_enc_ms_dec_config.yaml",
    help="Path to config file for model and adapter.",
)
parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    default="radio",
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b', 'vit_t', 'radio'].",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)

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
                layers.append(nn.LayerNorm([hidden_dim, 64, 64]))  # 假设 1024x1024，H/16=64
                layers.append(nn.GELU())  # 使用 GELU，与 TinyViT 一致
            elif i == layer_num - 1:
                layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1))
            else:
                layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))
                layers.append(nn.LayerNorm([hidden_dim, 64, 64]))
                layers.append(nn.GELU())

        self.layers = nn.Sequential(*layers)
        self._initialize_weights()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图，形状为 (B, C, H, W)
        Returns:
            输出特征图，形状为 (B, out_channels, H, W)
        """
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

        # 使用高参数量的维度适配器：从1280维降到320维
        if cfg is not None:
            self.dim_adapter = DimAdapter(1280, 320, layer_num=cfg.get('adapter_layer_num', 3))

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]

        if self.dtype is not None:
            x = x.to(dtype=self.dtype)

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.dtype is None):
            output = self.radio(x)
        #我想打印出radio的所有中间结果
        # for key, value in output.items():
        #     print(f"radio中间结果 - {key}: {value.features.shape}")
        #     print(f"radio中间结果 - {key}: {value.summary.shape}")
        features = output["sam"].features

        rows = h // 16
        cols = w // 16

        features = rearrange(features, 'b (h w) c -> b c h w', h=rows, w=cols)
        # 添加维度适配
        adapted_feature = self.dim_adapter(features)

        self.tiny_feat = adapted_feature

        output_features = self.neck(adapted_feature)

        return output_features


def save_annotated_mask(masks: List[Dict[str, Any]], image: Any, save_path: str, iou_thresh: float = 0.8, keep_all: bool = False) -> None:
    """
    按照stability_score排序并进行NMS合并，输出annotated图片。
    """
    import numpy as np
    import random
    # 按stability_score降序排序
    masks_sorted = sorted(masks, key=lambda x: x["stability_score"], reverse=True)
    if keep_all:
        print("keep all!")
        selected_masks = [m["segmentation"] for m in masks_sorted]
    else:
        selected_masks = []
        bboxes = []
        def compute_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[0]+box1[2], box2[0]+box2[2])
            y2 = min(box1[1]+box1[3], box2[1]+box2[3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            area1 = box1[2]*box1[3]
            area2 = box2[2]*box2[3]
            union = area1 + area2 - inter
            return inter / union if union > 0 else 0
        for mask_data in masks_sorted:
            bbox = mask_data["bbox"]
            keep = True
            for b in bboxes:
                if compute_iou(bbox, b) > iou_thresh:
                    keep = False
                    break
            if keep:
                selected_masks.append(mask_data["segmentation"])
                bboxes.append(bbox)
    # RGBA可视化
    h, w = image.shape[:2]
    annotated = np.zeros((h, w, 4), dtype=np.uint8)
    annotated[..., :3] = image  # RGB
    annotated[..., 3] = 255     # Alpha全不透明
    # 预设/随机颜色
    color_list = [
        (255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128), (255, 255, 0, 128),
        (255, 0, 255, 128), (0, 255, 255, 128), (255, 128, 0, 128), (128, 0, 255, 128),
        (0, 128, 255, 128), (128, 255, 0, 128), (255, 0, 128, 128), (0, 255, 128, 128)
    ]
    for idx, m in enumerate(selected_masks):
        color = color_list[idx % len(color_list)] if len(color_list) > 0 else tuple(random.randint(0,255) for _ in range(4))
        mask = m.astype(bool)
        alpha_mask = color[3] / 255.0
        for c in range(3):
            # 叠加颜色到mask区域（alpha混合）
            annotated[..., c][mask] = (
                (1 - alpha_mask) * annotated[..., c][mask] + alpha_mask * color[c]
            ).astype(np.uint8)
        # alpha通道也做混合，增强透明度效果
        annotated[..., 3][mask] = (
            np.maximum(annotated[..., 3][mask], color[3])
        ).astype(np.uint8)
    # 原图与mask透明融合
    rgb = annotated[..., :3].astype(np.float32)
    alpha = annotated[..., 3:4].astype(np.float32) / 255.0
    orig = image.astype(np.float32)
    out = orig * (1 - alpha) + rgb * alpha
    out = out.astype(np.uint8)
    # 拼接alpha通道
    out_rgba = np.concatenate([out, (alpha*255).astype(np.uint8)], axis=-1)
    cv2.imwrite(save_path, cv2.cvtColor(out_rgba, cv2.COLOR_RGBA2BGRA))
    return

def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


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


from omegaconf import OmegaConf
def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)
    print("Loading model...")
    if args.model_type != 'radio':
        print("loading MobileSAM model...")
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    else:
        print("Loading RADIO model...")
        sam = sam_model_registry["vit_t"](checkpoint=args.checkpoint)
        # radio_model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=args.checkpoint, adaptor_names='sam', vitdet_window_size=16)
        #e-radio_v2 c-radio_v3-h
        model_version="e-radio_v2"
        _ensure_hub_weights_only_compat()  # ensure hub compat before loading RADIO
        radio_model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version,adaptor_names=['sam'], progress=True, skip_validation=True)
        sam.image_encoder = RADIOVenc(radio_model, sam.image_encoder,cfg=cfg)

        ckpt= torch.load('/hdd/caoyuhao/VLN_ws/MTU3D/third_party/RADIOSAM/checkpoints/eradio_3/best.pth', map_location="cuda")
        sam.image_encoder.dim_adapter.load_state_dict(
           ckpt['model_state_dict']
        )

        preproc = radio_model.make_preprocessor_external()
        sam.pixel_mean = preproc.norm_mean * 255
        sam.pixel_std = preproc.norm_std * 255

    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    


    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]

    os.makedirs(args.output, exist_ok=True)

    import time
    start_time = time.time()
    print(f"Processing {len(targets)} images...")
    for t in targets:
        print(f"Processing '{t}'...")
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_start = time.time()
        masks = generator.generate(image)
        mask_end = time.time()
        print(f"Mask generation for '{t}' took {mask_end - mask_start:.2f} seconds.")

        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)
        if output_mode == "binary_mask":
            os.makedirs(save_base, exist_ok=True)
            write_masks_to_folder(masks, save_base)
            # 保存annotated图片
            annotated_path = os.path.join(save_base, "annotated.png")
            save_annotated_mask(
                masks, image, annotated_path, iou_thresh=0.8, keep_all=False
            )
        else:
            save_file = save_base + ".json"
            with open(save_file, "w") as f:
                json.dump(masks, f)
    print(f"Done! Total time: {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
