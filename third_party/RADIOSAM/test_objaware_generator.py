
"""
Simple test script for ObjAwareMaskGenerator.

Example:
  python third_party/RADIOSAM/test_objaware_generator.py \
    --image /hdd/caoyuhao/VLN_ws/MTU3D/test_img.png \
    --checkpoint /hdd/caoyuhao/VLN_ws/MTU3D/third_party/RADIOSAM/mobile_sam.pt \
    --obj_model /hdd/caoyuhao/VLN_ws/MobileSAM/MobileSAMv2/PromptGuidedDecoder/ObjectAwareModel.pt \
    --prompt_decoder /hdd/caoyuhao/VLN_ws/MobileSAM/MobileSAMv2/PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt \
    --mobilesamv2_path /hdd/caoyuhao/VLN_ws/MobileSAM/MobileSAMv2 \
    --out ./
"""
import argparse
import os
import sys
from typing import List

import cv2
import numpy as np
import torch

# Ensure repo root on sys.path
# REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# if REPO_ROOT not in sys.path:
#     sys.path.append(REPO_ROOT)

# from third_party.RADIOSAM.RADIOSAM import ObjAwareMaskGenerator  # noqa: E402

sys.path.append('/hdd/caoyuhao/VLN_ws/MTU3D/third_party')
sys.path.append('/hdd/caoyuhao/VLN_ws/MTU3D/third_party/RADIOSAM')
from RADIOSAM import ObjAwareMaskGenerator  # noqa: E402


def draw_masks_transparent(image_rgb: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    if len(masks) == 0:
        return image_rgb.copy()
    h, w = image_rgb.shape[:2]
    annotated = np.zeros((h, w, 4), dtype=np.uint8)
    annotated[..., :3] = image_rgb
    annotated[..., 3] = 255
    colors = [
        (255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128), (255, 255, 0, 128),
        (255, 0, 255, 128), (0, 255, 255, 128), (255, 128, 0, 128), (128, 0, 255, 128),
        (0, 128, 255, 128), (128, 255, 0, 128), (255, 0, 128, 128), (0, 255, 128, 128)
    ]
    for i, m in enumerate(masks):
        mask = m.astype(bool)
        c = colors[i % len(colors)]
        alpha = c[3] / 255.0
        for ch in range(3):
            annotated[..., ch][mask] = (
                (1 - alpha) * annotated[..., ch][mask] + alpha * c[ch]
            ).astype(np.uint8)
        annotated[..., 3][mask] = np.maximum(annotated[..., 3][mask], c[3]).astype(np.uint8)
    # alpha blend to RGB
    rgb = annotated[..., :3].astype(np.float32)
    a = (annotated[..., 3:4].astype(np.float32) / 255.0)
    out = (image_rgb.astype(np.float32) * (1 - a) + rgb * a).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to test image")
    ap.add_argument("--checkpoint", default="/hdd/caoyuhao/VLN_ws/MTU3D/third_party/RADIOSAM/mobile_sam.pt", help="MobileSAM checkpoint path")
    ap.add_argument("--obj_model", default="/hdd/caoyuhao/VLN_ws/MobileSAM/MobileSAMv2/PromptGuidedDecoder/ObjectAwareModel.pt", help="ObjectAwareModel .pt path")
    ap.add_argument("--prompt_decoder", default="/hdd/caoyuhao/VLN_ws/MobileSAM/MobileSAMv2/PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt", help="Prompt_guided_Mask_Decoder .pt path")
    ap.add_argument("--mobilesamv2_path", default="/hdd/caoyuhao/VLN_ws/MobileSAM/MobileSAMv2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--iou", type=float, default=0.9)
    ap.add_argument("--out", default="outputs/objaware_test.png")
    args = ap.parse_args()

    gen = ObjAwareMaskGenerator(
        checkpoint=args.checkpoint,
        obj_model_path=args.obj_model,
        prompt_decoder_path=args.prompt_decoder,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        retina=True,
        mobilesamv2_path=args.mobilesamv2_path,
    )

    # Run generator (FastSAM-like)
    results = gen(args.image)
    assert len(results) == 1, "Expected single image result"
    res = results[0]
    print(f"Detections: {res.boxes.data.shape[0]}")
    if res.boxes.data.numel():
        print("First box (xyxy):", res.boxes.data[0].tolist())
        print("First conf:", float(res.boxes.conf[0].item()))

    # Save annotated image
    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(args.image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    masks_np = [m.numpy() > 0.5 for m in res.masks.data]
    vis = draw_masks_transparent(rgb, masks_np)

    # Resolve output path: if directory or missing extension, default filename
    out_path = args.out
    is_dir_like = out_path.endswith(os.sep) or os.path.isdir(out_path) or os.path.splitext(out_path)[1] == ""
    if is_dir_like:
        out_dir = out_path if (out_path.endswith(os.sep) or os.path.isdir(out_path)) else out_path
        if out_dir == "":
            out_dir = "."
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "objaware_test.png")
    else:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    ok = cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for: {out_path}")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
