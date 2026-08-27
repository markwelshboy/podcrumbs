#!/usr/bin/env python3
"""
Batch comparator for high-quality open-source background removal/matting.

Methods:
  rmbg2                    - BRIA RMBG-2.0 soft alpha, original RGB
  rmbg2_vitmatte           - RMBG-2.0 -> conservative auto trimap -> ViTMatte-B
  birefnet_hr              - BiRefNet_HR-matting alpha, original RGB
  birefnet_hr_refined      - same alpha + foreground-colour decontamination
  ben2_base                - BEN2 Base, refine_foreground=False
  ben2_refined             - BEN2 Base, refine_foreground=True
  birefnet_vitmatte        - BiRefNet HR -> conservative auto trimap -> ViTMatte-B

The harness saves full-resolution alpha/RGBA results plus visual diagnostics on
checker/white/black/mid-gray backgrounds and identical edge-detail crops.
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps


METHOD_LABELS = {
    "rmbg2": "BRIA RMBG-2.0",
    "rmbg2_vitmatte": "RMBG-2.0 → ViTMatte-B",
    "birefnet_hr": "BiRefNet HR matte",
    "birefnet_hr_refined": "BiRefNet HR + FG refine",
    "ben2_base": "BEN2 Base",
    "ben2_refined": "BEN2 Base + refine",
    "birefnet_vitmatte": "BiRefNet HR → ViTMatte-B",
}


@dataclass
class MethodResult:
    method: str
    ok: bool
    seconds: float = 0.0
    error: str = ""
    foreground_fraction: float = 0.0
    uncertain_fraction: float = 0.0
    mean_alpha: float = 0.0
    alpha_mae_vs_birefnet: Optional[float] = None


# ----------------------------- basic image I/O -----------------------------

def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def save_l_alpha(alpha: np.ndarray, path: Path) -> None:
    arr = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(arr, "L").save(path)


def load_l_alpha(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def save_rgba(rgb: np.ndarray, alpha: np.ndarray, path: Path) -> None:
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    a = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb, a])
    Image.fromarray(rgba, "RGBA").save(path)


def load_rgba_parts(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as im:
        rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    return rgba[..., :3], rgba[..., 3].astype(np.float32) / 255.0


def resize_max_side(im: Image.Image, max_side: int, resample=Image.Resampling.LANCZOS) -> Image.Image:
    w, h = im.size
    if max(w, h) <= max_side:
        return im.copy()
    scale = max_side / float(max(w, h))
    return im.resize((max(1, round(w * scale)), max(1, round(h * scale))), resample)


def composite_rgb(fg_rgb: np.ndarray, alpha: np.ndarray, bg: Tuple[int, int, int] | np.ndarray) -> np.ndarray:
    fg = fg_rgb.astype(np.float32)
    a = alpha.astype(np.float32)[..., None]
    if isinstance(bg, tuple):
        b = np.empty_like(fg)
        b[...] = np.array(bg, dtype=np.float32)
    else:
        b = bg.astype(np.float32)
    return np.clip(fg * a + b * (1.0 - a) + 0.5, 0, 255).astype(np.uint8)


def make_checkerboard(w: int, h: int, size: int = 32) -> Image.Image:
    yy, xx = np.mgrid[:h, :w]
    cells = ((xx // size) + (yy // size)) & 1
    v = np.where(cells[..., None] == 0, 196, 236).astype(np.uint8)
    rgb = np.repeat(v, 3, axis=2)
    return Image.fromarray(rgb, "RGB")


# ----------------------- foreground colour refinement ----------------------

def _blur_fusion_pass(image: np.ndarray, fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray, radius: int):
    """Approximate foreground/background colour estimation using local blur fusion."""
    a3 = alpha[..., None]
    k = max(1, int(radius))
    blurred_a = cv2.blur(alpha, (k, k))[..., None]
    blurred_fga = cv2.blur(fg * a3, (k, k))
    blurred_fg = blurred_fga / (blurred_a + 1e-5)
    blurred_bga = cv2.blur(bg * (1.0 - a3), (k, k))
    blurred_bg = blurred_bga / ((1.0 - blurred_a) + 1e-5)
    out_fg = blurred_fg + a3 * (image - a3 * blurred_fg - (1.0 - a3) * blurred_bg)
    return np.clip(out_fg, 0.0, 1.0), np.clip(blurred_bg, 0.0, 1.0)


def foreground_estimate_blur_fusion(rgb_u8: np.ndarray, alpha: np.ndarray, large_blur: int = 90, small_blur: int = 6) -> np.ndarray:
    """Estimate uncontaminated foreground RGB from source RGB + alpha matte.

    Alpha is intentionally unchanged; this isolates the effect of foreground
    colour estimation from the effect of a different matte.
    """
    image = rgb_u8.astype(np.float32) / 255.0
    fg1, bg1 = _blur_fusion_pass(image, image, image, alpha, large_blur)
    fg2, _ = _blur_fusion_pass(image, fg1, bg1, alpha, small_blur)
    return np.clip(fg2 * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------- auto trimap ---------------------------------

def _odd_kernel(px: int) -> np.ndarray:
    px = max(1, int(px))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))


def build_trimap(
    alpha: np.ndarray,
    sure_foreground_threshold: float,
    possible_foreground_threshold: float,
    erode_px: int,
    dilate_px: int,
) -> np.ndarray:
    """Create a deliberately conservative trimap from a soft source matte.

    White = definitely foreground, black = definitely background, gray = let
    ViTMatte reconsider it. Low-confidence hair strands are kept in unknown
    territory rather than declared background.
    """
    sure_fg = (alpha >= sure_foreground_threshold).astype(np.uint8)
    possible = (alpha >= possible_foreground_threshold).astype(np.uint8)
    if erode_px > 0:
        sure_fg = cv2.erode(sure_fg, _odd_kernel(erode_px), iterations=1)
    if dilate_px > 0:
        possible = cv2.dilate(possible, _odd_kernel(dilate_px), iterations=1)

    trimap = np.zeros(alpha.shape, dtype=np.uint8)
    trimap[possible > 0] = 128
    trimap[sure_fg > 0] = 255
    return trimap


# ----------------------------- diagnostics --------------------------------

def alpha_stats(alpha: np.ndarray) -> Dict[str, float]:
    return {
        "foreground_fraction": float(np.mean(alpha >= 0.5)),
        "uncertain_fraction": float(np.mean((alpha > 0.02) & (alpha < 0.98))),
        "mean_alpha": float(np.mean(alpha)),
    }


def _font(size: int = 22):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def _fit_panel(im: Image.Image, box: Tuple[int, int], bg=(40, 40, 40)) -> Image.Image:
    panel = Image.new("RGB", box, bg)
    t = im.copy()
    t.thumbnail(box, Image.Resampling.LANCZOS)
    panel.paste(t, ((box[0]-t.width)//2, (box[1]-t.height)//2))
    return panel


def make_method_preview(rgb: np.ndarray, alpha: np.ndarray, out_path: Path, max_side: int, checker_size: int, midgray: int, label: str):
    h, w = alpha.shape
    checker = np.asarray(make_checkerboard(w, h, checker_size), dtype=np.uint8)
    variants = [
        ("Checker", composite_rgb(rgb, alpha, checker)),
        ("White", composite_rgb(rgb, alpha, (255, 255, 255))),
        ("Black", composite_rgb(rgb, alpha, (0, 0, 0))),
        ("Mid-gray", composite_rgb(rgb, alpha, (midgray, midgray, midgray))),
    ]
    alpha_rgb = np.repeat((alpha[..., None] * 255).astype(np.uint8), 3, axis=2)
    variants.append(("Alpha", alpha_rgb))

    panel_w = max(320, max_side // 2)
    panel_h = max(320, int(panel_w * h / max(w, 1)))
    panel_h = min(panel_h, max_side // 2 + 120)
    title_h = 38
    canvas = Image.new("RGB", (panel_w * 3, (panel_h + title_h) * 2 + 44), (28, 28, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), label, fill="white", font=_font(24))
    y0 = 44
    for i, (name, arr) in enumerate(variants):
        r, c = divmod(i, 3)
        x = c * panel_w
        y = y0 + r * (panel_h + title_h)
        p = _fit_panel(Image.fromarray(arr, "RGB"), (panel_w, panel_h))
        canvas.paste(p, (x, y + title_h))
        ImageDraw.Draw(canvas).text((x + 8, y + 7), name, fill="white", font=_font(20))
    canvas.save(out_path, quality=92)


def common_detail_boxes(alpha: np.ndarray, crop_px: int = 640) -> List[Tuple[int, int, int, int]]:
    """Choose common high-resolution silhouette detail crops from a reference matte."""
    h, w = alpha.shape
    fg = alpha >= 0.1
    ys, xs = np.where(fg)
    if len(xs) == 0:
        return [(max(0, (w-crop_px)//2), max(0, (h-crop_px)//2), min(w, (w+crop_px)//2), min(h, (h+crop_px)//2))]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    size = min(crop_px, w, h)
    # Top is deliberately first: for portraits this is usually the hair test.
    pts = [
        ((x0+x1)//2, y0 + size//5),
        (x0 + size//5, (y0+y1)//2),
        (x1 - size//5, (y0+y1)//2),
        ((x0+x1)//2, y1 - size//5),
    ]
    boxes = []
    for cx, cy in pts:
        lx = int(np.clip(cx - size//2, 0, max(0, w-size)))
        ty = int(np.clip(cy - size//2, 0, max(0, h-size)))
        boxes.append((lx, ty, lx+size, ty+size))
    return boxes


def make_detail_preview(rgb: np.ndarray, alpha: np.ndarray, boxes: Sequence[Tuple[int,int,int,int]], out_path: Path, label: str):
    tiles = []
    for box in boxes:
        x0, y0, x1, y1 = box
        fg = rgb[y0:y1, x0:x1]
        a = alpha[y0:y1, x0:x1]
        hh, ww = a.shape
        split = np.zeros((hh, ww, 3), dtype=np.uint8)
        split[:, ww//2:] = 255
        comp = composite_rgb(fg, a, split)
        tiles.append(Image.fromarray(comp, "RGB"))
    if not tiles:
        return
    tile = 420
    title_h = 42
    canvas = Image.new("RGB", (tile*2, (tile+title_h)*2 + 44), (28,28,28))
    d = ImageDraw.Draw(canvas)
    d.text((12, 8), label + " — source-resolution edge crops (black | white)", fill="white", font=_font(20))
    for i, im in enumerate(tiles[:4]):
        r, c = divmod(i, 2)
        p = _fit_panel(im, (tile, tile))
        y = 44 + r*(tile+title_h)
        x = c*tile
        canvas.paste(p, (x, y+title_h))
        ImageDraw.Draw(canvas).text((x+8, y+8), ["Top/hair", "Left edge", "Right edge", "Bottom edge"][i], fill="white", font=_font(18))
    canvas.save(out_path, quality=94)


# ----------------------------- model stages -------------------------------

def torch_cleanup():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def load_rmbg2(cfg: dict):
    import torch
    from transformers import AutoModelForImageSegmentation
    model_id = cfg["rmbg2"]["model"]
    device = cfg["runtime"].get("device", "cuda")
    dtype_name = cfg["runtime"].get("dtype", "float16")
    dtype = getattr(torch, dtype_name)
    model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(device=device, dtype=dtype).eval()
    return model, device, dtype


def rmbg2_alpha(model, image: Image.Image, cfg: dict, device: str, dtype) -> np.ndarray:
    import torch
    size = int(cfg["rmbg2"].get("input_size", 1024))
    resized = image.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.inference_mode():
        pred = model(x)[-1].sigmoid()[0].squeeze().float().cpu().numpy()
    alpha_small = Image.fromarray(np.clip(pred * 255 + 0.5, 0, 255).astype(np.uint8), "L")
    alpha = np.asarray(alpha_small.resize(image.size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    return alpha


def load_birefnet(cfg: dict):
    import torch
    from transformers import AutoModelForImageSegmentation
    model_id = cfg["birefnet"]["model"]
    device = cfg["runtime"].get("device", "cuda")
    dtype_name = cfg["runtime"].get("dtype", "float16")
    dtype = getattr(torch, dtype_name)
    model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(device=device, dtype=dtype).eval()
    return model, device, dtype


def birefnet_alpha(model, image: Image.Image, cfg: dict, device: str, dtype) -> np.ndarray:
    import torch
    size = int(cfg["birefnet"].get("input_size", 2048))
    resized = image.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    x = torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.inference_mode():
        pred = model(x)[-1].sigmoid()[0].squeeze().float().cpu().numpy()
    alpha_small = Image.fromarray(np.clip(pred*255+0.5,0,255).astype(np.uint8), "L")
    alpha = np.asarray(alpha_small.resize(image.size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    return alpha


def load_vitmatte(cfg: dict):
    import torch
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor
    mid = cfg["vitmatte"]["model"]
    device = cfg["runtime"].get("device", "cuda")
    dtype_name = cfg["runtime"].get("dtype", "float16")
    dtype = getattr(torch, dtype_name)
    processor = VitMatteImageProcessor.from_pretrained(mid)
    model = VitMatteForImageMatting.from_pretrained(mid).to(device=device, dtype=dtype).eval()
    return processor, model, device


def vitmatte_alpha(processor, model, device: str, image: Image.Image, base_alpha: np.ndarray, cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    import torch
    vc = cfg["vitmatte"]
    max_side = int(vc.get("max_long_side", 2048))
    work = resize_max_side(image, max_side)
    ww, wh = work.size
    base_pil = Image.fromarray(np.clip(base_alpha*255+0.5,0,255).astype(np.uint8), "L").resize((ww, wh), Image.Resampling.LANCZOS)
    a = np.asarray(base_pil, dtype=np.float32) / 255.0
    scale = max(ww, wh) / 2048.0
    erode_px = max(1, round(float(vc.get("erode_px_at_2048", 10)) * scale))
    dilate_px = max(1, round(float(vc.get("dilate_px_at_2048", 22)) * scale))
    trimap = build_trimap(
        a,
        float(vc.get("sure_foreground_threshold", 0.90)),
        float(vc.get("possible_foreground_threshold", 0.03)),
        erode_px,
        dilate_px,
    )
    trimap_pil = Image.fromarray(trimap, "L")
    inputs = processor(images=work, trimaps=trimap_pil, return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=(model.dtype if v.is_floating_point() else v.dtype)) for k,v in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs).alphas[0,0].float().cpu().numpy()
    out = np.clip(out, 0.0, 1.0)
    out_pil = Image.fromarray(np.clip(out*255+0.5,0,255).astype(np.uint8), "L").resize(image.size, Image.Resampling.LANCZOS)
    alpha = np.asarray(out_pil, dtype=np.float32) / 255.0
    trimap_full = np.asarray(trimap_pil.resize(image.size, Image.Resampling.NEAREST), dtype=np.uint8)
    return alpha, trimap_full


def load_ben2(cfg: dict):
    import torch
    from ben2 import AutoModel
    device = torch.device(cfg["runtime"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(cfg["ben2"]["model"])
    model.to(device).eval()
    return model


def ben2_infer(model, image: Image.Image, refine: bool) -> Tuple[np.ndarray, np.ndarray]:
    # Current official package exposes refine_foreground on inference.
    try:
        out = model.inference(image, refine_foreground=refine)
    except TypeError:
        if refine:
            raise RuntimeError("Installed BEN2 package does not expose refine_foreground on AutoModel.inference")
        out = model.inference(image)
    if isinstance(out, list):
        out = out[0]
    rgba = np.asarray(out.convert("RGBA"), dtype=np.uint8)
    if (rgba.shape[1], rgba.shape[0]) != image.size:
        out = out.convert("RGBA").resize(image.size, Image.Resampling.LANCZOS)
        rgba = np.asarray(out, dtype=np.uint8)
    return rgba[..., :3], rgba[..., 3].astype(np.float32) / 255.0


# ---------------------------- batch orchestration --------------------------

def discover_images(root: Path, extensions: Sequence[str], recursive: bool) -> List[Path]:
    extset = {x.lower() for x in extensions}
    it = root.rglob("*") if recursive else root.glob("*")
    return sorted([p for p in it if p.is_file() and p.suffix.lower() in extset])


def safe_key(path: Path, input_root: Path) -> str:
    rel = path.relative_to(input_root)
    stem = "__".join(rel.with_suffix("").parts)
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_result_artifacts(image: Image.Image, out_dir: Path, method: str, rgb: np.ndarray, alpha: np.ndarray, cfg: dict, detail_boxes, base_alpha: Optional[np.ndarray]):
    md = out_dir / method
    md.mkdir(parents=True, exist_ok=True)
    save_l_alpha(alpha, md / "alpha.png")
    save_rgba(rgb, alpha, md / "foreground.png")
    make_method_preview(rgb, alpha, md / "preview.jpg", int(cfg["preview"].get("max_side",900)), int(cfg["preview"].get("checker_size",32)), int(cfg["preview"].get("midgray",128)), METHOD_LABELS.get(method,method))
    make_detail_preview(rgb, alpha, detail_boxes, md / "details.jpg", METHOD_LABELS.get(method,method))


def run_rmbg2_stage(images, input_root, output_root, cfg, states):
    methods = set(cfg["methods"])
    if not ({"rmbg2", "rmbg2_vitmatte"} & methods):
        return
    print("\n== Loading BRIA RMBG-2.0 ==")
    model = device = dtype = None
    try:
        model, device, dtype = load_rmbg2(cfg)
        for idx, path in enumerate(images, 1):
            key = safe_key(path, input_root)
            od = output_root / key
            od.mkdir(parents=True, exist_ok=True)
            try:
                image = open_rgb(path)
                rgb = np.asarray(image, dtype=np.uint8)
                t0 = time.perf_counter()
                alpha = rmbg2_alpha(model, image, cfg, device, dtype)
                secs = time.perf_counter() - t0
                base_path = od / "rmbg2_base_alpha.png"
                save_l_alpha(alpha, base_path)
                states[key]["rmbg2_alpha"] = str(base_path)
                boxes = states[key].get("detail_boxes") or common_detail_boxes(alpha, crop_px=min(700, image.width, image.height))
                if not states[key].get("detail_boxes"):
                    states[key]["detail_boxes"] = boxes
                    write_json(od / "detail_boxes.json", boxes)
                if "rmbg2" in methods:
                    biref = load_l_alpha(Path(states[key]["base_alpha"])) if states[key].get("base_alpha") else None
                    save_result_artifacts(image, od, "rmbg2", rgb, alpha, cfg, boxes, biref)
                    st = alpha_stats(alpha)
                    mae = float(np.mean(np.abs(alpha - biref))) if biref is not None else None
                    states[key]["results"]["rmbg2"] = MethodResult("rmbg2", True, secs, alpha_mae_vs_birefnet=mae, **st)
                print(f"[{idx}/{len(images)}] RMBG-2.0: {path.name} ({secs:.2f}s)")
            except Exception as e:
                print(f"ERROR RMBG-2.0 {path}: {e}", file=sys.stderr)
                for m in ["rmbg2", "rmbg2_vitmatte"]:
                    if m in methods:
                        states[key]["results"].setdefault(m, MethodResult(m, False, error=str(e)))
    except Exception as e:
        print(f"FATAL RMBG-2.0 load: {e}", file=sys.stderr)
        for path in images:
            key = safe_key(path, input_root)
            for m in ["rmbg2", "rmbg2_vitmatte"]:
                if m in methods:
                    states[key]["results"].setdefault(m, MethodResult(m, False, error=f"RMBG-2.0 load: {e}"))
    finally:
        del model
        torch_cleanup()


def run_birefnet_stage(images, input_root, output_root, cfg, states):
    methods = set(cfg["methods"])
    if not ({"birefnet_hr", "birefnet_hr_refined", "birefnet_vitmatte"} & methods):
        return
    print("\n== Loading BiRefNet HR-matting ==")
    model = device = dtype = None
    try:
        model, device, dtype = load_birefnet(cfg)
        for idx, path in enumerate(images, 1):
            key = safe_key(path, input_root)
            od = output_root / key
            od.mkdir(parents=True, exist_ok=True)
            try:
                image = open_rgb(path)
                rgb = np.asarray(image, dtype=np.uint8)
                t0 = time.perf_counter()
                alpha = birefnet_alpha(model, image, cfg, device, dtype)
                secs = time.perf_counter()-t0
                save_l_alpha(alpha, od / "birefnet_base_alpha.png")
                boxes = common_detail_boxes(alpha, crop_px=min(700, image.width, image.height))
                write_json(od / "detail_boxes.json", boxes)
                states[key]["base_alpha"] = str(od / "birefnet_base_alpha.png")
                states[key]["detail_boxes"] = boxes
                if "birefnet_hr" in methods:
                    save_result_artifacts(image, od, "birefnet_hr", rgb, alpha, cfg, boxes, alpha)
                    st = alpha_stats(alpha)
                    states[key]["results"]["birefnet_hr"] = MethodResult("birefnet_hr", True, secs, **st)
                if "birefnet_hr_refined" in methods:
                    t1 = time.perf_counter()
                    fr = cfg["foreground_refine"]
                    fg = foreground_estimate_blur_fusion(rgb, alpha, int(fr.get("large_blur",90)), int(fr.get("small_blur",6)))
                    rsecs = time.perf_counter()-t1
                    save_result_artifacts(image, od, "birefnet_hr_refined", fg, alpha, cfg, boxes, alpha)
                    st = alpha_stats(alpha)
                    states[key]["results"]["birefnet_hr_refined"] = MethodResult("birefnet_hr_refined", True, secs+rsecs, alpha_mae_vs_birefnet=0.0, **st)
                print(f"[{idx}/{len(images)}] BiRefNet: {path.name} ({secs:.2f}s)")
            except Exception as e:
                print(f"ERROR BiRefNet {path}: {e}", file=sys.stderr)
                for m in ["birefnet_hr","birefnet_hr_refined"]:
                    if m in methods:
                        states[key]["results"].setdefault(m, MethodResult(m, False, error=str(e)))
    except Exception as e:
        print(f"FATAL BiRefNet load: {e}", file=sys.stderr)
        for path in images:
            key = safe_key(path,input_root)
            for m in ["birefnet_hr","birefnet_hr_refined","birefnet_vitmatte"]:
                if m in methods:
                    states[key]["results"].setdefault(m, MethodResult(m, False, error=f"BiRefNet load: {e}"))
    finally:
        del model
        torch_cleanup()


def run_vitmatte_stage(images, input_root, output_root, cfg, states):
    methods = set(cfg["methods"])
    wanted = [m for m in ["birefnet_vitmatte", "rmbg2_vitmatte"] if m in methods]
    if not wanted:
        return
    print("\n== Loading ViTMatte-B ==")
    processor = model = device = None
    try:
        processor, model, device = load_vitmatte(cfg)
        for idx, path in enumerate(images, 1):
            key = safe_key(path, input_root)
            od = output_root / key
            image = open_rgb(path)
            rgb = np.asarray(image, dtype=np.uint8)
            biref = load_l_alpha(Path(states[key]["base_alpha"])) if states[key].get("base_alpha") else None
            sources = [
                ("birefnet_vitmatte", states[key].get("base_alpha"), "BiRefNet"),
                ("rmbg2_vitmatte", states[key].get("rmbg2_alpha"), "RMBG-2.0"),
            ]
            for method, base_path, source_label in sources:
                if method not in methods:
                    continue
                if not base_path:
                    states[key]["results"].setdefault(method, MethodResult(method, False, error=f"{source_label} base alpha unavailable"))
                    continue
                try:
                    base_alpha = load_l_alpha(Path(base_path))
                    boxes = states[key].get("detail_boxes") or common_detail_boxes(base_alpha)
                    t0 = time.perf_counter()
                    alpha, trimap = vitmatte_alpha(processor, model, device, image, base_alpha, cfg)
                    secs = time.perf_counter() - t0
                    md = od / method
                    md.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(trimap, "L").save(md / "trimap.png")
                    save_result_artifacts(image, od, method, rgb, alpha, cfg, boxes, biref)
                    st = alpha_stats(alpha)
                    mae = float(np.mean(np.abs(alpha - biref))) if biref is not None else None
                    states[key]["results"][method] = MethodResult(method, True, secs, alpha_mae_vs_birefnet=mae, **st)
                    source_delta = float(np.mean(np.abs(alpha - base_alpha)))
                    print(f"[{idx}/{len(images)}] {METHOD_LABELS[method]}: {path.name} ({secs:.2f}s, delta vs source matte {source_delta:.4f})")
                except Exception as e:
                    print(f"ERROR {METHOD_LABELS[method]} {path}: {e}", file=sys.stderr)
                    states[key]["results"][method] = MethodResult(method, False, error=str(e))
    except Exception as e:
        print(f"FATAL ViTMatte load: {e}", file=sys.stderr)
        for path in images:
            key = safe_key(path, input_root)
            for method in wanted:
                states[key]["results"][method] = MethodResult(method, False, error=f"ViTMatte load: {e}")
    finally:
        del processor, model
        torch_cleanup()


def run_ben2_stage(images,input_root,output_root,cfg,states):
    methods=set(cfg["methods"])
    wanted=[m for m in ["ben2_base","ben2_refined"] if m in methods]
    if not wanted:
        return
    print("\n== Loading BEN2 Base ==")
    model=None
    try:
        model=load_ben2(cfg)
        for idx,path in enumerate(images,1):
            key=safe_key(path,input_root)
            od=output_root/key
            try:
                image=open_rgb(path)
                base_alpha=load_l_alpha(Path(states[key]["base_alpha"])) if states[key].get("base_alpha") else None
                boxes=states[key].get("detail_boxes") or (common_detail_boxes(base_alpha) if base_alpha is not None else [(0,0,min(700,image.width),min(700,image.height))])
                for m, refine in [("ben2_base",False),("ben2_refined",True)]:
                    if m not in methods:
                        continue
                    t0=time.perf_counter()
                    rgb,alpha=ben2_infer(model,image,refine)
                    secs=time.perf_counter()-t0
                    save_result_artifacts(image,od,m,rgb,alpha,cfg,boxes,base_alpha)
                    st=alpha_stats(alpha)
                    mae=float(np.mean(np.abs(alpha-base_alpha))) if base_alpha is not None else None
                    states[key]["results"][m]=MethodResult(m,True,secs,alpha_mae_vs_birefnet=mae,**st)
                    print(f"[{idx}/{len(images)}] {METHOD_LABELS[m]}: {path.name} ({secs:.2f}s)")
            except Exception as e:
                print(f"ERROR BEN2 {path}: {e}",file=sys.stderr)
                for m in wanted:
                    states[key]["results"].setdefault(m,MethodResult(m,False,error=str(e)))
    except Exception as e:
        print(f"FATAL BEN2 load: {e}",file=sys.stderr)
        for path in images:
            key=safe_key(path,input_root)
            for m in wanted:
                states[key]["results"][m]=MethodResult(m,False,error=f"BEN2 load: {e}")
    finally:
        del model
        torch_cleanup()


def create_original_previews(images,input_root,output_root,cfg,states):
    for path in images:
        key=safe_key(path,input_root)
        od=output_root/key
        od.mkdir(parents=True,exist_ok=True)
        im=open_rgb(path)
        prev=resize_max_side(im,int(cfg["preview"].get("max_side",900)))
        prev.save(od/"original_preview.jpg",quality=92)
        states[key]["source"] = str(path)
        states[key]["size"] = [im.width,im.height]


def build_report(images,input_root,output_root,cfg,states):
    methods=list(cfg["methods"])
    # CSV
    with (output_root/"comparison.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["image","method","ok","seconds","foreground_fraction","uncertain_fraction","mean_alpha","alpha_mae_vs_birefnet","error"])
        for path in images:
            key=safe_key(path,input_root)
            for m in methods:
                r=states[key]["results"].get(m,MethodResult(m,False,error="not run"))
                w.writerow([str(path.relative_to(input_root)),m,r.ok,f"{r.seconds:.4f}",f"{r.foreground_fraction:.6f}",f"{r.uncertain_fraction:.6f}",f"{r.mean_alpha:.6f}","" if r.alpha_mae_vs_birefnet is None else f"{r.alpha_mae_vs_birefnet:.6f}",r.error])

    cards=[]
    for path in images:
        key=safe_key(path,input_root)
        state=states[key]
        cells=[]
        for m in methods:
            r=state["results"].get(m,MethodResult(m,False,error="not run"))
            if r.ok:
                mae="—" if r.alpha_mae_vs_birefnet is None else f"{r.alpha_mae_vs_birefnet:.4f}"
                cells.append(f'''<div class="method"><h3>{html.escape(METHOD_LABELS.get(m,m))}</h3>
<img loading="lazy" src="{key}/{m}/preview.jpg"><img loading="lazy" src="{key}/{m}/details.jpg">
<div class="stats">{r.seconds:.2f}s · uncertain α {100*r.uncertain_fraction:.2f}% · α MAE vs BiRefNet {mae}</div>
<div class="links"><a href="{key}/{m}/foreground.png">RGBA</a> · <a href="{key}/{m}/alpha.png">alpha</a>{' · <a href="'+key+'/'+m+'/trimap.png">trimap</a>' if m.endswith('_vitmatte') else ''}</div></div>''')
            else:
                cells.append(f'''<div class="method fail"><h3>{html.escape(METHOD_LABELS.get(m,m))}</h3><pre>{html.escape(r.error)}</pre></div>''')
        cards.append(f'''<section><h2>{html.escape(str(path.relative_to(input_root)))}</h2><div class="source"><img src="{key}/original_preview.jpg"><div>Source: {state['size'][0]}×{state['size'][1]}</div></div><div class="grid">{''.join(cells)}</div></section>''')

    doc=f'''<!doctype html><html><head><meta charset="utf-8"><title>Background-removal comparison</title><style>
body{{background:#151515;color:#eee;font:15px system-ui,sans-serif;margin:24px}} a{{color:#91c8ff}} section{{border-top:1px solid #444;padding-top:24px;margin-top:30px}} .source img{{max-width:520px;max-height:520px}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:22px;margin-top:18px}} .method{{background:#222;padding:12px;border-radius:10px}} .method img{{display:block;width:100%;margin:8px 0;background:#111}} .stats{{font-family:ui-monospace,monospace;color:#bbb}} .fail{{border:1px solid #a44}} pre{{white-space:pre-wrap}} h3{{margin:2px 0 8px}}
</style></head><body><h1>Background-removal comparison</h1><p>Each method shows checker/white/black/mid-gray/alpha plus identical source-resolution silhouette detail crops. The detail crops use a black|white split background to expose light/dark halos. “Uncertain α” is the fraction of pixels with 0.02&lt;α&lt;0.98; it is descriptive, not a quality score.</p>{''.join(cards)}</body></html>'''
    (output_root/"report.html").write_text(doc,encoding="utf-8")

    serial={}
    for k,v in states.items():
        serial[k]={**{kk:vv for kk,vv in v.items() if kk!="results"},"results":{m:asdict(r) for m,r in v["results"].items()}}
    write_json(output_root/"metadata.json",serial)


def load_config(path: Path) -> dict:
    with path.open("r",encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap=argparse.ArgumentParser(description="Compare open-source background-removal/matting pipelines on a batch of images.")
    ap.add_argument("input",type=Path,help="Input image file or directory")
    ap.add_argument("output",type=Path,help="Output directory")
    ap.add_argument("--config",type=Path,default=Path(__file__).with_name("config.yaml"))
    ap.add_argument("--methods",nargs="+",choices=sorted(METHOD_LABELS),help="Override methods in config")
    ap.add_argument("--limit",type=int,default=0,help="Only process first N images")
    args=ap.parse_args()

    cfg=load_config(args.config)
    if args.methods:
        cfg["methods"]=args.methods
    args.output.mkdir(parents=True,exist_ok=True)

    if args.input.is_file():
        input_root=args.input.parent
        images=[args.input]
    else:
        input_root=args.input
        images=discover_images(input_root,cfg["input"]["extensions"],bool(cfg["input"].get("recursive",True)))
    if args.limit:
        images=images[:args.limit]
    if not images:
        raise SystemExit("No matching images found")

    states={safe_key(p,input_root):{"results":{}} for p in images}
    create_original_previews(images,input_root,args.output,cfg,states)

    print(f"Images: {len(images)}")
    print("Methods:",", ".join(cfg["methods"]))
    run_birefnet_stage(images,input_root,args.output,cfg,states)
    run_rmbg2_stage(images,input_root,args.output,cfg,states)
    run_vitmatte_stage(images,input_root,args.output,cfg,states)
    run_ben2_stage(images,input_root,args.output,cfg,states)
    build_report(images,input_root,args.output,cfg,states)
    print(f"\nDone. Open: {args.output/'report.html'}")
    print(f"CSV: {args.output/'comparison.csv'}")

if __name__=="__main__":
    main()
