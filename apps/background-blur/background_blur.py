#!/usr/bin/env python3
"""
Batch background-depth-blur harness for high-resolution character images.

Pipeline:
  input -> optional FBCNN JPEG restoration -> BiRefNet HR matte
        -> optional ViTMatte-B refinement -> Depth Pro metric depth
        -> subject-free background plate -> depth-aware blur pyramid
        -> original subject recomposited through soft alpha -> PNG output

The expensive neural stages are run model-major so only one major model needs
GPU residency at a time. Results are cached to make preset tuning cheap.
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
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
PRESET_ORDER = ["subtle", "natural", "strong"]


# ---------------------------------------------------------------------------
# Image / filesystem helpers
# ---------------------------------------------------------------------------

def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def save_png(im: Image.Image, path: Path, compress_level: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="PNG", compress_level=int(compress_level))


def save_rgb_array(arr: np.ndarray, path: Path, compress_level: int = 6) -> None:
    save_png(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB"), path, compress_level)


def save_alpha(alpha: np.ndarray, path: Path, compress_level: int = 6) -> None:
    a = np.clip(alpha * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    save_png(Image.fromarray(a, "I;16"), path, compress_level)


def load_alpha(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    return a.astype(np.float32) / 255.0


def discover_images(root: Path, exts: Sequence[str], recursive: bool) -> List[Path]:
    extset = {e.lower() for e in exts}
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in extset)


def safe_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    s = "__".join(rel.parts)
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def torch_cleanup() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def resize_max_side(im: Image.Image, max_side: int) -> Image.Image:
    if max(im.size) <= max_side:
        return im.copy()
    scale = max_side / float(max(im.size))
    return im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.Resampling.LANCZOS)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _font(size=20):
    for f in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(f).exists():
            return ImageFont.truetype(f, size=size)
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# FBCNN JPEG restoration
# ---------------------------------------------------------------------------

def fbcnn_should_run(path: Path, mode: str) -> bool:
    mode = mode.lower()
    if mode == "on":
        return True
    if mode == "off":
        return False
    if mode != "auto":
        raise ValueError(f"Unknown FBCNN mode: {mode}")
    return path.suffix.lower() in {".jpg", ".jpeg"}


def ensure_fbcnn_assets(cfg: dict) -> Tuple[Path, Path]:
    fc = cfg["fbcnn"]
    repo = (ROOT / fc["repo_dir"]).resolve()
    model_path = (ROOT / fc["model_path"]).resolve()
    if not (repo / "models" / "network_fbcnn.py").exists():
        raise RuntimeError(
            f"FBCNN source not found at {repo}. Run ./bootstrap.sh first."
        )
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = fc["model_url"]
        print(f"Downloading FBCNN weights: {url}")
        urllib.request.urlretrieve(url, model_path)
    return repo, model_path


def load_fbcnn(cfg: dict):
    import torch
    repo, model_path = ensure_fbcnn_assets(cfg)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from models.network_fbcnn import FBCNN as FBCNNNet

    device = cfg["runtime"].get("device", "cuda") if torch.cuda.is_available() else "cpu"
    model = FBCNNNet(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode="R")
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, device


def _pad_tensor_multiple(x, multiple: int = 8):
    import torch.nn.functional as F
    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, h, w


def fbcnn_infer_tile(model, device: str, rgb: np.ndarray, qf_setting) -> Tuple[np.ndarray, float]:
    import torch
    x = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
    x, h, w = _pad_tensor_multiple(x, 8)
    qf_input = None
    if qf_setting != "auto":
        qf = float(qf_setting)
        if not 1 <= qf <= 100:
            raise ValueError("FBCNN qf must be auto or 1..100")
        qf_input = torch.tensor([[1.0 - qf / 100.0]], device=device, dtype=x.dtype)
    with torch.inference_mode():
        if qf_input is None:
            y, q = model(x)
        else:
            y, q = model(x, qf_input)
    y = y[..., :h, :w].clamp(0, 1)[0].float().cpu().numpy().transpose(1, 2, 0)
    predicted_qf = float((1.0 - q.float().mean().cpu()).item() * 100.0)
    return np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8), predicted_qf


def _tile_starts(length: int, tile: int, overlap: int) -> List[int]:
    if tile <= 0 or length <= tile:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, max(1, length - tile + 1), stride))
    last = max(0, length - tile)
    if starts[-1] != last:
        starts.append(last)
    return starts


def _tile_weight(h: int, w: int, overlap: int, top: bool, bottom: bool, left: bool, right: bool) -> np.ndarray:
    wy = np.ones(h, np.float32)
    wx = np.ones(w, np.float32)
    ovy = min(overlap, h // 2)
    ovx = min(overlap, w // 2)
    if ovy > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, ovy, dtype=np.float32))
        if not top:
            wy[:ovy] = ramp
        if not bottom:
            wy[-ovy:] = ramp[::-1]
    if ovx > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, ovx, dtype=np.float32))
        if not left:
            wx[:ovx] = ramp
        if not right:
            wx[-ovx:] = ramp[::-1]
    return wy[:, None] * wx[None, :]


def fbcnn_restore(model, device: str, image: Image.Image, cfg: dict) -> Tuple[np.ndarray, float]:
    fc = cfg["fbcnn"]
    arr = np.asarray(image, np.uint8)
    h, w = arr.shape[:2]
    tile = int(fc.get("tile_size", 1024))
    overlap = int(fc.get("tile_overlap", 96))
    qf_setting = fc.get("qf", "auto")
    if tile <= 0 or (h <= tile and w <= tile):
        return fbcnn_infer_tile(model, device, arr, qf_setting)

    ys = _tile_starts(h, tile, overlap)
    xs = _tile_starts(w, tile, overlap)
    acc = np.zeros((h, w, 3), np.float32)
    weights = np.zeros((h, w), np.float32)
    qfs = []
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
            out, qf = fbcnn_infer_tile(model, device, arr[y0:y1, x0:x1], qf_setting)
            qfs.append(qf)
            wt = _tile_weight(
                y1 - y0, x1 - x0, overlap,
                top=(y0 == 0), bottom=(y1 == h), left=(x0 == 0), right=(x1 == w),
            )
            acc[y0:y1, x0:x1] += out.astype(np.float32) * wt[..., None]
            weights[y0:y1, x0:x1] += wt
    result = acc / np.maximum(weights[..., None], 1e-6)
    return np.clip(result + 0.5, 0, 255).astype(np.uint8), float(np.median(qfs))


# ---------------------------------------------------------------------------
# Matting: BiRefNet HR -> optional ViTMatte-B
# ---------------------------------------------------------------------------

def load_birefnet(cfg: dict):
    import torch
    from transformers import AutoModelForImageSegmentation
    device = cfg["runtime"].get("device", "cuda") if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, cfg["runtime"].get("dtype", "float16")) if device != "cpu" else torch.float32
    model = AutoModelForImageSegmentation.from_pretrained(
        cfg["birefnet"]["model"], trust_remote_code=True
    ).to(device=device, dtype=dtype).eval()
    return model, device, dtype


def birefnet_alpha(model, device: str, dtype, image: Image.Image, cfg: dict) -> np.ndarray:
    import torch
    size = int(cfg["birefnet"].get("input_size", 2048))
    resized = image.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(resized, np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.inference_mode():
        pred = model(x)[-1].sigmoid()[0].squeeze().float().cpu().numpy()
    small = Image.fromarray(np.clip(pred * 65535 + 0.5, 0, 65535).astype(np.uint16), "I;16")
    full = small.resize(image.size, Image.Resampling.LANCZOS)
    return np.asarray(full, np.float32) / 65535.0


def _ellipse(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def build_trimap(alpha: np.ndarray, cfg: dict, work_long_side: int) -> np.ndarray:
    vc = cfg["vitmatte"]
    scale = work_long_side / 2048.0
    erode = max(1, round(float(vc.get("erode_px_at_2048", 10)) * scale))
    dilate = max(1, round(float(vc.get("dilate_px_at_2048", 24)) * scale))
    sure = (alpha >= float(vc.get("sure_foreground_threshold", 0.90))).astype(np.uint8)
    possible = (alpha >= float(vc.get("possible_foreground_threshold", 0.03))).astype(np.uint8)
    sure = cv2.erode(sure, _ellipse(erode), iterations=1)
    possible = cv2.dilate(possible, _ellipse(dilate), iterations=1)
    tri = np.zeros(alpha.shape, np.uint8)
    tri[possible > 0] = 128
    tri[sure > 0] = 255
    return tri


def load_vitmatte(cfg: dict):
    import torch
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor
    device = cfg["runtime"].get("device", "cuda") if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, cfg["runtime"].get("dtype", "float16")) if device != "cpu" else torch.float32
    proc = VitMatteImageProcessor.from_pretrained(cfg["vitmatte"]["model"])
    model = VitMatteForImageMatting.from_pretrained(cfg["vitmatte"]["model"]).to(device=device, dtype=dtype).eval()
    return proc, model, device


def vitmatte_refine(proc, model, device: str, image: Image.Image, base_alpha: np.ndarray, cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    import torch
    max_side = int(cfg["vitmatte"].get("max_long_side", 2048))
    work = resize_max_side(image, max_side)
    a_work = np.asarray(
        Image.fromarray(np.clip(base_alpha * 65535 + 0.5, 0, 65535).astype(np.uint16), "I;16")
        .resize(work.size, Image.Resampling.LANCZOS),
        np.float32,
    ) / 65535.0
    trimap = build_trimap(a_work, cfg, max(work.size))
    tri_im = Image.fromarray(trimap, "L")
    inputs = proc(images=work, trimaps=tri_im, return_tensors="pt")
    inputs = {
        k: v.to(device=device, dtype=(model.dtype if v.is_floating_point() else v.dtype))
        for k, v in inputs.items()
    }
    with torch.inference_mode():
        out = model(**inputs).alphas[0, 0].float().cpu().numpy()
    out = np.clip(out, 0, 1)
    full = Image.fromarray(np.clip(out * 65535 + 0.5, 0, 65535).astype(np.uint16), "I;16").resize(image.size, Image.Resampling.LANCZOS)
    tri_full = tri_im.resize(image.size, Image.Resampling.NEAREST)
    return np.asarray(full, np.float32) / 65535.0, np.asarray(tri_full, np.uint8)


# ---------------------------------------------------------------------------
# Depth Pro
# ---------------------------------------------------------------------------

def load_depthpro(cfg: dict):
    import torch
    from transformers import DepthProForDepthEstimation, DepthProImageProcessor
    device = cfg["runtime"].get("device", "cuda") if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, cfg["runtime"].get("dtype", "float16")) if device != "cpu" else torch.float32
    mid = cfg["depth"]["model"]
    proc = DepthProImageProcessor.from_pretrained(mid)
    model = DepthProForDepthEstimation.from_pretrained(
        mid,
        use_fov_model=bool(cfg["depth"].get("use_fov_model", False)),
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval()
    return proc, model, device, dtype


def depthpro_infer(proc, model, device: str, dtype, image: Image.Image) -> np.ndarray:
    import torch
    inputs = proc(images=image, return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=(dtype if v.is_floating_point() else v.dtype)) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    pp = proc.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
    depth = pp[0]["predicted_depth"].float().cpu().numpy().astype(np.float32)
    depth[~np.isfinite(depth)] = np.nan
    return depth


def depth_visualization(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), np.uint8)
    lo, hi = np.nanpercentile(depth[valid], [2, 98])
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    # Near bright, far dark for quick inspection.
    u8 = np.clip((1.0 - norm) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Background plate + depth-of-field blur
# ---------------------------------------------------------------------------

def make_background_plate(rgb: np.ndarray, alpha: np.ndarray, cfg: dict) -> np.ndarray:
    pc = cfg["plate"]
    mask = alpha >= float(pc.get("foreground_threshold", 0.01))
    scale = max(rgb.shape[0], rgb.shape[1]) / 4000.0
    expand = max(0, round(float(pc.get("expand_px_at_4k", 10)) * scale))
    if expand > 0:
        mask = cv2.dilate(mask.astype(np.uint8), _ellipse(expand), iterations=1).astype(bool)
    if not np.any(mask) or np.all(mask):
        return rgb.copy()

    # For every masked pixel, copy the nearest unmasked/background pixel.
    # This is not meant to be a visible inpaint: it only prevents subject colours
    # from bleeding into the blurred plate beneath the final subject composite.
    inds = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    plate = rgb.copy()
    plate[mask] = rgb[inds[0][mask], inds[1][mask]]
    return plate


def subject_focus_depth(depth: np.ndarray, alpha: np.ndarray, cfg: dict) -> Tuple[float, np.ndarray]:
    bc = cfg["blur"]
    core = (alpha >= float(bc.get("subject_core_alpha", 0.95))).astype(np.uint8)
    scale = max(alpha.shape) / 4000.0
    erode = max(1, round(float(bc.get("subject_core_erode_px_at_4k", 24)) * scale))
    if np.any(core):
        core = cv2.erode(core, _ellipse(erode), iterations=1)
    valid = (core > 0) & np.isfinite(depth) & (depth > 1e-4)
    if np.count_nonzero(valid) < 64:
        valid = (alpha >= 0.8) & np.isfinite(depth) & (depth > 1e-4)
    if np.count_nonzero(valid) < 16:
        valid = np.isfinite(depth) & (depth > 1e-4)
    if not np.any(valid):
        raise RuntimeError("Depth map has no valid pixels")
    return float(np.median(depth[valid])), core.astype(np.uint8)


def build_blur_map(depth: np.ndarray, alpha: np.ndarray, preset: dict, cfg: dict) -> Tuple[np.ndarray, float]:
    focus, _ = subject_focus_depth(depth, alpha, cfg)
    valid = np.isfinite(depth) & (depth > 1e-4)
    z = np.where(valid, depth, focus).astype(np.float32)
    inv = 1.0 / np.maximum(z, 1e-4)
    inv_focus = 1.0 / max(focus, 1e-4)
    delta = np.abs(inv - inv_focus)

    bg = valid & (alpha < 0.5)
    vals = delta[bg]
    if vals.size < 64:
        vals = delta[valid]
    denom = float(np.percentile(vals, float(cfg["blur"].get("depth_normalization_percentile", 96.0)))) if vals.size else 1.0
    denom = max(denom, 1e-6)
    normalized = delta / denom

    tol = float(preset.get("focus_tolerance", 0.04))
    amount = np.clip((normalized - tol) / max(1.0 - tol, 1e-6), 0.0, 1.0)
    amount = np.power(amount, float(preset.get("gamma", 1.15)))
    amount *= float(preset.get("strength", 0.72))
    amount = np.clip(amount, 0, 1).astype(np.float32)

    if bool(cfg["blur"].get("protect_subject", True)):
        # Smoothly suppress blur where subject alpha rises instead of a hard edge.
        amount *= np.clip(1.0 - alpha, 0.0, 1.0).astype(np.float32)
    return amount, focus


def uniform_blur_map(alpha: np.ndarray, strength: float) -> np.ndarray:
    return (np.clip(1.0 - alpha, 0, 1) * float(strength)).astype(np.float32)


def variable_gaussian_blur(plate: np.ndarray, blur_map: np.ndarray, max_radius: float, fractions: Sequence[float]) -> np.ndarray:
    """Interpolate among a small Gaussian pyramid without holding every level."""
    fractions = sorted(set(float(x) for x in fractions))
    if not fractions or fractions[0] != 0.0:
        fractions = [0.0] + fractions
    if fractions[-1] != 1.0:
        fractions.append(1.0)
    radii = np.array(fractions, np.float32) * float(max_radius)
    target = blur_map.astype(np.float32) * float(max_radius)
    acc = np.zeros_like(plate, np.float32)
    platef = plate.astype(np.float32)

    for i, r in enumerate(radii):
        if i == 0 or r < 0.35:
            level = platef
        else:
            # OpenCV selects an optimized kernel size from sigma when ksize=(0,0).
            level = cv2.GaussianBlur(plate, (0, 0), sigmaX=float(r), sigmaY=float(r), borderType=cv2.BORDER_REFLECT101).astype(np.float32)

        if i == 0:
            right = radii[1]
            weight = np.clip((right - target) / max(right - r, 1e-6), 0, 1)
        elif i == len(radii) - 1:
            left = radii[i - 1]
            weight = np.clip((target - left) / max(r - left, 1e-6), 0, 1)
        else:
            left, right = radii[i - 1], radii[i + 1]
            w_left = np.clip((target - left) / max(r - left, 1e-6), 0, 1)
            w_right = np.clip((right - target) / max(right - r, 1e-6), 0, 1)
            weight = np.minimum(w_left, w_right)
        acc += level * weight[..., None]

    return np.clip(acc + 0.5, 0, 255).astype(np.uint8)


def composite_subject(original: np.ndarray, blurred_plate: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.clip(alpha, 0, 1).astype(np.float32)[..., None]
    out = original.astype(np.float32) * a + blurred_plate.astype(np.float32) * (1.0 - a)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def preset_max_radius(image: Image.Image, preset: dict) -> float:
    return float(preset.get("max_radius_px_at_4k", 34)) * max(image.size) / 4000.0


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def make_preview(paths: Dict[str, Path], output: Path, long_side: int = 1200) -> None:
    items = []
    for label, path in paths.items():
        if path.exists():
            im = open_rgb(path)
            im = resize_max_side(im, long_side // 2)
            items.append((label, im))
    if not items:
        return
    cols = 2
    cell_w = max(im.width for _, im in items) + 20
    cell_h = max(im.height for _, im in items) + 48
    rows = math.ceil(len(items) / cols)
    canvas = Image.new("RGB", (cell_w * cols, cell_h * rows), (28, 28, 28))
    d = ImageDraw.Draw(canvas)
    for i, (label, im) in enumerate(items):
        r, c = divmod(i, cols)
        x, y = c * cell_w, r * cell_h
        canvas.paste(im, (x + 10, y + 38))
        d.text((x + 10, y + 8), label, fill="white", font=_font(20))
    save_png(canvas, output)


def write_report(output_root: Path, states: Dict[str, dict], presets: Sequence[str], compare: bool) -> None:
    rows = []
    for key, st in states.items():
        rows.append({
            "key": key,
            "source": st["source"],
            "fbcnn": st.get("fbcnn_mode"),
            "fbcnn_applied": st.get("fbcnn_applied", False),
            "fbcnn_qf": st.get("fbcnn_predicted_qf"),
            "matte": st.get("matte_mode"),
            "focus_depth_m": st.get("focus_depth_m"),
            "error": st.get("error", ""),
        })
    with (output_root / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["key"])
        w.writeheader(); w.writerows(rows)

    cards = []
    for key, st in states.items():
        rel = Path(key)
        imgs = [
            ("Input", "source.png"),
            ("Working", "working.png"),
            ("Alpha", "alpha_preview.png"),
            ("Depth", "depth.png"),
        ]
        if compare:
            imgs += [("Uniform", "uniform.png")]
        imgs += [(p.title(), f"depth_{p}.png") for p in presets]
        img_html = "".join(
            f'<div class="tile"><div>{html.escape(label)}</div><a href="{key}/{name}"><img src="{key}/{name}"></a></div>'
            for label, name in imgs if (output_root / key / name).exists()
        )
        err = html.escape(st.get("error", ""))
        meta = f"FBCNN: {st.get('fbcnn_mode')} / applied={st.get('fbcnn_applied')} / QF={st.get('fbcnn_predicted_qf')} &nbsp; Matte: {st.get('matte_mode')} &nbsp; Focus≈{st.get('focus_depth_m')}m"
        cards.append(f'<section><h2>{html.escape(st["source"])}</h2><p>{meta}</p>{"<p class=err>"+err+"</p>" if err else ""}<div class="grid">{img_html}</div></section>')
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>Background blur comparison</title>
<style>body{{font-family:sans-serif;background:#171717;color:#eee;margin:20px}}section{{border-top:1px solid #555;padding:18px 0}}.grid{{display:flex;flex-wrap:wrap;gap:12px}}.tile{{width:280px}}img{{max-width:280px;max-height:360px;object-fit:contain;background:#333}}a{{color:#9cf}}.err{{color:#ff9b9b}}</style></head><body>
<h1>Background blur comparison</h1><p>All final image artifacts are lossless PNG. Click thumbnails for source-resolution files.</p>{''.join(cards)}</body></html>"""
    (output_root / "report.html").write_text(doc, encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def prepare_sources(images: List[Path], input_root: Path, output_root: Path, cfg: dict, states: Dict[str, dict], fbcnn_mode: str, force: bool):
    comp = int(cfg["output"].get("png_compress_level", 6))
    for path in images:
        key = safe_key(path, input_root)
        od = output_root / key
        od.mkdir(parents=True, exist_ok=True)
        source_path = od / "source.png"
        if force or not source_path.exists():
            save_png(open_rgb(path), source_path, comp)
        states[key] = {
            "source": str(path.relative_to(input_root)),
            "original_path": str(path),
            "fbcnn_mode": fbcnn_mode,
            "fbcnn_applied": fbcnn_should_run(path, fbcnn_mode),
            "matte_mode": cfg["matte"].get("mode", "maximum"),
        }


def run_fbcnn_stage(images, input_root, output_root, cfg, states, fbcnn_mode, force):
    comp = int(cfg["output"].get("png_compress_level", 6))
    wanted = [p for p in images if fbcnn_should_run(p, fbcnn_mode)]
    # First establish working.png for images that do not require restoration.
    for p in images:
        key = safe_key(p, input_root); od = output_root / key
        wp = od / "working.png"
        if not fbcnn_should_run(p, fbcnn_mode) and (force or not wp.exists()):
            shutil.copyfile(od / "source.png", wp)
    if not wanted:
        return
    print("\n== FBCNN JPEG artifact restoration ==")
    model = device = None
    try:
        model, device = load_fbcnn(cfg)
        for idx, path in enumerate(wanted, 1):
            key = safe_key(path, input_root); od = output_root / key
            wp = od / "working.png"
            if wp.exists() and not force:
                print(f"[{idx}/{len(wanted)}] FBCNN cached: {path.name}")
                continue
            t0 = time.perf_counter()
            restored, qf = fbcnn_restore(model, device, open_rgb(od / "source.png"), cfg)
            save_rgb_array(restored, wp, comp)
            save_rgb_array(restored, od / "fbcnn_restored.png", comp)
            states[key]["fbcnn_predicted_qf"] = round(qf, 2)
            states[key]["fbcnn_seconds"] = round(time.perf_counter() - t0, 3)
            print(f"[{idx}/{len(wanted)}] FBCNN: {path.name} (pred QF≈{qf:.1f}, {states[key]['fbcnn_seconds']}s)")
    except Exception as e:
        print(f"FBCNN stage failed: {e}", file=sys.stderr)
        for p in wanted:
            key = safe_key(p, input_root); od = output_root / key
            states[key]["error"] = f"FBCNN: {e}"
            # Fail soft: leave an un-restored working image so the blur test can continue.
            shutil.copyfile(od / "source.png", od / "working.png")
    finally:
        del model
        torch_cleanup()


def run_birefnet_stage(images, input_root, output_root, cfg, states, force):
    print("\n== BiRefNet HR matting ==")
    model = device = dtype = None
    comp = int(cfg["output"].get("png_compress_level", 6))
    try:
        model, device, dtype = load_birefnet(cfg)
        for idx, path in enumerate(images, 1):
            key = safe_key(path, input_root); od = output_root / key
            bp = od / "alpha_birefnet.png"
            if bp.exists() and not force:
                print(f"[{idx}/{len(images)}] BiRefNet cached: {path.name}")
                continue
            image = open_rgb(od / "working.png")
            t0 = time.perf_counter()
            alpha = birefnet_alpha(model, device, dtype, image, cfg)
            save_alpha(alpha, bp, comp)
            states[key]["birefnet_seconds"] = round(time.perf_counter() - t0, 3)
            print(f"[{idx}/{len(images)}] BiRefNet: {path.name} ({states[key]['birefnet_seconds']}s)")
    except Exception as e:
        print(f"BiRefNet stage failed: {e}", file=sys.stderr)
        for p in images:
            states[safe_key(p, input_root)]["error"] = f"BiRefNet: {e}"
        raise
    finally:
        del model
        torch_cleanup()


def run_vitmatte_stage(images, input_root, output_root, cfg, states, force):
    mode = cfg["matte"].get("mode", "maximum").lower()
    comp = int(cfg["output"].get("png_compress_level", 6))
    if mode not in {"maximum", "vitmatte", "birefnet-vitmatte"}:
        for p in images:
            key = safe_key(p, input_root); od = output_root / key
            target = od / "alpha.png"
            if force or not target.exists():
                shutil.copyfile(od / "alpha_birefnet.png", target)
        return
    print("\n== ViTMatte-B refinement ==")
    proc = model = device = None
    try:
        proc, model, device = load_vitmatte(cfg)
        for idx, path in enumerate(images, 1):
            key = safe_key(path, input_root); od = output_root / key
            target = od / "alpha.png"
            if target.exists() and not force:
                print(f"[{idx}/{len(images)}] ViTMatte cached: {path.name}")
                continue
            image = open_rgb(od / "working.png")
            base = load_alpha(od / "alpha_birefnet.png")
            t0 = time.perf_counter()
            alpha, tri = vitmatte_refine(proc, model, device, image, base, cfg)
            save_alpha(alpha, target, comp)
            save_png(Image.fromarray(tri, "L"), od / "trimap.png", comp)
            states[key]["vitmatte_seconds"] = round(time.perf_counter() - t0, 3)
            print(f"[{idx}/{len(images)}] ViTMatte: {path.name} ({states[key]['vitmatte_seconds']}s)")
    except Exception as e:
        print(f"ViTMatte stage failed: {e}", file=sys.stderr)
        # Fail soft to BiRefNet matte.
        for p in images:
            key = safe_key(p, input_root); od = output_root / key
            states[key]["error"] = (states[key].get("error", "") + f" ViTMatte: {e}").strip()
            shutil.copyfile(od / "alpha_birefnet.png", od / "alpha.png")
    finally:
        del proc, model
        torch_cleanup()


def run_depth_stage(images, input_root, output_root, cfg, states, force):
    print("\n== Depth Pro ==")
    proc = model = device = dtype = None
    comp = int(cfg["output"].get("png_compress_level", 6))
    try:
        proc, model, device, dtype = load_depthpro(cfg)
        for idx, path in enumerate(images, 1):
            key = safe_key(path, input_root); od = output_root / key
            dp = od / "depth.npy"
            if dp.exists() and not force:
                print(f"[{idx}/{len(images)}] Depth cached: {path.name}")
                continue
            image = open_rgb(od / "working.png")
            t0 = time.perf_counter()
            depth = depthpro_infer(proc, model, device, dtype, image)
            np.save(dp, depth.astype(np.float32))
            save_rgb_array(depth_visualization(depth), od / "depth.png", comp)
            states[key]["depth_seconds"] = round(time.perf_counter() - t0, 3)
            print(f"[{idx}/{len(images)}] Depth Pro: {path.name} ({states[key]['depth_seconds']}s)")
    except Exception as e:
        print(f"Depth stage failed: {e}", file=sys.stderr)
        for p in images:
            states[safe_key(p, input_root)]["error"] = (states[safe_key(p, input_root)].get("error", "") + f" Depth: {e}").strip()
        raise
    finally:
        del proc, model
        torch_cleanup()


def render_stage(images, input_root, output_root, cfg, states, presets, compare, force):
    print("\n== Rendering depth-aware blur ==")
    comp = int(cfg["output"].get("png_compress_level", 6))
    for idx, path in enumerate(images, 1):
        key = safe_key(path, input_root); od = output_root / key
        try:
            image = open_rgb(od / "working.png")
            original = np.asarray(image, np.uint8)
            alpha = load_alpha(od / "alpha.png")
            depth = np.load(od / "depth.npy")
            # Convenience 8-bit alpha preview while retaining 16-bit alpha.png.
            save_png(Image.fromarray(np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8), "L"), od / "alpha_preview.png", comp)

            plate_path = od / "background_plate.png"
            if force or not plate_path.exists():
                plate = make_background_plate(original, alpha, cfg)
                if bool(cfg["output"].get("save_background_plate", True)):
                    save_rgb_array(plate, plate_path, comp)
            else:
                plate = np.asarray(open_rgb(plate_path), np.uint8)

            focus_values = []
            for preset_name in presets:
                outp = od / f"depth_{preset_name}.png"
                mapp = od / f"blurmap_{preset_name}.png"
                if outp.exists() and not force:
                    continue
                preset = cfg["blur"]["presets"][preset_name]
                bmap, focus = build_blur_map(depth, alpha, preset, cfg)
                focus_values.append(focus)
                max_r = preset_max_radius(image, preset)
                blurred = variable_gaussian_blur(plate, bmap, max_r, cfg["blur"].get("pyramid_fractions", [0, .1, .25, .5, .75, 1]))
                final = composite_subject(original, blurred, alpha)
                save_rgb_array(final, outp, comp)
                save_png(Image.fromarray(np.clip(bmap * 255 + 0.5, 0, 255).astype(np.uint8), "L"), mapp, comp)

            if compare:
                up = od / "uniform.png"
                if force or not up.exists():
                    p = cfg["blur"]["presets"]["natural"]
                    ub = uniform_blur_map(alpha, float(p.get("strength", 0.72)))
                    blurred = variable_gaussian_blur(plate, ub, preset_max_radius(image, p), cfg["blur"].get("pyramid_fractions", [0, .1, .25, .5, .75, 1]))
                    save_rgb_array(composite_subject(original, blurred, alpha), up, comp)

            if focus_values:
                states[key]["focus_depth_m"] = round(float(np.median(focus_values)), 4)
            if presets:
                # final.png is the selected/default preset when not comparing; natural when present.
                preferred = "natural" if "natural" in presets else presets[0]
                shutil.copyfile(od / f"depth_{preferred}.png", od / "final.png")
            write_json(od / "metadata.json", states[key])
            print(f"[{idx}/{len(images)}] Rendered: {path.name}")
        except Exception as e:
            states[key]["error"] = (states[key].get("error", "") + f" Render: {e}").strip()
            print(f"Render failed {path}: {e}", file=sys.stderr)


def parse_args():
    ap = argparse.ArgumentParser(description="Depth-aware background blur harness for character images")
    ap.add_argument("input", type=Path, help="Input image directory")
    ap.add_argument("output", type=Path, help="Output directory")
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--fbcnn", choices=["on", "auto", "off"], default=None, help="JPEG artifact restoration. auto only detects true .jpg/.jpeg inputs; use on for ex-JPEG PNGs.")
    ap.add_argument("--fbcnn-qf", default=None, help="FBCNN control QF: auto or integer 1..100 (lower = stronger cleanup)")
    ap.add_argument("--matte", choices=["high", "maximum"], default=None, help="high=BiRefNet HR, maximum=BiRefNet HR -> ViTMatte-B")
    ap.add_argument("--preset", choices=PRESET_ORDER, default="natural")
    ap.add_argument("--compare", action="store_true", help="Render subtle/natural/strong depth-aware variants plus uniform blur")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-recursive", action="store_true")
    ap.add_argument("--force", action="store_true", help="Recompute cached stages")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    if args.fbcnn:
        cfg["fbcnn"]["mode"] = args.fbcnn
    if args.fbcnn_qf is not None:
        cfg["fbcnn"]["qf"] = args.fbcnn_qf if args.fbcnn_qf == "auto" else int(args.fbcnn_qf)
    if args.matte:
        cfg["matte"]["mode"] = args.matte

    input_root = args.input.resolve(); output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    images = discover_images(input_root, cfg["input"]["extensions"], bool(cfg["input"].get("recursive", True)) and not args.no_recursive)
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"No supported images found under {input_root}")

    fbcnn_mode = str(cfg["fbcnn"].get("mode", "auto")).lower()
    presets = PRESET_ORDER if args.compare else [args.preset]
    states: Dict[str, dict] = {}

    prior_cfg_path = output_root / "run_config.json"
    if prior_cfg_path.exists() and not args.force:
        try:
            prior_cfg = json.loads(prior_cfg_path.read_text(encoding="utf-8"))
            if prior_cfg != cfg:
                raise SystemExit(
                    "This output directory was created with a different pipeline config. "
                    "Use --force to invalidate caches, or choose a new output directory."
                )
        except json.JSONDecodeError:
            pass

    print(f"Images: {len(images)}")
    print(f"FBCNN: {fbcnn_mode} (QF={cfg['fbcnn'].get('qf','auto')})")
    print(f"Matte: {cfg['matte'].get('mode','maximum')}")
    print(f"Blur outputs: {', '.join(presets)}" + (" + uniform" if args.compare else ""))

    prepare_sources(images, input_root, output_root, cfg, states, fbcnn_mode, args.force)
    run_fbcnn_stage(images, input_root, output_root, cfg, states, fbcnn_mode, args.force)
    run_birefnet_stage(images, input_root, output_root, cfg, states, args.force)
    run_vitmatte_stage(images, input_root, output_root, cfg, states, args.force)
    run_depth_stage(images, input_root, output_root, cfg, states, args.force)
    render_stage(images, input_root, output_root, cfg, states, presets, args.compare, args.force)
    write_report(output_root, states, presets, args.compare)
    write_json(output_root / "run_config.json", cfg)
    print(f"\nReport: {output_root / 'report.html'}")
    print(f"Final PNG per image: <image>/final.png")


if __name__ == "__main__":
    main()
