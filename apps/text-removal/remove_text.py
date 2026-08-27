#!/usr/bin/env python3
"""
Batch text / watermark removal harness.

Pipeline:
  source image
    -> PaddleOCR high-resolution text polygons
    -> merge nearby polygons into repair regions
    -> dilated hard repair mask + feathered blend mask
    -> generous context crop
    -> Qwen-Image-Edit-2511 reconstruction of the crop
    -> geometric registration of Qwen output to untouched crop context
    -> paste back ONLY through the local blend mask
    -> per-image intermediates + HTML batch report

The Qwen model is loaded lazily: `--dry-run` performs detection/mask/crop QC
without downloading or loading Qwen-Image-Edit.
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import math
import os
import re
import shutil
import sys
import traceback
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()


# ------------------------------- data types ---------------------------------


@dataclass
class Detection:
    polygon: list[list[int]]
    text: str
    score: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        a = np.asarray(self.polygon, dtype=np.int32)
        return int(a[:, 0].min()), int(a[:, 1].min()), int(a[:, 0].max()), int(a[:, 1].max())


@dataclass
class Region:
    index: int
    detection_indices: list[int]
    bbox: tuple[int, int, int, int]
    crop_box: tuple[int, int, int, int]


# -------------------------------- utilities ---------------------------------


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def save_rgb(im: Image.Image, path: Path, jpeg_quality: int = 96) -> None:
    ensure_dir(path.parent)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        im.save(path, quality=jpeg_quality, subsampling=0)
    else:
        im.save(path)


def safe_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parts = list(rel.with_suffix("").parts)
    return "__".join(re.sub(r"[^A-Za-z0-9._-]+", "_", p) for p in parts)


def list_images(root: Path, extensions: set[str], recursive: bool) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in extensions else []
    globber = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in globber if p.is_file() and p.suffix.lower() in extensions)


def bbox_union(boxes: Sequence[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def boxes_near(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 + gap < bx1 or bx2 + gap < ax1 or ay2 + gap < by1 or by2 + gap < ay1)


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def crop_box_with_context(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    context_scale: float,
    min_context_px: int,
    max_aspect: float,
) -> tuple[int, int, int, int]:
    """Expand a repair bbox into a generous, not-too-skinny context crop."""
    W, H = image_size
    x1, y1, x2, y2 = bbox
    rw = max(1, x2 - x1 + 1)
    rh = max(1, y2 - y1 + 1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    target_w = max(rw * context_scale, rw + 2 * min_context_px)
    target_h = max(rh * context_scale, rh + 2 * min_context_px)

    if target_w / target_h > max_aspect:
        target_h = target_w / max_aspect
    elif target_h / target_w > max_aspect:
        target_w = target_h / max_aspect

    target_w = min(float(W), target_w)
    target_h = min(float(H), target_h)

    left = int(round(cx - target_w / 2))
    top = int(round(cy - target_h / 2))
    right = left + int(round(target_w))
    bottom = top + int(round(target_h))

    # Shift, rather than shrink, when a crop hits an edge.
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > W:
        left -= right - W
        right = W
    if bottom > H:
        top -= bottom - H
        bottom = H

    left = clamp(left, 0, W - 1)
    top = clamp(top, 0, H - 1)
    right = clamp(right, left + 1, W)
    bottom = clamp(bottom, top + 1, H)
    return left, top, right, bottom


def work_size(
    source_size: tuple[int, int], target_megapixels: float, multiple: int
) -> tuple[int, int]:
    """Match Qwen/Diffusers geometry: preserve ratio, target an area, snap to 32."""
    w, h = source_size
    ratio = w / max(1, h)
    target_area = max(0.1, float(target_megapixels)) * 1024.0 * 1024.0
    tw = math.sqrt(target_area * ratio)
    th = tw / ratio
    tw = max(multiple, int(round(tw / multiple)) * multiple)
    th = max(multiple, int(round(th / multiple)) * multiple)
    return tw, th


def pil_mask(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8), mode="L")


def feather_mask(hard_mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return hard_mask.astype(np.float32) / 255.0
    # Dilate slightly before Gaussian blur so the originally requested repair
    # area remains at full opacity and only the outer seam feathers.
    pre = max(1, radius // 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pre * 2 + 1, pre * 2 + 1))
    expanded = cv2.dilate(hard_mask, kernel)
    sigma = max(0.5, radius / 2.5)
    blurred = cv2.GaussianBlur(expanded, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(blurred.astype(np.float32) / 255.0, 0.0, 1.0)


def alpha_blend(original: Image.Image, edited: Image.Image, alpha: np.ndarray) -> Image.Image:
    a = np.asarray(original.convert("RGB"), dtype=np.float32)
    b = np.asarray(edited.convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Blend images differ: {a.shape} vs {b.shape}")
    m = alpha[..., None].astype(np.float32)
    out = a * (1.0 - m) + b * m
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    return obj


# ------------------------------- OCR detector --------------------------------


class PaddleTextDetector:
    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = cfg
        self._ocr = None

    def _load(self) -> None:
        if self._ocr is not None:
            return
        from paddleocr import PaddleOCR

        kwargs: dict[str, Any] = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "engine": self.cfg.get("engine", "transformers"),
            "device": self.cfg.get("device", "cpu"),
            "lang": self.cfg.get("lang", "en"),
            "ocr_version": self.cfg.get("ocr_version", "PP-OCRv6"),
        }
        for key in (
            "text_det_limit_side_len",
            "text_det_limit_type",
            "text_det_thresh",
            "text_det_box_thresh",
            "text_det_unclip_ratio",
        ):
            if self.cfg.get(key) is not None:
                kwargs[key] = self.cfg[key]

        console.print(f"[cyan]Loading PaddleOCR[/cyan] ({kwargs['engine']}, {kwargs['device']})")
        self._ocr = PaddleOCR(**kwargs)

    @staticmethod
    def _result_mapping(res: Any) -> Mapping[str, Any]:
        candidates: list[Any] = []
        for attr in ("res", "json"):
            if hasattr(res, attr):
                value = getattr(res, attr)
                try:
                    value = value() if callable(value) else value
                except TypeError:
                    pass
                candidates.append(value)
        candidates.append(res)

        for value in candidates:
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except Exception:
                    continue
            if isinstance(value, Mapping):
                # PaddleX result objects are commonly {'res': {...}}.
                inner = value.get("res")
                if isinstance(inner, Mapping):
                    return inner
                return value
        raise TypeError(f"Could not extract PaddleOCR result mapping from {type(res)!r}")

    def detect(self, image: Image.Image) -> tuple[list[Detection], list[dict[str, Any]]]:
        self._load()
        assert self._ocr is not None
        # Use the already EXIF-normalized RGB pixels so OCR coordinates always
        # match the image that will later be cropped and composited.
        raw_results = list(self._ocr.predict(np.asarray(image.convert("RGB"))))
        detections: list[Detection] = []
        raw_json: list[dict[str, Any]] = []
        min_rec_score = float(self.cfg.get("min_rec_score", 0.0))

        for res in raw_results:
            data = dict(self._result_mapping(res))
            raw_json.append(jsonable(data))

            # Prefer recognized polygons because they align with rec_texts/scores.
            polys = data.get("rec_polys")
            texts = data.get("rec_texts")
            scores = data.get("rec_scores")
            if texts is None:
                texts = []
            if scores is None:
                scores = []

            if polys is None or len(polys) == 0:
                # Fallbacks for detection-only or future PaddleOCR result variants.
                for key in ("dt_polys", "text_det_polys", "polys"):
                    if data.get(key) is not None:
                        polys = data[key]
                        break

            if polys is None:
                continue

            for i, poly in enumerate(polys):
                arr = np.asarray(poly, dtype=np.int32)
                if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
                    continue
                text = str(texts[i]) if i < len(texts) else ""
                score = float(scores[i]) if i < len(scores) else 1.0
                if score < min_rec_score:
                    continue
                detections.append(
                    Detection(polygon=arr.tolist(), text=text, score=score)
                )

        return detections, raw_json


# ---------------------------- geometry / selection ---------------------------


def filter_detections(dets: list[Detection], cfg: Mapping[str, Any]) -> list[Detection]:
    include = cfg.get("include_regex")
    exclude = cfg.get("exclude_regex")
    inc_re = re.compile(include) if include else None
    exc_re = re.compile(exclude) if exclude else None
    out: list[Detection] = []
    for d in dets:
        if inc_re and not inc_re.search(d.text):
            continue
        if exc_re and exc_re.search(d.text):
            continue
        out.append(d)
    return out


def group_detections(
    dets: Sequence[Detection], image_size: tuple[int, int], cfg: Mapping[str, Any]
) -> list[Region]:
    n = len(dets)
    if n == 0:
        return []
    gap = int(cfg.get("group_gap_px", 32))
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    boxes = [d.bbox for d in dets]
    for i in range(n):
        for j in range(i + 1, n):
            if boxes_near(boxes[i], boxes[j], gap):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    regions: list[Region] = []
    for idx, ids in enumerate(sorted(groups.values(), key=lambda xs: min(boxes[x][1] for x in xs))):
        bbox = bbox_union([boxes[i] for i in ids])
        cbox = crop_box_with_context(
            bbox,
            image_size,
            float(cfg.get("context_scale", 3.0)),
            int(cfg.get("min_context_px", 160)),
            float(cfg.get("max_crop_aspect", 2.0)),
        )
        regions.append(Region(idx, ids, bbox, cbox))
    return regions


def mask_for_detection_indices(
    image_size: tuple[int, int], dets: Sequence[Detection], ids: Iterable[int], dilate_px: int
) -> np.ndarray:
    W, H = image_size
    mask = np.zeros((H, W), dtype=np.uint8)
    for i in ids:
        poly = np.asarray(dets[i].polygon, dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)
    if dilate_px > 0 and mask.any():
        k = dilate_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel)
    return mask


def draw_detection_overlay(im: Image.Image, dets: Sequence[Detection], regions: Sequence[Region]) -> Image.Image:
    out = im.copy()
    draw = ImageDraw.Draw(out)
    for i, d in enumerate(dets):
        pts = [tuple(p) for p in d.polygon]
        draw.line(pts + [pts[0]], fill=(255, 60, 60), width=max(2, im.width // 1000))
        label = f"{i}: {d.text[:40]} ({d.score:.2f})"
        x, y = d.bbox[0], max(0, d.bbox[1] - 16)
        draw.text((x, y), label, fill=(255, 255, 0))
    for r in regions:
        x1, y1, x2, y2 = r.crop_box
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(80, 210, 255), width=max(2, im.width // 800))
        draw.text((x1 + 4, y1 + 4), f"region {r.index}", fill=(80, 210, 255))
    return out


# ------------------------------ editor backends ------------------------------


def torch_dtype_from_name(torch: Any, name: str) -> Any:
    dtype = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(str(name).lower())
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {name}")
    return dtype


class CropEditor:
    """Small common interface for local image-edit backends."""

    def __init__(self, name: str, cfg: Mapping[str, Any]):
        self.name = name
        self.cfg = cfg
        self.pipe = None
        self.torch = None
        self.load_seconds: float | None = None

    @property
    def label(self) -> str:
        return str(self.cfg.get("label", self.name))

    @property
    def model_id(self) -> str:
        return str(self.cfg.get("model", ""))

    def _load(self) -> None:
        raise NotImplementedError

    def edit(self, crop: Image.Image, seed: int) -> tuple[Image.Image, dict[str, Any]]:
        raise NotImplementedError

    def _generator(self, seed: int) -> Any:
        assert self.torch is not None
        device = str(self.cfg.get("generator_device", "cpu"))
        try:
            return self.torch.Generator(device=device).manual_seed(int(seed))
        except Exception:
            return self.torch.Generator(device="cpu").manual_seed(int(seed))

    def _model_input(self, crop: Image.Image, tw: int, th: int) -> Image.Image:
        # Explicit pre-resize makes input and requested output geometry agree.
        # Qwen/FireRed would resize internally anyway; doing it here makes the
        # geometry we benchmark explicit and reduces independent resize choices.
        if bool(self.cfg.get("pre_resize_input", True)):
            return crop.resize((tw, th), Image.Resampling.LANCZOS)
        return crop

    def close(self) -> None:
        pipe = self.pipe
        self.pipe = None
        if pipe is not None:
            try:
                del pipe
            except Exception:
                pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


class QwenFamilyCropEditor(CropEditor):
    """Qwen-Image-Edit-2511 and FireRed 1.1 share QwenImageEditPlusPipeline."""

    def _load(self) -> None:
        if self.pipe is not None:
            return
        started = time.perf_counter()
        import torch
        from diffusers import QwenImageEditPlusPipeline

        self.torch = torch
        dtype_name = str(self.cfg.get("dtype", "bfloat16"))
        dtype = torch_dtype_from_name(torch, dtype_name)
        console.print(f"[cyan]Loading {self.label}[/cyan]: {self.model_id} ({dtype_name})")
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(self.model_id, torch_dtype=dtype)
        self.pipe.set_progress_bar_config(disable=False)
        if bool(self.cfg.get("cpu_offload", False)):
            console.print(f"[yellow]{self.label}: enabling model CPU offload[/yellow]")
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(str(self.cfg.get("device", "cuda")))
        self.load_seconds = time.perf_counter() - started

    def edit(self, crop: Image.Image, seed: int) -> tuple[Image.Image, dict[str, Any]]:
        self._load()
        assert self.pipe is not None and self.torch is not None
        torch = self.torch
        tw, th = work_size(
            crop.size,
            float(self.cfg.get("target_megapixels", 1.0)),
            int(self.cfg.get("dimension_multiple", 32)),
        )
        model_input = self._model_input(crop, tw, th)
        gen = self._generator(seed)
        kwargs: dict[str, Any] = {
            "image": [model_input],
            "prompt": str(self.cfg["prompt"]),
            "generator": gen,
            "true_cfg_scale": float(self.cfg.get("true_cfg_scale", 4.0)),
            "negative_prompt": str(self.cfg.get("negative_prompt", " ")),
            "num_inference_steps": int(self.cfg.get("num_inference_steps", 40)),
            "guidance_scale": float(self.cfg.get("guidance_scale", 1.0)),
            "num_images_per_prompt": 1,
            "height": th,
            "width": tw,
        }
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                result = self.pipe(**kwargs)
        except TypeError as e:
            # Compatibility fallback for pipeline revisions that choose output
            # size entirely from the reference image.
            if "height" not in str(e) and "width" not in str(e):
                raise
            kwargs.pop("height", None)
            kwargs.pop("width", None)
            with torch.inference_mode():
                result = self.pipe(**kwargs)
        infer_seconds = time.perf_counter() - started
        edited = result.images[0].convert("RGB")
        if edited.size != (tw, th):
            edited = edited.resize((tw, th), Image.Resampling.LANCZOS)
        source_res = edited.resize(crop.size, Image.Resampling.LANCZOS)
        return source_res, {
            "backend": self.name,
            "model": self.model_id,
            "source_crop_size": list(crop.size),
            "work_size": [tw, th],
            "target_megapixels": float(self.cfg.get("target_megapixels", 1.0)),
            "seed": int(seed),
            "num_inference_steps": int(self.cfg.get("num_inference_steps", 40)),
            "inference_seconds": infer_seconds,
        }


class FluxKleinCropEditor(CropEditor):
    """FLUX.2 Klein single-reference editing through Flux2KleinPipeline."""

    def _load(self) -> None:
        if self.pipe is not None:
            return
        started = time.perf_counter()
        import torch
        from diffusers import Flux2KleinPipeline

        self.torch = torch
        dtype_name = str(self.cfg.get("dtype", "bfloat16"))
        dtype = torch_dtype_from_name(torch, dtype_name)
        console.print(f"[cyan]Loading {self.label}[/cyan]: {self.model_id} ({dtype_name})")
        try:
            self.pipe = Flux2KleinPipeline.from_pretrained(self.model_id, torch_dtype=dtype)
        except OSError as e:
            if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
                raise RuntimeError(
                    "FLUX.2 Klein 9B is gated on Hugging Face. Accept the model license "
                    "on its Hugging Face page and authenticate the pod with HF_TOKEN or `hf auth login`."
                ) from e
            raise
        self.pipe.set_progress_bar_config(disable=False)
        if bool(self.cfg.get("cpu_offload", False)):
            console.print(f"[yellow]{self.label}: enabling model CPU offload[/yellow]")
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(str(self.cfg.get("device", "cuda")))
        self.load_seconds = time.perf_counter() - started

    def edit(self, crop: Image.Image, seed: int) -> tuple[Image.Image, dict[str, Any]]:
        self._load()
        assert self.pipe is not None and self.torch is not None
        torch = self.torch
        tw, th = work_size(
            crop.size,
            float(self.cfg.get("target_megapixels", 1.0)),
            int(self.cfg.get("dimension_multiple", 16)),
        )
        model_input = self._model_input(crop, tw, th)
        gen = self._generator(seed)
        kwargs: dict[str, Any] = {
            "image": model_input,
            "prompt": str(self.cfg["prompt"]),
            "height": th,
            "width": tw,
            "num_inference_steps": int(self.cfg.get("num_inference_steps", 4)),
            "guidance_scale": float(self.cfg.get("guidance_scale", 1.0)),
            "num_images_per_prompt": 1,
            "generator": gen,
        }
        if self.cfg.get("caption_upsample_temperature") is not None:
            kwargs["caption_upsample_temperature"] = float(self.cfg["caption_upsample_temperature"])
        started = time.perf_counter()
        with torch.inference_mode():
            result = self.pipe(**kwargs)
        infer_seconds = time.perf_counter() - started
        edited = result.images[0].convert("RGB")
        if edited.size != (tw, th):
            edited = edited.resize((tw, th), Image.Resampling.LANCZOS)
        source_res = edited.resize(crop.size, Image.Resampling.LANCZOS)
        return source_res, {
            "backend": self.name,
            "model": self.model_id,
            "source_crop_size": list(crop.size),
            "work_size": [tw, th],
            "target_megapixels": float(self.cfg.get("target_megapixels", 1.0)),
            "seed": int(seed),
            "num_inference_steps": int(self.cfg.get("num_inference_steps", 4)),
            "inference_seconds": infer_seconds,
        }


def build_editor(name: str, editors_cfg: Mapping[str, Any]) -> CropEditor:
    if name not in editors_cfg:
        raise KeyError(f"Unknown editor {name!r}; available: {', '.join(editors_cfg)}")
    cfg = editors_cfg[name]
    if not isinstance(cfg, Mapping):
        raise ValueError(f"editors.{name} must be a mapping")
    kind = str(cfg.get("kind", "qwen_family"))
    if kind == "qwen_family":
        return QwenFamilyCropEditor(name, cfg)
    if kind == "flux_klein":
        return FluxKleinCropEditor(name, cfg)
    raise ValueError(f"Unsupported editor kind {kind!r} for {name}")


def _registration_stable_mask(hard_mask: np.ndarray, expand_px: int) -> np.ndarray:
    """Pixels safe to use for registration: everything well outside the repair area."""
    if hard_mask.ndim != 2:
        raise ValueError("hard_mask must be single-channel")
    mask = hard_mask.astype(np.uint8)
    if expand_px > 0 and mask.any():
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (expand_px * 2 + 1, expand_px * 2 + 1)
        )
        mask = cv2.dilate(mask, k)
    stable = cv2.bitwise_not(mask)
    border = max(2, min(stable.shape[:2]) // 100)
    stable[:border, :] = 0
    stable[-border:, :] = 0
    stable[:, :border] = 0
    stable[:, -border:] = 0
    return stable


def _masked_gray_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("inf")
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sel = mask > 0
    return float(np.mean(np.abs(ga[sel] - gb[sel])))


def register_edit_to_source(
    source: Image.Image,
    edited: Image.Image,
    hard_mask: np.ndarray,
    cfg: Mapping[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    """Register Qwen output to source coordinates using only untouched context."""
    if not bool(cfg.get("enabled", True)):
        return edited, {"enabled": False, "accepted": False, "reason": "disabled"}

    src = np.asarray(source.convert("RGB"))
    edt = np.asarray(edited.convert("RGB"))
    if src.shape != edt.shape:
        raise ValueError(f"Registration size mismatch: {src.shape} vs {edt.shape}")

    h, w = hard_mask.shape
    stable = _registration_stable_mask(
        hard_mask, int(cfg.get("exclude_mask_expand_px", 48))
    )
    stable_fraction = float(np.count_nonzero(stable)) / float(stable.size)
    if stable_fraction < float(cfg.get("min_stable_fraction", 0.20)):
        return edited, {
            "enabled": True,
            "accepted": False,
            "reason": "too little stable context",
            "stable_fraction": stable_fraction,
        }

    src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    edt_gray = cv2.cvtColor(edt, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(
        nfeatures=int(cfg.get("sift_features", 4000)),
        contrastThreshold=float(cfg.get("sift_contrast_threshold", 0.02)),
    )
    kp_e, des_e = sift.detectAndCompute(edt_gray, stable)
    kp_s, des_s = sift.detectAndCompute(src_gray, stable)
    min_matches = int(cfg.get("min_matches", 10))
    if des_e is None or des_s is None or len(kp_e) < min_matches or len(kp_s) < min_matches:
        return edited, {
            "enabled": True,
            "accepted": False,
            "reason": "insufficient SIFT features",
            "edited_keypoints": len(kp_e),
            "source_keypoints": len(kp_s),
            "stable_fraction": stable_fraction,
        }

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(des_e, des_s, k=2)
    ratio_test = float(cfg.get("match_ratio", 0.75))
    good = [m for m, n in pairs if m.distance < ratio_test * n.distance]
    if len(good) < min_matches:
        return edited, {
            "enabled": True,
            "accepted": False,
            "reason": "insufficient good matches",
            "good_matches": len(good),
            "stable_fraction": stable_fraction,
        }

    pts_e = np.float32([kp_e[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_s = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(
        pts_e,
        pts_s,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(cfg.get("ransac_reproj_px", 3.0)),
        maxIters=int(cfg.get("ransac_max_iters", 3000)),
        confidence=float(cfg.get("ransac_confidence", 0.995)),
        refineIters=20,
    )
    if M is None or inliers is None:
        return edited, {
            "enabled": True,
            "accepted": False,
            "reason": "transform estimation failed",
            "good_matches": len(good),
            "stable_fraction": stable_fraction,
        }

    inlier_count = int(inliers.ravel().sum())
    inlier_ratio = inlier_count / max(1, len(good))
    a, b, tx = [float(x) for x in M[0]]
    c, d, ty = [float(x) for x in M[1]]
    scale_x = math.hypot(a, c)
    scale_y = math.hypot(b, d)
    scale = (scale_x + scale_y) / 2.0
    rotation_deg = math.degrees(math.atan2(c, a))
    translation_px = math.hypot(tx, ty)

    geom_ok = (
        abs(scale - 1.0) <= float(cfg.get("max_scale_delta", 0.05))
        and abs(rotation_deg) <= float(cfg.get("max_rotation_deg", 2.0))
        and translation_px <= max(w, h) * float(cfg.get("max_translation_fraction", 0.08))
        and inlier_count >= min_matches
        and inlier_ratio >= float(cfg.get("min_inlier_ratio", 0.35))
    )

    meta: dict[str, Any] = {
        "enabled": True,
        "good_matches": len(good),
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "stable_fraction": stable_fraction,
        "matrix": M.tolist(),
        "scale": scale,
        "rotation_deg": rotation_deg,
        "translation_x_px": tx,
        "translation_y_px": ty,
        "translation_px": translation_px,
    }
    if not geom_ok:
        meta.update({"accepted": False, "reason": "transform outside safety bounds"})
        return edited, meta

    warped = cv2.warpAffine(
        edt,
        M,
        (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    before = _masked_gray_mae(src, edt, stable)
    after = _masked_gray_mae(src, warped, stable)
    improvement = 0.0 if before <= 1e-6 else (before - after) / before
    meta.update({
        "mae_before": before,
        "mae_after": after,
        "mae_improvement_fraction": improvement,
    })
    if not math.isfinite(after) or improvement < float(cfg.get("min_mae_improvement", 0.03)):
        meta.update({
            "accepted": False,
            "reason": "registration did not improve stable context",
        })
        return edited, meta

    meta.update({"accepted": True, "reason": "similarity registration accepted"})
    return Image.fromarray(warped, mode="RGB"), meta


# ------------------------------ report / metrics -----------------------------


def read_metadata(item_dir: Path) -> dict[str, Any]:
    p = item_dir / "metadata.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_comparison_csv(out_root: Path, items: list[dict[str, Any]], editor_names: Sequence[str]) -> None:
    path = out_root / "comparison.csv"
    fields = [
        "source", "editor", "status", "detections", "regions", "total_inference_seconds",
        "registration_accepted", "registration_attempted", "mean_mae_before", "mean_mae_after",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for item in items:
            meta = read_metadata(out_root / item["dir"])
            for name in editor_names:
                ed = meta.get("editors", {}).get(name, {})
                regs = [r.get("registration", {}) for r in ed.get("regions", [])]
                before = [float(r["mae_before"]) for r in regs if r.get("mae_before") is not None]
                after = [float(r["mae_after"]) for r in regs if r.get("mae_after") is not None]
                w.writerow({
                    "source": item.get("source", ""),
                    "editor": name,
                    "status": ed.get("status", "not run"),
                    "detections": meta.get("selected_detection_count", item.get("detections", 0)),
                    "regions": len(meta.get("regions", [])),
                    "total_inference_seconds": ed.get("total_inference_seconds", ""),
                    "registration_accepted": sum(1 for r in regs if r.get("accepted")),
                    "registration_attempted": len(regs),
                    "mean_mae_before": (sum(before) / len(before)) if before else "",
                    "mean_mae_after": (sum(after) / len(after)) if after else "",
                })


def write_report(out_root: Path, items: list[dict[str, Any]], editor_names: Sequence[str], labels: Mapping[str, str]) -> None:
    cards: list[str] = []
    for item in items:
        rel = item["dir"]
        meta = read_metadata(out_root / rel)
        status = html.escape(str(item.get("status", meta.get("status", ""))))
        detected = int(meta.get("selected_detection_count", item.get("detections", 0)))
        error = item.get("error")
        error_html = f"<pre>{html.escape(error)}</pre>" if error else ""

        def img(name: str, label: str) -> str:
            p = out_root / rel / name
            if not p.exists():
                return ""
            src = f"{html.escape(rel)}/{html.escape(name)}"
            return f'<figure><a href="{src}"><img src="{src}"></a><figcaption>{html.escape(label)}</figcaption></figure>'

        editor_figures: list[str] = []
        editor_stats: list[str] = []
        for name in editor_names:
            editor_figures.append(img(f"final_{name}.png", labels.get(name, name)))
            ed = meta.get("editors", {}).get(name)
            if isinstance(ed, Mapping):
                secs = ed.get("total_inference_seconds")
                regions = ed.get("regions", [])
                accepted = sum(1 for r in regions if r.get("registration", {}).get("accepted"))
                attempted = len(regions)
                time_text = f"{float(secs):.1f}s" if isinstance(secs, (int, float)) else "n/a"
                editor_stats.append(
                    f"<li><b>{html.escape(labels.get(name,name))}</b>: {html.escape(str(ed.get('status','')))}, "
                    f"inference {time_text}, registration {accepted}/{attempted}</li>"
                )

        cards.append(f"""
<section class="card">
  <h2>{html.escape(item['source'])}</h2>
  <p><b>{status}</b> — {detected} selected text detections</p>
  <div class="grid prep">
    {img('original.png', 'Original')}
    {img('detection.png', 'OCR + context regions')}
    {img('mask.png', 'Repair mask')}
  </div>
  <div class="grid finals">{''.join(editor_figures)}</div>
  <ul>{''.join(editor_stats)}</ul>
  {error_html}
  <p><a href="{html.escape(rel)}/metadata.json">metadata.json</a></p>
</section>
""")

    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Text removal editor comparison</title>
<style>
body {{ font-family: system-ui, sans-serif; margin:24px; background:#111; color:#eee; }}
a {{ color:#8bd5ff; }} .card {{ border:1px solid #444; border-radius:12px; padding:16px; margin:0 0 22px; background:#191919; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; margin-bottom:14px; }}
figure {{ margin:0; }} img {{ width:100%; height:340px; object-fit:contain; background:#080808; border-radius:8px; }}
figcaption {{ text-align:center; color:#bbb; padding:5px; }} pre {{ white-space:pre-wrap; color:#ffaaaa; }}
.finals {{ border-top:1px solid #333; padding-top:14px; }} ul {{ color:#ccc; }}
</style></head><body>
<h1>Text / watermark removal editor comparison</h1>
<p>Every editor receives the same OCR-selected context. Each generated crop is independently registered to the untouched source context, then only the local repair mask is blended back.</p>
<p><a href="comparison.csv">comparison.csv</a></p>
{''.join(cards)}
</body></html>"""
    (out_root / "report.html").write_text(document, encoding="utf-8")
    write_comparison_csv(out_root, items, editor_names)


# ------------------------------- preparation ---------------------------------


def metadata_to_detections(meta: Mapping[str, Any]) -> list[Detection]:
    return [Detection(polygon=d["polygon"], text=d.get("text", ""), score=float(d.get("score", 1.0))) for d in meta.get("detections", [])]


def metadata_to_regions(meta: Mapping[str, Any]) -> list[Region]:
    return [Region(index=int(r["index"]), detection_indices=list(r["detection_indices"]), bbox=tuple(r["bbox"]), crop_box=tuple(r["crop_box"])) for r in meta.get("regions", [])]


def prepare_image(
    source: Path,
    input_root: Path,
    out_root: Path,
    cfg: Mapping[str, Any],
    detector: PaddleTextDetector,
    force: bool,
) -> dict[str, Any]:
    key = safe_key(source, input_root if input_root.is_dir() else input_root.parent)
    item_dir = ensure_dir(out_root / key)
    meta_path = item_dir / "metadata.json"
    prior = read_metadata(item_dir)
    if prior.get("prep_status") == "prepared" and not force:
        return {"source": str(source), "dir": key, "status": "prepared (cached)", "detections": int(prior.get("selected_detection_count", 0))}

    im = load_rgb(source)
    W, H = im.size
    out_cfg = cfg.get("output", {})
    jpeg_quality = int(out_cfg.get("jpeg_quality", 96))
    if bool(out_cfg.get("save_original", True)):
        save_rgb(im, item_dir / "original.png", jpeg_quality)

    detections_all, ocr_raw = detector.detect(im)
    detections = filter_detections(detections_all, cfg.get("selection", {}))
    regions = group_detections(detections, im.size, cfg.get("regions", {}))
    dilate_px = int(cfg.get("regions", {}).get("mask_dilate_px", 18))
    full_mask = mask_for_detection_indices(im.size, detections, range(len(detections)), dilate_px)

    if bool(out_cfg.get("save_detection_overlay", True)):
        draw_detection_overlay(im, detections, regions).save(item_dir / "detection.png")
    if bool(out_cfg.get("save_masks", True)):
        pil_mask(full_mask).save(item_dir / "mask.png")

    for r in regions:
        x1, y1, x2, y2 = r.crop_box
        crop = im.crop((x1, y1, x2, y2))
        if bool(out_cfg.get("save_crops", True)):
            crop.save(item_dir / f"region_{r.index:02d}_crop.png")
        rmask = mask_for_detection_indices(im.size, detections, r.detection_indices, dilate_px)
        if bool(out_cfg.get("save_masks", True)):
            pil_mask(rmask[y1:y2, x1:x2]).save(item_dir / f"region_{r.index:02d}_mask.png")

    metadata: dict[str, Any] = {
        "source": str(source),
        "source_size": [W, H],
        "all_ocr_detection_count": len(detections_all),
        "selected_detection_count": len(detections),
        "detections": [asdict(d) | {"bbox": list(d.bbox)} for d in detections],
        "regions": [{"index": r.index, "detection_indices": r.detection_indices, "bbox": list(r.bbox), "crop_box": list(r.crop_box)} for r in regions],
        "ocr_raw": ocr_raw,
        "prep_status": "prepared",
        "status": "no text detected" if not detections else "prepared",
        "editors": prior.get("editors", {}) if isinstance(prior.get("editors"), dict) else {},
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"source": str(source), "dir": key, "status": metadata["status"], "detections": len(detections)}


# ----------------------------- backend execution -----------------------------


def run_editor_on_item(
    item: dict[str, Any],
    out_root: Path,
    cfg: Mapping[str, Any],
    editor: CropEditor,
    force: bool,
    copy_primary_to_final: bool,
) -> dict[str, Any]:
    item_dir = out_root / item["dir"]
    meta_path = item_dir / "metadata.json"
    meta = read_metadata(item_dir)
    if not meta:
        raise RuntimeError(f"Missing preparation metadata: {meta_path}")
    detections = metadata_to_detections(meta)
    regions = metadata_to_regions(meta)
    name = editor.name
    final_path = item_dir / f"final_{name}.png"

    prior_editor = meta.get("editors", {}).get(name, {})
    if final_path.exists() and prior_editor.get("status") == "completed" and not force:
        return item

    im = load_rgb(Path(meta["source"]))
    if not detections:
        save_rgb(im, final_path, int(cfg.get("output", {}).get("jpeg_quality", 96)))
        meta.setdefault("editors", {})[name] = {"status": "no text detected", "regions": [], "total_inference_seconds": 0.0}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return item

    working = im.copy()
    out_cfg = cfg.get("output", {})
    jpeg_quality = int(out_cfg.get("jpeg_quality", 96))
    dilate_px = int(cfg.get("regions", {}).get("mask_dilate_px", 18))
    feather_px = int(cfg.get("regions", {}).get("blend_feather_px", 12))
    base_seed = int(editor.cfg.get("seed", cfg.get("benchmark", {}).get("seed", 12345)))
    region_meta: list[dict[str, Any]] = []
    total_inference = 0.0

    for r in regions:
        x1, y1, x2, y2 = r.crop_box
        crop = working.crop((x1, y1, x2, y2))
        rmask_full = mask_for_detection_indices(im.size, detections, r.detection_indices, dilate_px)
        hard_crop = rmask_full[y1:y2, x1:x2]
        blend = feather_mask(hard_crop, feather_px)
        seed = base_seed + r.index

        try:
            edited_crop, edit_meta = editor.edit(crop, seed)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError(
                    f"CUDA OOM in {editor.label} region {r.index}. Enable editors.{name}.cpu_offload, "
                    f"reduce target_megapixels, or use a larger GPU."
                ) from e
            raise

        total_inference += float(edit_meta.get("inference_seconds", 0.0))
        if bool(out_cfg.get("save_editor_edits", True)):
            edited_crop.save(item_dir / f"region_{r.index:02d}_{name}_raw.png")

        aligned_crop, registration_meta = register_edit_to_source(crop, edited_crop, hard_crop, cfg.get("registration", {}))
        if bool(out_cfg.get("save_editor_edits", True)):
            aligned_crop.save(item_dir / f"region_{r.index:02d}_{name}_aligned.png")

        blended_crop = alpha_blend(crop, aligned_crop, blend)
        working.paste(blended_crop, (x1, y1))
        region_meta.append({"region": r.index, **jsonable(edit_meta), "registration": jsonable(registration_meta)})

    save_rgb(working, final_path, jpeg_quality)
    if copy_primary_to_final:
        save_rgb(working, item_dir / "final.png", jpeg_quality)
    meta.setdefault("editors", {})[name] = {
        "status": "completed",
        "label": editor.label,
        "model": editor.model_id,
        "load_seconds": editor.load_seconds,
        "total_inference_seconds": total_inference,
        "regions": region_meta,
    }
    meta["status"] = "edited"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return item


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch OCR-localized text/watermark removal with pluggable Qwen, FireRed, and FLUX.2 Klein editors."
    )
    p.add_argument("input", type=Path, help="Input image or directory")
    p.add_argument("output", type=Path, help="Output directory")
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    p.add_argument("--dry-run", action="store_true", help="OCR/mask/crop only; load no image editor")
    p.add_argument("--editor", choices=["qwen", "firered", "klein"], help="Run one editor (otherwise config benchmark.default_editor)")
    p.add_argument("--compare-editors", action="store_true", help="Run every editor listed in config benchmark.compare_editors")
    p.add_argument("--force", action="store_true", help="Re-run OCR and editor outputs")
    p.add_argument("--force-edits", action="store_true", help="Keep cached OCR prep but re-run editor outputs")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N images")
    p.add_argument("--recursive", action="store_true", help="Override config and recurse through input directories")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    input_path = args.input.resolve()
    out_root = ensure_dir(args.output.resolve())
    recursive = bool(args.recursive or cfg.get("input", {}).get("recursive", False))
    extensions = {x.lower() for x in cfg.get("input", {}).get("extensions", [])}
    if not input_path.exists():
        console.print(f"[red]Input does not exist:[/red] {input_path}")
        return 2

    input_root = input_path if input_path.is_dir() else input_path.parent
    images = list_images(input_path, extensions, recursive)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        console.print("[yellow]No supported images found.[/yellow]")
        return 1

    benchmark_cfg = cfg.get("benchmark", {})
    editors_cfg = cfg.get("editors", {})
    if args.compare_editors:
        editor_names = [str(x) for x in benchmark_cfg.get("compare_editors", ["qwen", "firered", "klein"])]
    elif args.editor:
        editor_names = [args.editor]
    else:
        editor_names = [str(benchmark_cfg.get("default_editor", "firered"))]
    if args.dry_run:
        editor_names = []
    for name in editor_names:
        if name not in editors_cfg:
            raise KeyError(f"Editor {name!r} is not defined in config.yaml")

    labels = {name: str(editors_cfg.get(name, {}).get("label", name)) for name in editor_names}
    console.print(f"Found [bold]{len(images)}[/bold] image(s).")
    console.print(f"Mode: [bold]{'dry-run' if args.dry_run else ', '.join(editor_names)}[/bold]")

    detector = PaddleTextDetector(cfg.get("ocr", {}))
    items: list[dict[str, Any]] = []
    prep_failed = 0
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("prepare", total=len(images))
        for source in images:
            progress.update(task, description=f"OCR {source.name}")
            try:
                item = prepare_image(source, input_root, out_root, cfg, detector, args.force)
            except Exception as exc:
                key = safe_key(source, input_root)
                item_dir = ensure_dir(out_root / key)
                tb = traceback.format_exc()
                (item_dir / "error.txt").write_text(tb, encoding="utf-8")
                console.print(f"[red]PREP FAILED[/red] {source}: {exc}")
                item = {"source": str(source), "dir": key, "status": "PREP FAILED", "detections": 0, "error": tb}
                prep_failed += 1
            items.append(item)
            progress.advance(task)
    write_report(out_root, items, editor_names, labels)

    if args.dry_run:
        console.print(f"Report: [bold cyan]{out_root / 'report.html'}[/bold cyan]")
        return 1 if prep_failed else 0

    edit_failures = 0
    # Intentionally run backend-major, not image-major: each large model loads once,
    # processes the whole batch, then is released before the next backend loads.
    for editor_index, name in enumerate(editor_names):
        editor = build_editor(name, editors_cfg)
        console.print(f"\n[bold]=== {editor.label} ===[/bold]")
        try:
            # Resolve no-text and already-completed items before loading a giant
            # model. A fully cached run should perform zero model loads.
            pending: list[dict[str, Any]] = []
            for item in items:
                if item.get("status") == "PREP FAILED":
                    continue
                item_dir = out_root / item["dir"]
                meta = read_metadata(item_dir)
                if int(meta.get("selected_detection_count", 0)) == 0:
                    run_editor_on_item(
                        item, out_root, cfg, editor,
                        force=bool(args.force or args.force_edits),
                        copy_primary_to_final=(len(editor_names) == 1),
                    )
                    continue
                prior = meta.get("editors", {}).get(name, {}) if isinstance(meta.get("editors"), dict) else {}
                if (
                    prior.get("status") == "completed"
                    and (item_dir / f"final_{name}.png").exists()
                    and not (args.force or args.force_edits)
                ):
                    continue
                pending.append(item)

            if not pending:
                console.print(f"[green]{editor.label}: nothing to run (cached/no text).[/green]")
                continue

            # Load once before entering the image loop. A gated-model/auth or
            # dependency failure should fail the backend once, not once per image.
            editor._load()
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
                task = progress.add_task(name, total=len(pending))
                for item in pending:
                    progress.update(task, description=f"{name}: {Path(item['source']).name}")
                    try:
                        run_editor_on_item(
                            item,
                            out_root,
                            cfg,
                            editor,
                            force=bool(args.force or args.force_edits),
                            copy_primary_to_final=(len(editor_names) == 1),
                        )
                    except Exception as exc:
                        item_dir = out_root / item["dir"]
                        tb = traceback.format_exc()
                        (item_dir / f"error_{name}.txt").write_text(tb, encoding="utf-8")
                        meta = read_metadata(item_dir)
                        meta.setdefault("editors", {})[name] = {"status": "FAILED", "error": tb, "regions": []}
                        (item_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                        console.print(f"[red]{name} FAILED[/red] {item['source']}: {exc}")
                        edit_failures += 1
                    write_report(out_root, items, editor_names, labels)
                    progress.advance(task)
        except Exception as exc:
            tb = traceback.format_exc()
            console.print(f"[red]{name} BACKEND FAILED[/red]: {exc}")
            for item in pending if 'pending' in locals() else items:
                if item.get("status") == "PREP FAILED":
                    continue
                item_dir = out_root / item["dir"]
                meta = read_metadata(item_dir)
                existing = meta.get("editors", {}).get(name, {}) if isinstance(meta.get("editors"), dict) else {}
                if existing.get("status") == "completed" and not (args.force or args.force_edits):
                    continue
                meta.setdefault("editors", {})[name] = {"status": "BACKEND FAILED", "error": tb, "regions": []}
                (item_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            (out_root / f"backend_error_{name}.txt").write_text(tb, encoding="utf-8")
            edit_failures += 1
        finally:
            editor.close()
            write_report(out_root, items, editor_names, labels)

    console.print()
    console.print(f"Report: [bold cyan]{out_root / 'report.html'}[/bold cyan]")
    console.print(f"CSV:    [bold cyan]{out_root / 'comparison.csv'}[/bold cyan]")
    return 1 if (prep_failed or edit_failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
