#!/usr/bin/env python3
"""Boundary-safe blur-map and matte-refinement operations for background blur."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy import ndimage


def _ellipse(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _subject_mask(alpha: np.ndarray, threshold: float, expand_px_at_4k: float) -> np.ndarray:
    mask = alpha >= float(threshold)
    scale = max(alpha.shape) / 4000.0
    expand = max(0, round(float(expand_px_at_4k) * scale))
    if expand > 0:
        mask = cv2.dilate(mask.astype(np.uint8), _ellipse(expand), iterations=1).astype(bool)
    return mask


def background_exclusion_mask(alpha: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Return the RGB-plate subject mask using the plate's small safety expansion."""
    pc = cfg["plate"]
    return _subject_mask(
        alpha,
        float(pc.get("foreground_threshold", 0.01)),
        float(pc.get("expand_px_at_4k", 10)),
    )


def depth_exclusion_mask(alpha: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Return the wider guard used to reject foreground-depth edge bleed."""
    pc = cfg["plate"]
    dc = cfg.get("depth", {})
    return _subject_mask(
        alpha,
        float(pc.get("foreground_threshold", 0.01)),
        float(dc.get("edge_guard_px_at_4k", pc.get("expand_px_at_4k", 10))),
    )


def make_background_depth(depth: np.ndarray, alpha: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Fill subject, edge-guard and invalid depth from nearest true background."""
    excluded = depth_exclusion_mask(alpha, cfg)
    valid = np.isfinite(depth) & (depth > 1e-4)
    seeds = (~excluded) & valid

    if np.all(seeds):
        return depth.astype(np.float32, copy=True)
    if not np.any(seeds):
        fallback = float(np.median(depth[valid])) if np.any(valid) else 1.0
        return np.where(valid, depth, fallback).astype(np.float32)

    fill = ~seeds
    inds = ndimage.distance_transform_edt(fill, return_distances=False, return_indices=True)
    out = depth.astype(np.float32, copy=True)
    out[fill] = depth[inds[0][fill], inds[1][fill]]
    return out


def constrain_refined_alpha(base_alpha: np.ndarray, refined_alpha: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Keep ViTMatte as a refiner of BiRefNet rather than a new segmenter.

    On high-contrast edges ViTMatte can otherwise grow several pixels into the
    old background and even mark those pixels almost opaque. They become a dark
    or colored outline after the background is blurred. The base BiRefNet matte
    is the trusted silhouette support; ViTMatte may refine alpha inside it, with
    only an explicitly configured amount of outward growth.
    """
    base = np.clip(base_alpha, 0.0, 1.0).astype(np.float32)
    refined = np.clip(refined_alpha, 0.0, 1.0).astype(np.float32)
    vc = cfg.get("vitmatte", {})
    scale = max(base.shape) / 2048.0
    expand = max(0, round(float(vc.get("max_expand_px_at_2048", 0)) * scale))
    if expand > 0:
        support = cv2.dilate(base, _ellipse(expand), iterations=1)
    else:
        support = base
    return np.minimum(refined, support).astype(np.float32)


def install(impl) -> None:
    """Install boundary-safe callbacks into the mature background_blur module."""
    original_vitmatte_refine = impl.vitmatte_refine

    def vitmatte_refine(proc, model, device: str, image, base_alpha: np.ndarray, cfg: dict):
        refined, trimap = original_vitmatte_refine(proc, model, device, image, base_alpha, cfg)
        return constrain_refined_alpha(base_alpha, refined, cfg), trimap

    def build_blur_map(depth: np.ndarray, alpha: np.ndarray, preset: dict, cfg: dict):
        # Original depth is authoritative only for the subject focal distance.
        # Blur itself is driven by a subject-free depth field. Alpha is reserved
        # for the final composite and is not used to suppress blur a second time.
        focus, _ = impl.subject_focus_depth(depth, alpha, cfg)
        background_depth = make_background_depth(depth, alpha, cfg)
        valid = np.isfinite(background_depth) & (background_depth > 1e-4)
        z = np.where(valid, background_depth, focus).astype(np.float32)
        inv = 1.0 / np.maximum(z, 1e-4)
        inv_focus = 1.0 / max(focus, 1e-4)
        delta = np.abs(inv - inv_focus)

        threshold = float(cfg["plate"].get("foreground_threshold", 0.01))
        bg = valid & (alpha < threshold)
        vals = delta[bg]
        if vals.size < 64:
            vals = delta[valid]
        denom = (
            float(np.percentile(vals, float(cfg["blur"].get("depth_normalization_percentile", 96.0))))
            if vals.size else 1.0
        )
        denom = max(denom, 1e-6)
        normalized = delta / denom

        tol = float(preset.get("focus_tolerance", 0.04))
        amount = np.clip((normalized - tol) / max(1.0 - tol, 1e-6), 0.0, 1.0)
        amount = np.power(amount, float(preset.get("gamma", 1.15)))
        amount *= float(preset.get("strength", 0.72))
        return np.clip(amount, 0, 1).astype(np.float32), focus

    def uniform_blur_map(alpha: np.ndarray, strength: float) -> np.ndarray:
        # The plate contains no subject. Subject protection happens exactly once
        # in the final alpha composite.
        return np.full(alpha.shape, np.clip(float(strength), 0.0, 1.0), dtype=np.float32)

    impl.vitmatte_refine = vitmatte_refine
    impl.build_blur_map = build_blur_map
    impl.uniform_blur_map = uniform_blur_map
