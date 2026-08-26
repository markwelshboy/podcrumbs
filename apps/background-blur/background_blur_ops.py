#!/usr/bin/env python3
"""Boundary-safe blur-map operations for the background-blur harness."""
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
    """Return the wider guard used to reject foreground-depth edge bleed.

    Depth estimators can extend the subject's focal-depth values beyond the
    actual alpha edge.  That contamination must not seed the background-only
    depth field or it produces a narrow low-blur band around one side of a
    silhouette.  The depth guard is deliberately wider than the RGB plate guard.
    """
    pc = cfg["plate"]
    dc = cfg.get("depth", {})
    return _subject_mask(
        alpha,
        float(pc.get("foreground_threshold", 0.01)),
        float(dc.get("edge_guard_px_at_4k", pc.get("expand_px_at_4k", 10))),
    )


def make_background_depth(depth: np.ndarray, alpha: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Fill subject, edge-guard and invalid depth from nearest true background.

    The original depth map remains authoritative for estimating the subject's
    focal distance.  Blur control uses this cleaned field, so both the subject
    itself and any small Depth-Pro halo outside its matte are replaced by nearby
    background depth before blur strength is calculated.
    """
    excluded = depth_exclusion_mask(alpha, cfg)
    valid = np.isfinite(depth) & (depth > 1e-4)
    seeds = (~excluded) & valid

    if np.all(seeds):
        return depth.astype(np.float32, copy=True)
    if not np.any(seeds):
        # No trustworthy background depth exists (for example, the subject fills
        # the frame). Preserve finite depth rather than invent spatial structure.
        fallback = float(np.median(depth[valid])) if np.any(valid) else 1.0
        return np.where(valid, depth, fallback).astype(np.float32)

    fill = ~seeds
    inds = ndimage.distance_transform_edt(fill, return_distances=False, return_indices=True)
    out = depth.astype(np.float32, copy=True)
    out[fill] = depth[inds[0][fill], inds[1][fill]]
    return out


def install(impl) -> None:
    """Install boundary-safe blur-map callbacks into background_blur module."""

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

        # Normalize from actual background pixels. Propagated under-subject depth
        # must not redefine the scene's blur scale.
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
        # The plate contains no subject. Uniform blur should therefore really be
        # uniform; subject protection happens once in the final alpha composite.
        return np.full(alpha.shape, np.clip(float(strength), 0.0, 1.0), dtype=np.float32)

    impl.build_blur_map = build_blur_map
    impl.uniform_blur_map = uniform_blur_map
