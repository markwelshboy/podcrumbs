#!/usr/bin/env python3
"""Offline smoke tests; downloads no models."""
import numpy as np
from PIL import Image

from compare_backgrounds import (
    build_trimap,
    composite_rgb,
    foreground_estimate_blur_fusion,
    make_checkerboard,
)


def main():
    h, w = 300, 240
    yy, xx = np.mgrid[:h, :w]
    alpha = np.clip(1.0 - np.sqrt(((xx-w/2)/(w*.35))**2 + ((yy-h/2)/(h*.42))**2), 0, 1).astype(np.float32)
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0] = np.linspace(40, 220, w, dtype=np.uint8)[None, :]
    rgb[..., 1] = 120
    rgb[..., 2] = 180

    trimap = build_trimap(alpha, 0.90, 0.03, 4, 8)
    vals = set(np.unique(trimap).tolist())
    assert vals.issubset({0, 128, 255}) and 128 in vals

    refined = foreground_estimate_blur_fusion(rgb, alpha, 25, 3)
    assert refined.shape == rgb.shape and refined.dtype == np.uint8

    out = composite_rgb(refined, alpha, (128, 128, 128))
    assert out.shape == rgb.shape

    cb = make_checkerboard(w, h, 24)
    assert cb.size == (w, h)
    print('Smoke tests: PASS')

if __name__ == '__main__':
    main()
