#!/usr/bin/env python3
"""Fast local test for geometry + registration. Downloads no AI models."""

import math
import cv2
import numpy as np
from PIL import Image, ImageDraw

from remove_text import register_edit_to_source, work_size


def synthetic_scene(w=960, h=720):
    im = Image.new("RGB", (w, h), (125, 135, 145))
    d = ImageDraw.Draw(im)
    for x in range(30, w, 80):
        d.line((x, 0, x, h), fill=(65 + x % 90, 80, 105), width=3)
    for y in range(35, h, 70):
        d.line((0, y, w, y), fill=(95, 70 + y % 100, 85), width=2)
    d.ellipse((220, 90, 710, 650), outline=(235, 210, 160), width=15)
    d.rectangle((80, 500, 880, 560), fill=(60, 80, 65))
    d.text((360, 330), "SOURCE", fill=(250, 250, 250))
    return im


def main():
    # Work-size sanity.
    a = work_size((1200, 900), 1.0, 32)
    assert a[0] % 32 == 0 and a[1] % 32 == 0
    assert abs((a[0] / a[1]) - (4 / 3)) < 0.04

    src = synthetic_scene()
    arr = np.asarray(src)
    h, w = arr.shape[:2]

    # Simulate editor drift: +1.2% scale, +0.25 degrees and a few pixels shift.
    angle = math.radians(0.25)
    scale = 1.012
    M = np.array([
        [scale * math.cos(angle), -scale * math.sin(angle), 7.0],
        [scale * math.sin(angle),  scale * math.cos(angle), -5.0],
    ], dtype=np.float32)
    drifted = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)
    edt = Image.fromarray(drifted)

    # Pretend center is the repair region; registration must ignore it.
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (340, 285), (620, 430), 255, -1)

    aligned, meta = register_edit_to_source(src, edt, mask, {
        "enabled": True,
        "exclude_mask_expand_px": 40,
        "min_stable_fraction": 0.20,
        "sift_features": 4000,
        "sift_contrast_threshold": 0.01,
        "match_ratio": 0.80,
        "min_matches": 8,
        "ransac_reproj_px": 3.0,
        "ransac_max_iters": 3000,
        "ransac_confidence": 0.995,
        "max_scale_delta": 0.05,
        "max_rotation_deg": 2.0,
        "max_translation_fraction": 0.08,
        "min_inlier_ratio": 0.25,
        "min_mae_improvement": 0.02,
    })
    assert meta.get("accepted"), meta
    assert float(meta["mae_after"]) < float(meta["mae_before"]), meta
    assert aligned.size == src.size
    print("PASS")
    print("work_size(1200x900):", a)
    print("registration:", {k: meta.get(k) for k in ("scale", "rotation_deg", "translation_x_px", "translation_y_px", "mae_before", "mae_after")})


if __name__ == "__main__":
    main()
