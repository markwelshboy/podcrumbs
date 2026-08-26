#!/usr/bin/env python3
from pathlib import Path
import tempfile
import numpy as np
import cv2
from PIL import Image

import background_blur as bb
from background_blur_ops import (
    constrain_refined_alpha,
    depth_exclusion_mask,
    install,
    make_background_depth,
)

install(bb)


def main():
    # Synthetic scene: foreground red ellipse over horizontal-detail background.
    h, w = 480, 640
    yy, xx = np.mgrid[:h, :w]
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0] = (xx * 255 // w).astype(np.uint8)
    rgb[..., 1] = ((yy // 8) % 2 * 120 + 60).astype(np.uint8)
    rgb[..., 2] = 180
    alpha = np.zeros((h, w), np.float32)
    subject = (((xx - 320) / 105) ** 2 + ((yy - 230) / 175) ** 2) <= 1
    alpha[subject] = 1.0
    rgb[subject] = [210, 55, 45]

    cfg = {
        "plate": {"foreground_threshold": 0.01, "expand_px_at_4k": 8},
        "depth": {"edge_guard_px_at_4k": 64},
        "vitmatte": {"max_expand_px_at_2048": 0},
        "blur": {
            "subject_core_alpha": 0.95,
            "subject_core_erode_px_at_4k": 20,
            "depth_normalization_percentile": 96.0,
        },
    }
    plate = bb.make_background_plate(rgb, alpha, cfg)
    assert not np.all(plate[230, 320] == np.array([210, 55, 45], np.uint8))

    # A refiner may change alpha inside BiRefNet support, but with zero expansion
    # it must not invent new foreground outside that support.
    refined = alpha.copy()
    outside = cv2.dilate(subject.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1).astype(bool) & ~subject
    refined[outside] = 0.9
    constrained = constrain_refined_alpha(alpha, refined, cfg)
    assert np.all(constrained[outside] == 0.0)
    assert np.all(constrained[subject] == 1.0)

    # Metric depth plane: subject at 2m, background varying from 2m to 8m.
    depth = 2.0 + (xx.astype(np.float32) / w) * 6.0
    depth[subject] = 2.0

    # Simulate a depth-estimator halo OUTSIDE the alpha silhouette. The wider
    # depth guard should reject this too, not merely the subject interior.
    halo = cv2.dilate(subject.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool) & ~subject
    depth[halo] = 2.0

    preset = {"strength": 0.8, "focus_tolerance": 0.04, "gamma": 1.1}
    guard = depth_exclusion_mask(alpha, cfg)
    background_depth = make_background_depth(depth, alpha, cfg)
    bmap, focus = bb.build_blur_map(depth, alpha, preset, cfg)
    assert 1.8 < focus < 2.2

    assert np.count_nonzero(guard & ~subject) > 0
    guarded_halo = halo & guard
    assert np.any(guarded_halo)
    assert np.all(background_depth[guarded_halo] > 2.0)

    assert background_depth[230, 320] > 2.0
    assert bmap[230, 320] > 0.0
    assert np.mean(bmap[guarded_halo]) > 0.0
    assert bmap[240, 620] > bmap[240, 400]

    # The comparison blur is genuinely uniform across the subject-free plate.
    ub = bb.uniform_blur_map(alpha, 0.72)
    assert np.allclose(ub, 0.72)

    out = bb.variable_gaussian_blur(plate, bmap, 18, [0, .1, .25, .5, .75, 1])
    final = bb.composite_subject(rgb, out, alpha)
    assert np.all(final[230, 320] == rgb[230, 320])
    assert final.shape == rgb.shape

    assert bb.fbcnn_should_run(Path("a.jpg"), "auto")
    assert not bb.fbcnn_should_run(Path("a.png"), "auto")
    assert bb.fbcnn_should_run(Path("a.png"), "on")
    assert not bb.fbcnn_should_run(Path("a.jpg"), "off")

    with tempfile.TemporaryDirectory() as td:
        Image.fromarray(final).save(Path(td) / "smoke.png")

    print("Smoke test: PASS")
    print(f"Focus depth recovered: {focus:.3f} m")
    print(f"Blur map range: {bmap.min():.3f} .. {bmap.max():.3f}")


if __name__ == "__main__":
    main()
