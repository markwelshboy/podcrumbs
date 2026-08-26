#!/usr/bin/env python3
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image

import background_blur as bb


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
        "blur": {
            "subject_core_alpha": 0.95,
            "subject_core_erode_px_at_4k": 20,
            "protect_subject": True,
            "depth_normalization_percentile": 96.0,
        },
    }
    plate = bb.make_background_plate(rgb, alpha, cfg)
    # Center should no longer be the red subject colour.
    assert not np.all(plate[230, 320] == np.array([210, 55, 45], np.uint8))

    # Metric depth plane: subject at 2m, background varying from 2m to 8m.
    depth = 2.0 + (xx.astype(np.float32) / w) * 6.0
    depth[subject] = 2.0
    preset = {"strength": 0.8, "focus_tolerance": 0.04, "gamma": 1.1}
    bmap, focus = bb.build_blur_map(depth, alpha, preset, cfg)
    assert 1.8 < focus < 2.2
    assert bmap[230, 320] == 0.0
    assert bmap[240, 620] > bmap[240, 400]

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
