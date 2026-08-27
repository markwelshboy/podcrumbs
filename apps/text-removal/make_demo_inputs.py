#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="demo_inputs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for i, size in enumerate([(2400, 1600), (1600, 2400), (3200, 1800)]):
        w, h = size
        im = Image.new("RGB", size, (116 + i * 20, 143, 126 + i * 15))
        d = ImageDraw.Draw(im)
        # Fake scene structure so repair has something non-uniform behind text.
        for y in range(0, h, max(20, h // 24)):
            shade = int(35 * y / h)
            d.rectangle((0, y, w, min(h, y + h // 24)), fill=(110 + shade, 130 + shade // 2, 120))
        d.ellipse((w * .28, h * .12, w * .72, h * .92), fill=(178, 145, 126))
        d.rectangle((w * .05, h * .72, w * .95, h * .78), fill=(75, 82, 70))

        text = "SAMPLE WATERMARK  © 2026"
        # Default font is intentionally portable; OCR difficulty is useful here.
        bbox = d.textbbox((0, 0), text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        scale = max(1, w // max(1, tw * 3))
        # Draw repeated default-font text to produce a larger readable overlay.
        x, y = int(w * .48), int(h * .84)
        # Shadow + translucent-ish approximation on RGB.
        for dx, dy in [(3, 3), (2, 2)]:
            d.text((x + dx, y + dy), text, fill=(25, 25, 25))
        d.text((x, y), text, fill=(245, 245, 245))
        d.text((int(w * .03), int(h * .06)), "@example_creator", fill=(240, 240, 240))
        im.save(out / f"demo_{i:02d}.jpg", quality=95)

    print(f"Wrote demo images to {out}")

if __name__ == "__main__":
    main()
