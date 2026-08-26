#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from podcrumbs_controls import add_control_arguments, load_control_definitions, load_yaml, record_run_contract, resolve_controls, write_runtime_config


def main() -> int:
    structural = ROOT / "config.yaml"
    controls_path = ROOT / "controls.yaml"
    definitions = load_control_definitions(controls_path)

    parser = argparse.ArgumentParser(description="Depth-aware Background Blur")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--structural-config", type=Path, default=structural, help=argparse.SUPPRESS)
    add_control_arguments(parser, definitions)
    args = parser.parse_args()
    resolved = resolve_controls(args, definitions)

    cfg = copy.deepcopy(load_yaml(args.structural_config))
    cfg.setdefault("input", {})["recursive"] = bool(resolved["recursive"])
    cfg.setdefault("fbcnn", {})["mode"] = resolved["fbcnn"]
    cfg["fbcnn"]["qf"] = resolved["fbcnn_qf"]
    cfg.setdefault("matte", {})["mode"] = resolved["matte"]

    output = args.output.resolve()
    record_run_contract(
        app_name="background-blur", app_dir=ROOT, output_dir=output,
        structural_config=args.structural_config.resolve(), controls_definition=controls_path,
        resolved=resolved,
    )
    runtime_cfg = write_runtime_config(cfg)
    try:
        cmd = [
            sys.executable, str(ROOT / "background_blur.py"), str(args.input), str(args.output),
            "--config", str(runtime_cfg), "--preset", str(resolved["preset"]),
        ]
        if resolved["compare"]:
            cmd.append("--compare")
        if int(resolved["limit"]) > 0:
            cmd += ["--limit", str(resolved["limit"])]
        if resolved["force"]:
            cmd.append("--force")
        return subprocess.call(cmd)
    finally:
        runtime_cfg.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
