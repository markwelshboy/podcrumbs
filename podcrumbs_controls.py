#!/usr/bin/env python3
"""Shared control contract helpers for Podcrumb applications."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def load_control_definitions(path: Path) -> dict[str, dict[str, Any]]:
    data = load_yaml(path)
    controls = data.get("controls")
    if not isinstance(controls, dict):
        raise ValueError(f"controls.yaml must contain a controls mapping: {path}")
    out: dict[str, dict[str, Any]] = {}
    for name, raw in controls.items():
        if not isinstance(raw, dict):
            raise ValueError(f"control {name!r} must be a mapping")
        spec = dict(raw)
        if not isinstance(spec.get("flag"), str) or not spec["flag"].startswith("--"):
            raise ValueError(f"control {name!r} must declare a --flag")
        if "default" not in spec:
            raise ValueError(f"control {name!r} must declare a default")
        kind = str(spec.get("type", "string"))
        if kind == "flag":
            if spec["default"] is not False:
                raise ValueError(f"flag control {name!r} must default to false; use type: toggle for a switchable true default")
            if spec.get("negative_flag"):
                raise ValueError(f"flag control {name!r} cannot declare negative_flag; use type: toggle")
        elif kind == "toggle":
            if not isinstance(spec["default"], bool):
                raise ValueError(f"toggle control {name!r} must have a boolean default")
            negative = spec.get("negative_flag")
            if not isinstance(negative, str) or not negative.startswith("--"):
                raise ValueError(f"toggle control {name!r} must declare a --negative_flag")
        out[str(name)] = spec
    return out


def add_control_arguments(parser: argparse.ArgumentParser, definitions: Mapping[str, Mapping[str, Any]]) -> None:
    for name, spec in definitions.items():
        flag = str(spec["flag"])
        kind = str(spec.get("type", "string"))
        help_text = str(spec.get("help", ""))
        metavar = spec.get("metavar")
        choices = spec.get("choices")
        kwargs: dict[str, Any] = {"dest": name, "default": None, "help": help_text}
        if metavar:
            kwargs["metavar"] = str(metavar)
        if isinstance(choices, list):
            kwargs["choices"] = [str(x) for x in choices]

        if kind == "flag":
            parser.add_argument(flag, dest=name, action="store_true", default=None, help=help_text)
            continue
        if kind == "toggle":
            group = parser.add_mutually_exclusive_group()
            group.add_argument(flag, dest=name, action="store_true", default=None, help=help_text)
            group.add_argument(str(spec["negative_flag"]), dest=name, action="store_false", help=argparse.SUPPRESS)
            continue
        if kind == "integer":
            kwargs["type"] = int
        elif kind == "number":
            kwargs["type"] = float
        elif kind == "multi_choice":
            kwargs["nargs"] = "+"
        elif kind != "string" and kind != "choice":
            raise ValueError(f"unsupported control type {kind!r} for {name}")
        parser.add_argument(flag, **kwargs)


def resolve_controls(args: argparse.Namespace, definitions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for name, spec in definitions.items():
        value = copy.deepcopy(spec.get("default"))
        override = getattr(args, name, None)
        if override is not None:
            value = override
        choices = spec.get("choices")
        if isinstance(choices, list):
            allowed = {str(x) for x in choices}
            values = value if isinstance(value, list) else [value]
            bad = [x for x in values if str(x) not in allowed]
            if bad:
                raise ValueError(f"invalid {name}: {bad}; choices: {sorted(allowed)}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if spec.get("minimum") is not None and value < spec["minimum"]:
                raise ValueError(f"{name} must be >= {spec['minimum']}")
            if spec.get("maximum") is not None and value > spec["maximum"]:
                raise ValueError(f"{name} must be <= {spec['maximum']}")
        resolved[name] = value
    return resolved


def write_runtime_config(config: Mapping[str, Any]) -> Path:
    work = os.environ.get("SL_WORK_DIR")
    directory = Path(work) if work else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="podcrumb-runtime-", dir=directory, delete=False, encoding="utf-8")
    try:
        yaml.safe_dump(dict(config), fh, sort_keys=False)
        return Path(fh.name)
    finally:
        fh.close()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip() or None
    except Exception:
        return None


def record_run_contract(
    *, app_name: str, app_dir: Path, output_dir: Path, structural_config: Path,
    controls_definition: Path, resolved: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(structural_config, output_dir / "structural-config.yaml")
    shutil.copyfile(controls_definition, output_dir / "controls-definition.yaml")
    with (output_dir / "resolved-controls.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(resolved), f, sort_keys=False)
    repo_root = app_dir.parents[1]
    metadata = {
        "schema": 1,
        "app": app_name,
        "podcrumbs_ref": os.environ.get("PODCRUMBS_REF", "main"),
        "podcrumbs_commit": _git_commit(repo_root),
        "structural_config_sha256": _sha256(structural_config),
        "controls_definition_sha256": _sha256(controls_definition),
    }
    (output_dir / "podcrumb-run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
