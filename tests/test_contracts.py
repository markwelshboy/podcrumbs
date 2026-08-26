from __future__ import annotations

import argparse
import unittest
from pathlib import Path

import yaml

from podcrumbs_controls import add_control_arguments, load_control_definitions, resolve_controls

ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def load(self, app: str):
        root = ROOT / "apps" / app
        config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
        controls = load_control_definitions(root / "controls.yaml")
        manifest = yaml.safe_load((root / "app.yaml").read_text(encoding="utf-8"))
        return root, config, controls, manifest

    def test_every_app_declares_structural_config_and_controls(self):
        for app in ("background-removal", "background-blur", "text-removal"):
            with self.subTest(app=app):
                root, _, _, manifest = self.load(app)
                self.assertEqual(manifest["config"], "config.yaml")
                self.assertEqual(manifest["controls"], "controls.yaml")
                self.assertTrue((root / manifest["config"]).is_file())
                self.assertTrue((root / manifest["controls"]).is_file())
                self.assertTrue((root / manifest["remote"]["entrypoint"]).is_file())

    def test_background_removal_only_exposes_structurally_available_methods(self):
        _, config, controls, _ = self.load("background-removal")
        self.assertTrue(set(controls["methods"]["choices"]).issubset(config["methods"]))
        self.assertTrue(set(controls["methods"]["default"]).issubset(controls["methods"]["choices"]))
        self.assertNotIn("recursive", config["input"])

    def test_background_blur_choices_match_structural_capabilities(self):
        _, config, controls, _ = self.load("background-blur")
        self.assertTrue(set(controls["matte"]["choices"]).issubset(config["matte"]["modes"]))
        self.assertTrue(set(controls["preset"]["choices"]).issubset(config["blur"]["presets"]))
        self.assertNotIn("mode", config["fbcnn"])
        self.assertNotIn("qf", config["fbcnn"])
        self.assertNotIn("mode", config["matte"])

    def test_text_editor_controls_match_structural_backends(self):
        _, config, controls, _ = self.load("text-removal")
        self.assertTrue(set(controls["editors"]["choices"]).issubset(config["editors"]))
        self.assertNotIn("benchmark", config)
        self.assertNotIn("recursive", config["input"])
        for editor in config["editors"].values():
            self.assertNotIn("seed", editor)

    def test_control_parser_uses_declared_defaults_and_overrides(self):
        _, _, controls, _ = self.load("background-removal")
        parser = argparse.ArgumentParser()
        add_control_arguments(parser, controls)
        args = parser.parse_args(["--methods", "rmbg2", "--limit", "2", "--no-recursive"])
        resolved = resolve_controls(args, controls)
        self.assertEqual(resolved["methods"], ["rmbg2"])
        self.assertEqual(resolved["limit"], 2)
        self.assertFalse(resolved["recursive"])


if __name__ == "__main__":
    unittest.main()
