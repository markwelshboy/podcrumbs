# podcrumbs

`podcrumbs` is the application catalog for [Podlets](https://github.com/markwelshboy/podlets): small GPU image-processing tools, their configs/bootstrap logic, and optional local companion apps.

The split is deliberate:

- **podlets** owns job orchestration, staging, lifecycle, logs, GPU gating and fetch.
- **pod-runtime** owns generic worker/bootstrap/Hugging Face helpers.
- **podcrumbs** owns runnable applications.

## Initial apps

| `sl` command | App | Purpose |
|---|---|---|
| `bg-remove` | `apps/background-removal` | Compare RMBG-2.0, BiRefNet, ViTMatte and BEN2 foreground extraction paths. |
| `bg-blur` | `apps/background-blur` | Matte-protected, Depth Pro-driven background blur with optional FBCNN restoration. |
| `text-remove` | `apps/text-removal` | OCR-guided text/watermark removal using Qwen, FireRed or FLUX.2 Klein editors. |

Each app keeps its original harness config, smoke test and bootstrap logic. Podlet adapters live in `commands/` and are intentionally thin.

## Podlets usage

With `podcrumbs` checked out at `~/git/podcrumbs`, a Podlets build with catalog discovery will find these commands automatically. Until then, or from a non-standard checkout path:

```bash
sl config command-dir ~/git/podcrumbs/commands
```

Examples:

```bash
sl run bg-remove \
  ~/images/input \
  bg-remove-01/ \
  --output-dir ~/results \
  -- \
  --methods rmbg2 birefnet_vitmatte ben2_refined

sl run bg-blur \
  ~/images/input \
  bg-blur-01/ \
  --output-dir ~/results \
  -- \
  --fbcnn on --compare

sl run text-remove \
  ~/images/input \
  text-remove-01/ \
  --output-dir ~/results \
  -- \
  --compare-editors
```

Remote app environments are isolated under `/workspace/.sl/cache/podcrumbs/envs/`. The apps share `/workspace/.sl/cache/huggingface` so common models such as BiRefNet and ViTMatte are not downloaded independently for every tool.

## Local companions

Companions run after GPU output has been fetched, so the expensive worker does not need to remain alive.

Background-removal review/export:

```bash
python -m pip install -r apps/background-removal/companions/requirements.txt
./bin/crumb bg-remove review ~/results/bg-remove-01
```

The review app compares successful foreground methods over black/white backgrounds, remembers browser selections locally, and exports selected full-resolution PNGs as a `.tar.gz`.

## Layout

```text
podcrumbs/
├── commands/                 # thin `sl` adapters
├── apps/
│   ├── background-removal/
│   ├── background-blur/
│   └── text-removal/
└── bin/crumb                 # local companion launcher
```

`app.yaml` is descriptive metadata for now. The first version deliberately does not build a plugin framework around it; the convention can harden after these apps have been used in anger.
