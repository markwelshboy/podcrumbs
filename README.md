# podcrumbs

`podcrumbs` is the application catalog for [Podlets](https://github.com/markwelshboy/podlets): small GPU image-processing tools, their structural configs, declared controls, bootstrap logic, and optional local companion apps.

The split is deliberate:

- **podlets** owns job orchestration, staging, lifecycle, logs, GPU gating and fetch.
- **pod-runtime** owns generic worker/bootstrap/Hugging Face helpers.
- **podcrumbs** owns runnable applications.

## App contract

Every Podcrumb separates three concerns:

```text
app.yaml        identity, entrypoints, artifacts and companions
config.yaml     structural implementation/capabilities
controls.yaml   explicitly declared user-facing knobs and their defaults
```

`config.yaml` is code-facing. It contains model IDs, backend definitions, pipeline structure, blur preset definitions, registration settings, OCR machinery, etc. Changing it generally means changing what the application can do and may require matching code changes.

`controls.yaml` is the normal run surface. A backend or structural option is not automatically user-selectable merely because it exists in `config.yaml`; it must be deliberately exposed as a control.

The thin app `run.py` resolves controls, creates the runtime configuration expected by the underlying harness, and records the run contract in the output directory:

```text
structural-config.yaml
controls-definition.yaml
resolved-controls.yaml
podcrumb-run.json
```

This makes fetched results reproducible even if application defaults later change.

## Initial apps

| `sl` command | App | Purpose |
|---|---|---|
| `bg-remove` | `apps/background-removal` | Compare RMBG-2.0, BiRefNet, ViTMatte and BEN2 foreground extraction paths. |
| `bg-blur` | `apps/background-blur` | Matte-protected, Depth Pro-driven background blur with optional FBCNN restoration. |
| `text-remove` | `apps/text-removal` | OCR-guided text/watermark removal using Qwen, FireRed or FLUX.2 Klein editors. |

## Discovering and inspecting commands

With `podcrumbs` checked out at `~/git/podcrumbs`, a Podlets build with catalog discovery finds the commands automatically. A non-standard location can still be configured with:

```bash
sl config command-dir /path/to/podcrumbs/commands
```

Inspect the declared run surface without touching a GPU worker:

```bash
sl commands
sl command help bg-remove
sl command controls bg-remove
sl command config bg-remove
sl command show bg-remove
```

The distinction is intentional:

- `help` renders normal user controls.
- `controls` shows the declaration/defaults.
- `config` shows the structural implementation config.
- `show` shows the low-level Podlets `.cmd` adapter.

## Examples

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
  --editors qwen firered klein
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
├── podcrumbs_controls.py     # shared control resolution/provenance helper
├── commands/                 # thin `sl` adapters
├── apps/
│   ├── background-removal/
│   ├── background-blur/
│   └── text-removal/
└── bin/crumb                 # local companion launcher
```
