#!/usr/bin/env python3
"""
Interactive selector/exporter for background-removal-harness output.

The script IS the web server; no Flask/FastAPI dependency is required.
Pillow is the only non-stdlib dependency.

Usage:
    python bg_output_selector_v3_1.py /workspace/bg_compare --host 127.0.0.1 --port 8090

Features:
  * compare successful background-removal methods for each source image
  * show each foreground over a split BLACK | WHITE preview
  * choose the preferred method per source image
  * independently choose WHITE or BLACK as that image's exported background
  * interactively hide/show method "flavors" in the browser
  * stream selected full-resolution RGB PNGs directly as a .tar.gz
  * remember selections, background choices, flavor visibility, and per-image detail level in localStorage
"""

from __future__ import annotations

import argparse
import html
import io
import json
import os
import tarfile
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

from PIL import Image


METHOD_LABELS = {
    "rmbg2": "RMBG-2.0",
    "rmbg2_vitmatte": "RMBG-2.0 → ViTMatte-B",
    "birefnet_hr": "BiRefNet HR matte",
    "birefnet_hr_refined": "BiRefNet HR + FG refine",
    "birefnet_vitmatte": "BiRefNet HR → ViTMatte-B",
    "ben2_base": "BEN2 Base",
    "ben2_refined": "BEN2 Base + refine",
}


CSS = r'''
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; background:#121212; color:#eee; font:14px system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
header { position:sticky; top:0; z-index:20; background:rgba(18,18,18,.97); backdrop-filter:blur(8px); border-bottom:1px solid #3b3b3b; padding:12px 18px; }
header h1 { font-size:20px; margin:0 0 4px; }
header .hint { color:#aaa; }
.flavor-panel { margin-top:10px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.flavor-title { color:#bbb; font-weight:650; margin-right:3px; }
.flavor-chip { display:flex; align-items:center; gap:5px; padding:5px 8px; border-radius:7px; background:#242424; border:1px solid #3d3d3d; cursor:pointer; user-select:none; }
.flavor-chip input { width:15px; height:15px; margin:0; }
.flavor-actions { display:flex; gap:5px; margin-left:4px; }
.flavor-actions button { padding:5px 8px; }
.detail-panel { margin-top:9px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.detail-title { color:#bbb; font-weight:650; }
.detail-select { background:#252525; color:#eee; border:1px solid #4a4a4a; border-radius:6px; padding:5px 8px; font:inherit; }
.detail-help { color:#888; font-size:12px; }
.row-titlebar { display:flex; gap:10px; align-items:center; justify-content:space-between; flex-wrap:wrap; margin-bottom:12px; }
.row-titlebar h2 { margin:0; }
.row-detail-wrap { display:flex; align-items:center; gap:6px; color:#aaa; font-size:12px; }
main { padding:12px 18px 110px; }
.image-row { border-bottom:1px solid #353535; padding:20px 0 26px; }
.image-row h2 { font-size:16px; margin:0; overflow-wrap:anywhere; }
.row-grid { display:grid; grid-template-columns:180px minmax(0,1fr); gap:14px; align-items:start; }
.candidates-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; align-items:start; min-width:0; }
.image-row[data-detail="mid"] .candidates-grid { grid-template-columns:repeat(auto-fit,minmax(390px,1fr)); }
.image-row[data-detail="full"] .candidates-grid { grid-template-columns:repeat(auto-fit,minmax(500px,1fr)); }
.source-card, .candidate { background:#202020; border:1px solid #353535; border-radius:10px; overflow:hidden; }
.source-card img, .candidate img { width:100%; height:280px; object-fit:contain; display:block; background:#0c0c0c; }
.candidate .mid-view img, .candidate .full-view img { height:auto; max-height:none; }
.basic-view, .mid-view, .full-view { display:none; }
.image-row[data-detail="basic"] .basic-view { display:block; }
.image-row[data-detail="mid"] .mid-view { display:block; }
.image-row[data-detail="full"] .mid-view, .image-row[data-detail="full"] .full-view { display:block; }
.image-row[data-detail="full"] .full-view { border-top:1px solid #3a3a3a; }
.source-card img { height:220px; }
.caption { padding:9px 10px; }
.source-card .caption { color:#bbb; }
.candidate { transition:border-color .1s, box-shadow .1s, transform .05s; position:relative; }
.candidate:hover { border-color:#777; }
.candidate.selected { border-color:#45a3ff; box-shadow:0 0 0 2px rgba(69,163,255,.25); }
.pick-line { display:flex; align-items:center; gap:7px; }
.pick-line > input { width:19px; height:19px; margin:0; flex:0 0 auto; }
.method-name { font-weight:650; }
.method-meta { color:#aaa; font-size:12px; margin-top:4px; }
.split-note { position:absolute; right:8px; top:8px; background:rgba(0,0,0,.68); padding:3px 6px; border-radius:5px; font-size:11px; color:#ddd; }
.bg-choice { display:flex; gap:7px; margin-top:9px; }
.bg-option { flex:1; display:flex; align-items:center; justify-content:center; gap:5px; padding:6px 7px; border-radius:6px; border:1px solid #444; cursor:pointer; user-select:none; }
.bg-option input { margin:0; width:15px; height:15px; }
.bg-option.white { background:#eee; color:#111; }
.bg-option.black { background:#050505; color:#eee; }
.bg-option:has(input:checked) { outline:2px solid #45a3ff; outline-offset:1px; }
.candidate.hidden-flavor { display:none; }
.toolbar { position:fixed; z-index:30; left:0; right:0; bottom:0; padding:12px 18px; background:rgba(22,22,22,.97); border-top:1px solid #444; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
button { border:0; border-radius:7px; padding:9px 13px; font:inherit; cursor:pointer; }
button.primary { background:#1687ea; color:white; font-weight:650; }
button.secondary { background:#393939; color:#eee; }
button:disabled { opacity:.45; cursor:not-allowed; }
.count { margin-left:auto; color:#bbb; }
.toggle { color:#bbb; display:flex; align-items:center; gap:6px; }
.toggle input { width:17px; height:17px; }
.empty { padding:40px; color:#aaa; }
@media (max-width:800px) {
  .row-grid { grid-template-columns:1fr; }
  .candidates-grid, .image-row[data-detail="mid"] .candidates-grid, .image-row[data-detail="full"] .candidates-grid { grid-template-columns:1fr; }
  .source-card img, .candidate img { height:auto; max-height:75vh; }
  .count { margin-left:0; }
}
'''


JS = r'''
const SELECT_KEY = 'bg-selector-v2:selection:' + ROOT_ID;
const FLAVOR_KEY = 'bg-selector-v2:flavors:' + ROOT_ID;
const DETAIL_KEY = 'bg-selector-v3:details:' + ROOT_ID;
const boxes = [...document.querySelectorAll('.candidate .candidate-pick')];
const flavorBoxes = [...document.querySelectorAll('.flavor-filter')];
const onePer = document.getElementById('onePerImage');
const globalDetail = document.getElementById('globalDetail');
const rowDetailSelects = [...document.querySelectorAll('.row-detail')];

function cardFor(box) { return box.closest('.candidate'); }
function visibleCards() { return [...document.querySelectorAll('.candidate:not(.hidden-flavor)')]; }

function getBg(card) {
  const checked = card.querySelector('.bg-choice input[type=radio]:checked');
  return checked ? checked.value : 'white';
}

function selectedSpecs() {
  return boxes.filter(b => b.checked).map(b => {
    const card = cardFor(b);
    return {id: b.value, background: getBg(card)};
  });
}

function saveSelection() {
  const obj = {};
  boxes.forEach(b => {
    const card = cardFor(b);
    obj[b.value] = {checked: b.checked, background: getBg(card)};
  });
  localStorage.setItem(SELECT_KEY, JSON.stringify(obj));
}

function updateUI(save=true) {
  document.querySelectorAll('.candidate').forEach(card => {
    card.classList.toggle('selected', card.querySelector('.candidate-pick').checked);
  });
  const specs = selectedSpecs();
  document.getElementById('count').textContent = specs.length + ' selected';
  document.getElementById('downloadBtn').disabled = specs.length === 0;
  document.getElementById('selectedField').value = JSON.stringify(specs);
  if (save) saveSelection();
}

function enforceOne(box) {
  if (!onePer.checked || !box.checked) return;
  const imageKey = cardFor(box).dataset.image;
  boxes.forEach(other => {
    if (other !== box && cardFor(other).dataset.image === imageKey) other.checked = false;
  });
}

boxes.forEach(box => box.addEventListener('change', () => {
  enforceOne(box);
  updateUI();
}));

document.querySelectorAll('.bg-choice input[type=radio]').forEach(radio => {
  radio.addEventListener('change', () => {
    const card = radio.closest('.candidate');
    const box = card.querySelector('.candidate-pick');
    box.checked = true;
    enforceOne(box);
    updateUI();
  });
});

document.querySelectorAll('.candidate img').forEach(img => {
  img.addEventListener('click', () => {
    const card = img.closest('.candidate');
    const box = card.querySelector('.candidate-pick');
    box.checked = !box.checked;
    enforceOne(box);
    updateUI();
  });
});

document.getElementById('clearBtn').onclick = () => {
  boxes.forEach(b => b.checked=false);
  updateUI();
};

document.getElementById('allBtn').onclick = () => {
  const cards = visibleCards();
  if (onePer.checked) {
    const seen = new Set();
    boxes.forEach(b => b.checked = false);
    cards.forEach(card => {
      const k = card.dataset.image;
      if (!seen.has(k)) {
        card.querySelector('.candidate-pick').checked = true;
        seen.add(k);
      }
    });
  } else {
    cards.forEach(card => card.querySelector('.candidate-pick').checked = true);
  }
  updateUI();
};

document.getElementById('whiteBtn').onclick = () => {
  boxes.filter(b => b.checked).forEach(b => {
    const r = cardFor(b).querySelector('.bg-choice input[value=white]');
    if (r) r.checked = true;
  });
  updateUI();
};

document.getElementById('blackBtn').onclick = () => {
  boxes.filter(b => b.checked).forEach(b => {
    const r = cardFor(b).querySelector('.bg-choice input[value=black]');
    if (r) r.checked = true;
  });
  updateUI();
};

onePer.addEventListener('change', () => {
  if (onePer.checked) {
    const seen = new Set();
    boxes.forEach(b => {
      if (b.checked) {
        const k = cardFor(b).dataset.image;
        if (seen.has(k)) b.checked=false;
        else seen.add(k);
      }
    });
  }
  updateUI();
});

function applyFlavorVisibility(save=true) {
  const visible = new Set(flavorBoxes.filter(b => b.checked).map(b => b.value));
  document.querySelectorAll('.candidate').forEach(card => {
    card.classList.toggle('hidden-flavor', !visible.has(card.dataset.method));
  });
  if (save) localStorage.setItem(FLAVOR_KEY, JSON.stringify([...visible]));
}

flavorBoxes.forEach(b => b.addEventListener('change', () => applyFlavorVisibility()));
document.getElementById('flavorAll').onclick = () => { flavorBoxes.forEach(b => b.checked=true); applyFlavorVisibility(); };
document.getElementById('flavorNone').onclick = () => { flavorBoxes.forEach(b => b.checked=false); applyFlavorVisibility(); };

function setRowDetail(row, level) {
  if (!['basic','mid','full'].includes(level)) level = 'basic';
  row.dataset.detail = level;
  const sel = row.querySelector('.row-detail');
  if (sel) sel.value = level;
}

function saveDetailState() {
  const rows = {};
  document.querySelectorAll('.image-row').forEach(row => {
    rows[row.dataset.key] = row.dataset.detail || 'basic';
  });
  localStorage.setItem(DETAIL_KEY, JSON.stringify({global: globalDetail.value, rows}));
}

function applyGlobalDetail(level, save=true) {
  if (!['basic','mid','full'].includes(level)) level = 'basic';
  globalDetail.value = level;
  document.querySelectorAll('.image-row').forEach(row => setRowDetail(row, level));
  if (save) saveDetailState();
}

globalDetail.addEventListener('change', () => applyGlobalDetail(globalDetail.value));
rowDetailSelects.forEach(sel => sel.addEventListener('change', () => {
  setRowDetail(sel.closest('.image-row'), sel.value);
  saveDetailState();
}));

try {
  const saved = JSON.parse(localStorage.getItem(SELECT_KEY) || '{}');
  boxes.forEach(b => {
    const spec = saved[b.value];
    if (!spec) return;
    b.checked = !!spec.checked;
    const bg = (spec.background === 'black') ? 'black' : 'white';
    const r = cardFor(b).querySelector('.bg-choice input[value=' + bg + ']');
    if (r) r.checked = true;
  });
} catch(e) {}

try {
  const raw = localStorage.getItem(FLAVOR_KEY);
  if (raw !== null) {
    const visible = new Set(JSON.parse(raw));
    flavorBoxes.forEach(b => b.checked = visible.has(b.value));
  }
} catch(e) {}

try {
  const savedDetail = JSON.parse(localStorage.getItem(DETAIL_KEY) || '{}');
  const globalLevel = ['basic','mid','full'].includes(savedDetail.global) ? savedDetail.global : 'basic';
  globalDetail.value = globalLevel;
  document.querySelectorAll('.image-row').forEach(row => {
    const savedRows = savedDetail.rows || {};
    const level = ['basic','mid','full'].includes(savedRows[row.dataset.key]) ? savedRows[row.dataset.key] : globalLevel;
    setRowDetail(row, level);
  });
} catch(e) {
  applyGlobalDetail('basic', false);
}

if (onePer.checked) {
  const seen = new Set();
  boxes.forEach(b => {
    if (b.checked) {
      const k = cardFor(b).dataset.image;
      if (seen.has(k)) b.checked=false;
      else seen.add(k);
    }
  });
}

applyFlavorVisibility(false);
updateUI(false);
'''


def safe_archive_name(name: str) -> str:
    name = Path(name).name.replace("\x00", "")
    return name or "image.png"


class SelectorIndex:
    def __init__(self, root: Path, preview_long_side: int = 720, methods: Optional[List[str]] = None):
        self.root = root.resolve()
        self.preview_long_side = preview_long_side
        self.allowed_methods = set(methods) if methods else None
        metadata_path = self.root / "metadata.json"
        if not metadata_path.exists():
            raise SystemExit(f"Missing comparator metadata: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.items, self.selections, self.methods = self._index()

    def _index(self):
        items = []
        selections: Dict[str, dict] = {}
        methods_seen: Dict[str, str] = {}
        for key, state in self.metadata.items():
            if not isinstance(state, dict):
                continue
            source = state.get("source") or key
            original_name = Path(source).name
            size = state.get("size") or []
            candidates = []
            results = state.get("results") or {}
            for method, result in results.items():
                if self.allowed_methods is not None and method not in self.allowed_methods:
                    continue
                if not isinstance(result, dict) or not result.get("ok"):
                    continue
                fg = (self.root / key / method / "foreground.png").resolve()
                try:
                    fg.relative_to(self.root)
                except ValueError:
                    continue
                if not fg.exists():
                    continue
                sid = f"{key}::{method}"
                label = METHOD_LABELS.get(method, method)
                methods_seen[method] = label
                method_preview = (self.root / key / method / "preview.jpg").resolve()
                details_preview = (self.root / key / method / "details.jpg").resolve()
                selections[sid] = {
                    "id": sid,
                    "key": key,
                    "method": method,
                    "label": label,
                    "foreground": fg,
                    "method_preview": method_preview if method_preview.exists() else None,
                    "details_preview": details_preview if details_preview.exists() else None,
                    "source": source,
                    "original_name": original_name,
                }
                candidates.append({
                    "id": sid,
                    "method": method,
                    "label": label,
                    "has_method_preview": method_preview.exists(),
                    "has_details_preview": details_preview.exists(),
                })
            if candidates:
                items.append({
                    "key": key,
                    "display_name": original_name,
                    "size_text": f"{size[0]}×{size[1]}" if len(size) >= 2 else "",
                    "original_preview": (self.root / key / "original_preview.jpg").exists(),
                    "candidates": candidates,
                })
        # Stable friendly order: known methods first in METHOD_LABELS order, then extras.
        ordered_methods = []
        for method, label in METHOD_LABELS.items():
            if method in methods_seen:
                ordered_methods.append((method, label))
        for method in sorted(methods_seen):
            if method not in METHOD_LABELS:
                ordered_methods.append((method, methods_seen[method]))
        return items, selections, ordered_methods

    def page_html(self) -> bytes:
        flavor_controls = []
        for method, label in self.methods:
            flavor_controls.append(
                f'<label class="flavor-chip"><input class="flavor-filter" type="checkbox" '
                f'value="{html.escape(method, quote=True)}" checked>{html.escape(label)}</label>'
            )

        body = []
        for item in self.items:
            key_e = html.escape(item["key"], quote=True)
            body.append(
                f'<section class="image-row" data-key="{key_e}" data-detail="basic">'
                f'<div class="row-titlebar"><h2>{html.escape(item["display_name"])}</h2>'
                f'<label class="row-detail-wrap">Detail <select class="detail-select row-detail">'
                f'<option value="basic">Basic</option><option value="mid">Mid</option><option value="full">Full</option>'
                f'</select></label></div><div class="row-grid">'
            )
            body.append('<div class="source-card">')
            if item["original_preview"]:
                body.append(f'<img loading="lazy" src="/original/{quote(item["key"], safe="")}">')
            body.append(f'<div class="caption">Original<br>{html.escape(item["size_text"])}</div></div>')
            body.append('<div class="candidates-grid">')
            for c in item["candidates"]:
                sid_e = html.escape(c["id"], quote=True)
                method_e = html.escape(c["method"], quote=True)
                # Radio group is unique to this candidate. White is the default export background.
                group = "bg_" + str(abs(hash(c["id"])))
                body.append(
                    f'<div class="candidate" data-selection="{sid_e}" data-image="{key_e}" data-method="{method_e}">'
                    f'<div class="basic-view"><span class="split-note">BLACK | WHITE</span>'
                    f'<img loading="lazy" src="/preview/{quote(c["id"], safe="")}" alt="{html.escape(c["label"], quote=True)}"></div>'
                    f'<div class="mid-view"><img loading="lazy" src="/method-preview/{quote(c["id"], safe="")}" alt="{html.escape(c["label"], quote=True)} comparison panel"></div>'
                    + (f'<div class="full-view"><img loading="lazy" src="/details/{quote(c["id"], safe="")}" alt="{html.escape(c["label"], quote=True)} edge details"></div>' if c.get("has_details_preview") else '') +
                    f'<div class="caption">'
                    f'<div class="pick-line"><input class="candidate-pick" type="checkbox" value="{sid_e}">'
                    f'<span class="method-name">{html.escape(c["label"])}</span></div>'
                    f'<div class="method-meta">{html.escape(c["method"])}</div>'
                    f'<div class="bg-choice">'
                    f'<label class="bg-option white"><input type="radio" name="{group}" value="white" checked>White</label>'
                    f'<label class="bg-option black"><input type="radio" name="{group}" value="black">Black</label>'
                    f'</div></div></div>'
                )
            body.append('</div></div></section>')
        if not self.items:
            body.append('<div class="empty">No successful foreground outputs found.</div>')
        root_json = json.dumps(str(self.root))
        doc = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Background Removal Selector</title><style>{CSS}</style></head><body>
<header><h1>Background Removal Selector <span style="font-size:12px;color:#69b7ff">v3.1</span></h1>
<div class="hint">Taste the methods you want, choose the best cutout per image, then choose whether that image should be exported over white or black.</div>
<div class="flavor-panel"><span class="flavor-title">Flavors to taste:</span>{''.join(flavor_controls)}
<span class="flavor-actions"><button class="secondary" id="flavorAll" type="button">All</button><button class="secondary" id="flavorNone" type="button">None</button></span></div>
<div class="detail-panel"><span class="detail-title">Detail level:</span>
<select class="detail-select" id="globalDetail"><option value="basic">Basic</option><option value="mid">Mid</option><option value="full">Full</option></select>
<span class="detail-help">Sets all image rows; each row can then be overridden independently.</span></div>
</header>
<main>{''.join(body)}</main>
<form id="downloadForm" method="post" action="/download"><input type="hidden" id="selectedField" name="selected" value="[]"></form>
<div class="toolbar"><button class="primary" id="downloadBtn" disabled onclick="document.getElementById('downloadForm').submit()">Download selected (.tar.gz)</button>
<button class="secondary" id="clearBtn" type="button">Clear</button><button class="secondary" id="allBtn" type="button">Select visible</button>
<button class="secondary" id="whiteBtn" type="button">Selected → White</button><button class="secondary" id="blackBtn" type="button">Selected → Black</button>
<label class="toggle"><input type="checkbox" id="onePerImage" checked> one best flavor per source</label><span class="count" id="count">0 selected</span></div>
<script>const ROOT_ID={root_json};{JS}</script></body></html>'''
        return doc.encode("utf-8")

    def split_preview(self, sid: str) -> bytes:
        sel = self.selections.get(sid)
        if not sel:
            raise KeyError(sid)
        with Image.open(sel["foreground"]) as im:
            rgba = im.convert("RGBA")
            rgba.thumbnail((self.preview_long_side, self.preview_long_side), Image.Resampling.LANCZOS)
            w, h = rgba.size
            bg = Image.new("RGB", (w, h), "black")
            bg.paste(Image.new("RGB", (max(1, w - w // 2), h), "white"), (w // 2, 0))
            bg.paste(rgba.convert("RGB"), (0, 0), rgba.getchannel("A"))
            out = io.BytesIO()
            bg.save(out, format="JPEG", quality=91, subsampling=0)
            return out.getvalue()

    def artifact_path(self, sid: str, kind: str) -> Path:
        sel = self.selections.get(sid)
        if not sel:
            raise KeyError(sid)
        field = {"method-preview": "method_preview", "details": "details_preview"}.get(kind)
        if not field:
            raise KeyError(kind)
        p = sel.get(field)
        if not p:
            raise FileNotFoundError(f"No {kind} artifact for {sid}")
        p = Path(p).resolve()
        p.relative_to(self.root)
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    def asset_path(self, sid: str, kind: str) -> Path:
        sel = self.selections.get(sid)
        if not sel:
            raise KeyError(sid)
        if kind == "preview":
            p = sel.get("preview")
        elif kind == "details":
            p = sel.get("details")
        else:
            raise KeyError(kind)
        if not p:
            raise FileNotFoundError(kind)
        p = Path(p).resolve()
        p.relative_to(self.root)
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    def original_path(self, key: str) -> Path:
        if key not in self.metadata:
            raise KeyError(key)
        p = (self.root / key / "original_preview.jpg").resolve()
        p.relative_to(self.root)
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    def select(self, specs: List[dict]):
        out, seen = [], set()
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            sid = str(spec.get("id", ""))
            if sid in seen or sid not in self.selections:
                continue
            seen.add(sid)
            bg = str(spec.get("background", "white")).lower()
            if bg not in {"white", "black"}:
                bg = "white"
            item = dict(self.selections[sid])
            item["background"] = bg
            out.append(item)
        return out


class SelectorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, index: SelectorIndex):
        super().__init__(address, handler)
        self.index = index


class Handler(BaseHTTPRequestHandler):
    server_version = "BGSelector/3.1"

    @property
    def idx(self) -> SelectorIndex:
        return self.server.index  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_bytes(self, data: bytes, content_type: str, status=200, cache="no-store"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(data)

    def send_error_text(self, status: int, text: str):
        self.send_bytes(text.encode("utf-8"), "text/plain; charset=utf-8", status)

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/":
                self.send_bytes(self.idx.page_html(), "text/html; charset=utf-8")
                return
            if path.startswith("/preview/"):
                sid = unquote(path[len("/preview/"):])
                self.send_bytes(self.idx.split_preview(sid), "image/jpeg", cache="public, max-age=3600")
                return
            if path.startswith("/method-preview/"):
                sid = unquote(path[len("/method-preview/"):])
                p = self.idx.artifact_path(sid, "method-preview")
                self.send_bytes(p.read_bytes(), "image/jpeg", cache="public, max-age=3600")
                return
            if path.startswith("/details/"):
                sid = unquote(path[len("/details/"):])
                p = self.idx.artifact_path(sid, "details")
                self.send_bytes(p.read_bytes(), "image/jpeg", cache="public, max-age=3600")
                return
            if path.startswith("/original/"):
                key = unquote(path[len("/original/"):])
                p = self.idx.original_path(key)
                self.send_bytes(p.read_bytes(), "image/jpeg", cache="public, max-age=3600")
                return
            if path.startswith("/asset/"):
                rest = path[len("/asset/"):]
                kind, sid = rest.split("/", 1)
                sid = unquote(sid)
                p = self.idx.asset_path(sid, kind)
                ctype = "image/jpeg" if p.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
                self.send_bytes(p.read_bytes(), ctype, cache="public, max-age=3600")
                return
            if path == "/api/index":
                data = json.dumps({
                    "images": len(self.idx.items),
                    "candidates": len(self.idx.selections),
                    "methods": [m for m, _ in self.idx.methods],
                }).encode()
                self.send_bytes(data, "application/json")
                return
            self.send_error_text(404, "Not found")
        except (KeyError, FileNotFoundError, ValueError):
            self.send_error_text(404, "Not found")
        except Exception as e:
            self.send_error_text(500, f"Internal error: {e}")

    def do_POST(self):
        if urlparse(self.path).path != "/download":
            self.send_error_text(404, "Not found")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            form = parse_qs(self.rfile.read(n).decode("utf-8", "replace"))
            requested = json.loads(form.get("selected", ["[]"])[0])
            if not isinstance(requested, list):
                raise ValueError("selected must be a list")
            selected = self.idx.select(requested)
            if not selected:
                self.send_error_text(400, "Nothing selected")
                return
            self.stream_tar(selected)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                self.send_error_text(400, f"Invalid request: {e}")
            except Exception:
                pass

    @staticmethod
    def _composite_to_spooled_png(foreground: Path, background: str):
        """Render one full-resolution RGBA cutout over white/black as RGB PNG.

        SpooledTemporaryFile keeps modest PNGs in RAM and transparently spills
        large ones to disk, so a whole batch is never held in memory.
        """
        f = tempfile.SpooledTemporaryFile(max_size=32 * 1024 * 1024, mode="w+b")
        with Image.open(foreground) as im:
            rgba = im.convert("RGBA")
            color = (255, 255, 255) if background == "white" else (0, 0, 0)
            out = Image.new("RGB", rgba.size, color)
            out.paste(rgba.convert("RGB"), (0, 0), rgba.getchannel("A"))
            out.save(f, format="PNG", optimize=False)
        f.flush()
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(0)
        return f, size

    @staticmethod
    def _add_bytes(tf: tarfile.TarFile, name: str, data: bytes):
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mtime = int(__import__("time").time())
        tf.addfile(info, io.BytesIO(data))

    def stream_tar(self, selected: List[dict]):
        per_key: Dict[str, int] = {}
        for s in selected:
            per_key[s["key"]] = per_key.get(s["key"], 0) + 1

        used = set()
        manifest = []
        archive_rows = []
        for s in selected:
            stem = Path(s["original_name"]).stem or Path(s["key"]).name
            # Normal case: one chosen flavor -> preserve the original stem.
            # If multiple flavors were deliberately selected for one source,
            # suffix method+background to avoid collisions and make comparison clear.
            if per_key[s["key"]] > 1:
                base = f"{stem}__{s['method']}__{s['background']}.png"
            else:
                base = f"{stem}.png"
            base = safe_archive_name(base)
            candidate = base
            counter = 2
            while candidate.lower() in used:
                candidate = f"{Path(base).stem}__{counter:02d}.png"
                counter += 1
            used.add(candidate.lower())
            archive_rows.append((candidate, s))
            manifest.append({
                "archive_name": candidate,
                "source": s["source"],
                "method": s["method"],
                "method_label": s["label"],
                "background": s["background"],
                "source_foreground": str(s["foreground"]),
            })

        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/gzip")
            self.send_header("Content-Disposition", 'attachment; filename="background-selections.tar.gz"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            # Streaming tar.gz: one full-res selected image is composited/encoded
            # at a time, then written immediately into the HTTP response.
            with tarfile.open(fileobj=self.wfile, mode="w|gz", compresslevel=6) as tf:
                for archive_name, s in archive_rows:
                    tmp, size = self._composite_to_spooled_png(s["foreground"], s["background"])
                    try:
                        info = tarfile.TarInfo(name=archive_name)
                        info.size = size
                        info.mtime = int(__import__("time").time())
                        tf.addfile(info, tmp)
                    finally:
                        tmp.close()
                self._add_bytes(tf, "manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser(
        description="Visually select background-removal flavors, choose white/black export backgrounds, and stream selected full-resolution PNGs as a tar.gz."
    )
    ap.add_argument("output_dir", type=Path, help="Root output directory from compare_backgrounds.py")
    ap.add_argument("--host", default="127.0.0.1", help="Bind address; use 0.0.0.0 on a remote pod")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--preview-long-side", type=int, default=720, help="Maximum black/white preview width/height")
    ap.add_argument(
        "--methods", nargs="+", default=None,
        help="Optional hard filter of methods to index, e.g. --methods rmbg2 rmbg2_vitmatte birefnet_vitmatte",
    )
    args = ap.parse_args()

    idx = SelectorIndex(args.output_dir, args.preview_long_side, args.methods)
    server = SelectorServer((args.host, args.port), Handler, idx)
    print(f"Indexed {len(idx.items)} source images / {len(idx.selections)} candidate cutouts")
    print("Methods:", ", ".join(m for m, _ in idx.methods) or "none")
    print(f"Open: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
