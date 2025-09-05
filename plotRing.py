#!/usr/bin/env python3
import argparse
import math
import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Recognize SAD blocks and definitions
BLOCK_HEADER_RE = re.compile(
    r"^\s*(DRIFT|BEND|QUAD|SEXT|MARK|MULT|SOL|CAVI|APERT)\b",
    re.IGNORECASE,
)
ELEM_DEF_RE = re.compile(r"([A-Za-z0-9_.]+)\s*=\s*\(([^)]*)\)")
KV_RE = re.compile(r"([A-Za-z0-9_]+)\s*=\s*([^\s,)]+)")
LINE_RE = re.compile(r"\bLINE\b\s+([A-Za-z0-9_.]+)\s*=\s*\((.*?)\)\s*;", re.IGNORECASE | re.DOTALL)
MULT_RE = re.compile(r"^\s*(\d+)\s*\*\s*([A-Za-z0-9_.]+)\s*$")

@dataclass
class Element:
    name: str
    sad_type: str
    params: Dict[str, float]

@dataclass
class SadModel:
    elements: Dict[str, Element] = field(default_factory=dict)
    lines: Dict[str, List[str]] = field(default_factory=dict)

    def ensure_line_exists(self, line_name: str) -> bool:
        return line_name in self.lines

def parse_sad(path: str) -> SadModel:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    model = SadModel()

    # Parse element blocks
    # Split text by semicolon to get blocks
    blocks = text.split(";")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        m = BLOCK_HEADER_RE.match(block)
        if not m:
            continue
        sad_type = m.group(1).upper()
        #if sad_type != "BEND":
        #    continue
        block_body = block[m.end():].strip()
        for em in ELEM_DEF_RE.finditer(block_body):
            name = em.group(1).strip()
            raw_params = em.group(2)
            params: Dict[str, float] = {}
            for kv in KV_RE.finditer(raw_params):
                k = kv.group(1).strip().upper()
                v = kv.group(2).strip().rstrip(",")
                try:
                    params[k] = float(v)
                except ValueError:
                    # Ignore non-numeric params
                    pass
            model.elements[name] = Element(name=name, sad_type=sad_type, params=params)

    for lm in LINE_RE.finditer(text):
        lname = lm.group(1).strip()
        seq_body = lm.group(2).strip()
        seq = expand_line_sequence(seq_body)
        model.lines[lname] = seq

    return model

def save_model(model: SadModel, path: str) -> None:
    data = {
        "elements": {name: {"type": el.sad_type, "params": el.params} for name, el in model.elements.items()},
        "lines": model.lines,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def expand_line_sequence(body: str) -> List[str]:
    tokens = [t.strip() for t in body.split(",") if t.strip()]
    out: List[str] = []
    for tok in tokens:
        out.append(tok)
    return out

def element_length(el: Element) -> float:
    # Prefer L; if not present, zero
    return float(el.params.get("L", 0.0))

def color_for_type(sad_type: str) -> str:
    t = sad_type.upper()
    # Choose distinct, readable colors
    if t == "BEND":
        return "#d62728"  # red
    if t == "QUAD":
        return "#1f77b4"  # blue
    if t == "SEXT":
        return "#2ca02c"  # green
    if t == "OCT":
        return "#9467bd"  # purple
    if t in ("HKICK", "VKICK", "KICKER"):
        return "#ff7f0e"  # orange
    if t == "SOL":
        return "#8c564b"  # brown
    if t in ("MULT", "MLT", "MLUT"):
        return "#17becf"  # cyan
    if t == "MARK":
        return "#7f7f7f"  # gray
    if t == "DRIFT":
        return "#bbbbbb"  # light gray
    return "#000000"      # default black

def draw_ring_circle(ax, R: float, color="#dddddd"):
    theta = np.linspace(0, 2*np.pi, 720)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    ax.plot(x, y, color=color, linewidth=1.0, zorder=1)

def draw_segment_arc(ax, R: float, theta0: float, theta1: float, color: str, lw: float = 6.0, z: int = 2):
    # Draw an arc along the circle between theta0 and theta1
    # Ensure theta0 < theta1
    if theta1 < theta0:
        theta0, theta1 = theta1, theta0
    n = max(8, int(200 * (theta1 - theta0) / (2*np.pi)))
    theta = np.linspace(theta0, theta1, n)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    ax.plot(x, y, color=color, linewidth=lw, solid_capstyle="butt", zorder=z)

def build_sequence(model: SadModel, line_name: str) -> List[Element]:
    if line_name not in model.lines:
        raise RuntimeError(f"LINE '{line_name}' not found in the SAD file.")
    seq_names = model.lines[line_name]
    seq_elems: List[Element] = []
    for n in seq_names:
        el = model.elements.get(n)
        if el is None:
            # Unknown element reference; create a zero-length marker placeholder
            el = Element(name=n, sad_type="MARK", params={})
        seq_elems.append(el)
    return seq_elems

def plot_ring_layout(model: SadModel, line_name: str, out_path: str, zero_len_ticks: bool = True):
    seq = build_sequence(model, line_name)

    # Build s-positions and total circumference
    s_pos: List[float] = [0.0]
    types: List[str] = []
    names: List[str] = []
    lengths: List[float] = []
    for el in seq:
        L = element_length(el)
        names.append(el.name)
        types.append(el.sad_type)
        lengths.append(L)
        s_pos.append(s_pos[-1] + max(L, 0.0))
    C = s_pos[-1]
    if C <= 0:
        raise RuntimeError("Total length is zero; cannot draw ring.")

    # Choose radius so the ring fits nicely
    R = 1.0  # arbitrary radius; absolute scale doesn't matter
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")

    # Base circle
    draw_ring_circle(ax, R, color="#dddddd")

    # Draw each element as an arc proportional to its length
    # Map s in [0,C] to theta in [0, 2pi)
    for i, (t, L) in enumerate(zip(types, lengths)):
        s0 = s_pos[i]
        s1 = s_pos[i+1]
        if L > 0:
            theta0 = 2 * math.pi * (s0 / C)
            theta1 = 2 * math.pi * (s1 / C)
            draw_segment_arc(ax, R, theta0, theta1, color_for_type(t))
        else:
            # Optionally draw a small tick for zero-length elements
            if zero_len_ticks:
                theta = 2 * math.pi * (s0 / C)
                # A short radial tick
                r0, r1 = R - 0.02, R + 0.02
                ax.plot([r0*math.cos(theta), r1*math.cos(theta)],
                        [r0*math.sin(theta), r1*math.sin(theta)],
                        color=color_for_type(t), linewidth=1.0, zorder=3)

    # Build a minimal legend from the types present
    present = []
    for t in types:
        if t not in present:
            present.append(t)
    # Show up to ~8 legend entries to avoid clutter
    handles = []
    labels = []
    for t in present[:8]:
        handles.append(plt.Line2D([0], [0], color=color_for_type(t), lw=4))
        labels.append(t)
    if handles:
        ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Plot a circular ring layout from a SAD lattice LINE.")
    ap.add_argument("--input", "-i", required=True, help="Path to SAD lattice file")
    ap.add_argument("--line", "-l", default="RINGH", help="LINE to plot (e.g. RINGH, SlicedRing, APTRing)")
    ap.add_argument("--output", "-o", default="ring.png", help="Output image path (PNG, PDF, etc.)")
    ap.add_argument("--no-zero-ticks", action="store_true", help="Do not draw ticks for zero-length elements")
    ap.add_argument("--save-model", "-s", default=None, help="Save parsed model to JSON file")
    args = ap.parse_args()

    model = parse_sad(args.input)

    if args.save_model:
        save_model(model, args.save_model)
    exit()
    plot_ring_layout(model, args.line, args.output, zero_len_ticks=not args.no_zero_ticks)
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()