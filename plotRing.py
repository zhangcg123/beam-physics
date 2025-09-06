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
        #print (seq_body)
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
    tokens = [t.strip() for t in re.split(r"[ \t]+", body) if t.strip()]
    #print ( tokens )
    #exit()
    out: List[str] = []
    for tok in tokens:
        out.append(tok)
    return out

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
            print(f"Warning: Element '{n}' not found; using MARK placeholder.")
            el = Element(name=n, sad_type="MARK", params={})
        seq_elems.append(el)
    return seq_elems

def plot_ring_layout(model: SadModel, line_name: str, out_path: str, zero_len_ticks: bool = True):
    seq = build_sequence(model, line_name)
    # Build the ring layout
    types: List[str] = []
    names: List[str] = []
    lengths: List[float] = []
    angles: List[float] = []
    xs: List[float] = [0.0]
    ys: List[float] = [0.0]
    for el in seq:
        L = float(el.params.get("L", 0.0))
        A = float(el.params.get("ANGLE", 0.0))
        if L == 0:
            continue
        lengths.append(L)
        angles.append(A)
        names.append(el.name)
        types.append(el.sad_type)
        xs.append( xs[-1] + L * math.cos( sum(angles) ) )
        ys.append( ys[-1] + L * math.sin( sum(angles) ) )   
    print(f"Number of elements: {len(seq)}")
    print(f"Total length: {sum(lengths):.3f} m")
    print(f"Total bending angle: {sum(angles):.3f} radians")
    plt.scatter(xs, ys, s=0.1, alpha=0.5)
    plt.savefig("debug_path.png")   
    plt.clf()
    b_xs = []
    b_ys = []
    m_xs = []
    m_ys = []
    d_xs = []
    d_ys = []
    o_xs = []
    o_ys = []
    for i in range(len(types)):
        t = types[i].upper()
        if t == "BEND":
            b_xs.append(xs[i])
            b_ys.append(ys[i])
        elif t == "DRIFT":
            d_xs.append(xs[i])
            d_ys.append(ys[i])
        elif t == "MULT" or t == "QUAD" or t == "SEXT":
            m_xs.append(xs[i])
            m_ys.append(ys[i])
    fig, axs = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

    # Panel 1: All elements
    xlim = [-1500,1500]
    ylim = [-20,20]
    axs[0].scatter(xs, ys, s=0.5, alpha=0.5, c="black", label="All Elements")
    #axs[0].set_title("All Elements")
    axs[0].set_ylabel("Y (m)")
    axs[0].set_xlabel("X (m)")
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].legend(markerscale=2)
    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)

    axs[1].scatter(d_xs, d_ys, s=20, c=color_for_type("DRIFT"), label="DRIFT", alpha=0.7)
    axs[1].set_title("BENDs and DRIFTs")
    axs[1].set_ylabel("Y (m)")
    axs[1].legend(markerscale=2)
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].set_xlim(xlim)
    axs[1].set_ylim(ylim)

    # Panel 3: BEND
    axs[2].scatter(b_xs, b_ys, s=20, c=color_for_type("BEND"), label="BEND", alpha=0.7)
    axs[2].set_title("BENDs")
    axs[2].set_ylabel("Y (m)")
    axs[2].legend(markerscale=2)
    axs[2].grid(True, linestyle='--', alpha=0.5)
    axs[2].set_aspect('equal', adjustable='box')
    axs[2].set_xlim(xlim)
    axs[2].set_ylim(ylim)

    # Panel 4: Multipoles
    axs[3].scatter(m_xs, m_ys, s=20, c=color_for_type("QUAD"), label="Multipoles", alpha=0.7)
    axs[3].set_title("Multipoles (QUAD, SEXT, MULT)")
    axs[3].set_ylabel("Y (m)")
    axs[3].set_xlabel("X (m)")
    axs[3].legend(markerscale=2)
    axs[3].grid(True, linestyle='--', alpha=0.5)
    axs[3].set_aspect('equal', adjustable='box')
    axs[3].set_xlim(xlim)
    axs[3].set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.clf()

def to_gmad(model: SadModel, out_path: str) -> None:        
    with open(out_path, "w", encoding="utf-8") as f:    
        f.write("! Converted from SAD to GMAD format\n")
        for name, el in model.elements.items():
            if el.sad_type == 'DRIFT':
                f.write(f"{name.lower()}: drift, l={el.params.get('L', 0.0)}*m;\n")
            elif el.sad_type == 'BEND':
                f.write(f"{name.lower()}: sbend, l={el.params.get('L', 0.0)}*m, angle={el.params.get('ANGLE', 0.0)}, e1={el.params.get('E1', 0.0)}, e2={el.params.get('E2', 0.0)};\n")
            elif el.sad_type == 'QUAD':
                f.write(f"{name.lower()}: quadrupole, l={el.params.get('L', 0.0)}*m, k1={el.params.get('K1', 0.0)};\n")
            elif el.sad_type == 'SEXT':
                f.write(f"{name.lower()}: sextupole, l={el.params.get('L', 0.0)}*m, k2={el.params.get('K2', 0.0)};\n")
            else:
                f.write(f"! {name.lower()}: {el.sad_type} not converted\n")
        f.write("\n")

def main():
    ap = argparse.ArgumentParser(description="Plot a circular ring layout from a SAD lattice LINE.")
    ap.add_argument("--input", "-i", required=True, help="Path to SAD lattice file")
    ap.add_argument("--line", "-l", default="RINGH", help="LINE to plot (e.g. RINGH, SlicedRing, APTRing)")
    ap.add_argument("--ring", "-r", default="ring.png", help="Output image path (PNG, PDF, etc.)")
    ap.add_argument("--no-zero-ticks", action="store_true", help="Do not draw ticks for zero-length elements")
    ap.add_argument("--save-model", "-s", default=None, help="Save parsed model to JSON file")
    ap.add_argument("--to-gmad", "-g", default=None, help="Convert to GMAD format and save to file")
    args = ap.parse_args()

    model = parse_sad(args.input)
    
    if args.save_model:
        save_model(model, args.save_model)

    if args.ring:
        plot_ring_layout(model, args.line, args.ring, zero_len_ticks=not args.no_zero_ticks)
        print(f"Ring layout saved to '{args.ring}'")

    if args.to_gmad:
        to_gmad(model, args.to_gmad)

if __name__ == "__main__":
    main()