#!/usr/bin/env python3
import argparse
import math
import re
import json
from dataclasses import dataclass, field, replace
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
    start_s: float = 0.0
    global_x: float = 0.0
    global_y: float = 0.0
    gmad_string: str = ""

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
        base = model.elements.get(n)
        if base is None:
            print(f"Warning: Element '{n}' not found in model elements.")
            base = Element(name=n, sad_type="MARK", params={})
        # Create a copy to avoid modifying the original
        el = Element(name=base.name, sad_type=base.sad_type, params=base.params.copy())
        seq_elems.append(el)
    return seq_elems

def process_ring(model: SadModel, line_name: str, out_path: str) -> List[Element]:
    seq = build_sequence(model, line_name)
    # Build the ring layout
    angles: List[float] = []
    lengths: List[float] = []
    xs: List[float] = [0]
    ys: List[float] = [0]
    for el in seq:
        L = float(el.params.get("L", 0.0))
        A = float(el.params.get("ANGLE", 0.0))
        angles.append(A)
        lengths.append(L)
        el.start_s = sum(lengths)
        xs.append(xs[-1] + L * math.cos(sum(angles)))
        ys.append(ys[-1] + L * math.sin(sum(angles)))
        el.global_x = xs[-1]
        el.global_y = ys[-1]
        el.gmad_string = convert(el)
        #print ( el.gmad_string )
    print(f"Number of elements: {len(seq)}")
    print(f"Total length: {sum(el.params.get('L', 0.0) for el in seq):.3f} m")
    print(f"Total angle: {sum(el.params.get('ANGLE', 0.0) for el in seq):.3f} rad")
    plt.scatter(
        [el.global_x for el in seq],
        [el.global_y for el in seq],
        s=1,
        c="black",
        alpha=0.5,
        label="Elements"
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"Ring Layout: {line_name}")
    ax = plt.gca()
    ax.text(
        0.02, 0.98,
        f"Length={sum(el.params.get('L', 0.0) for el in seq):.1f}  Angle={sum(el.params.get('ANGLE', 0.0) for el in seq):.3f} rad\n"
        f"Elements={len(seq)}  NBend={sum(1 for el in seq if el.sad_type.upper()=='BEND')}\n"
        f"NQuad={sum(1 for el in seq if el.sad_type.upper()=='QUAD')}\n"
        f"NSext={sum(1 for el in seq if el.sad_type.upper()=='SEXT')}\n"
        f"NDrift={sum(1 for el in seq if el.sad_type.upper()=='DRIFT')}\n"
        f"NMult={sum(1 for el in seq if el.sad_type.upper()=='MULT')}\n"
        f"NCavi={sum(1 for el in seq if el.sad_type.upper()=='CAVI')}\n"
        f"NSol={sum(1 for el in seq if el.sad_type.upper()=='SOL')}\n"
        f"NMark={sum(1 for el in seq if el.sad_type.upper()=='MARK')}",
        transform=ax.transAxes, fontsize=8, va="top", ha="left"
    )
    plt.axis('equal')
    plt.savefig('debug_ring.png', dpi=300)
    plt.clf()
    print ( 'The end point: x=', seq[-1].global_x, ', y=', seq[-1].global_y )
    return seq 
    '''
    b_xs = []
    b_ys = []
    m_xs = []
    m_ys = []
    d_xs = []
    d_ys = []
    o_xs = []
    o_ys = []
    r_xs = []
    r_ys = []
    k_xs = []
    k_ys = []
    s_xs = []
    s_ys = []
    for i in range(len(types)):
        if xs[i] < -1000 or xs[i] > 1000 or ys[i] < -20 or ys[i] > 20:
            continue
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
        elif t == "CAVI":
            r_xs.append(xs[i])
            r_ys.append(ys[i])
        elif t == "MARK":
            k_xs.append(xs[i])
            k_ys.append(ys[i])
        elif t == "SOL":
            s_xs.append(xs[i])
            s_ys.append(ys[i])
        else:
            o_xs.append(xs[i])
            o_ys.append(ys[i])
    fig, axs = plt.subplots(7, 1, figsize=(15, 35), sharex=True)
    print ( d_xs )
    # Panel 1: All elements
    xlim = [-1000,1000]
    ylim = [-20,20]
    axs[0].scatter(xs, ys, s=30, alpha=1, c="black", label="All Elements="+str(len(xs)))
    #axs[0].set_title("All Elements")
    axs[0].set_ylabel("Y (m)")
    axs[0].set_xlabel("X (m)")
    axs[0].grid(True, linestyle='--', alpha=0.5)
    #axs[0].set_aspect('equal', adjustable='box')
    axs[0].legend(markerscale=2)
    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)

    axs[1].scatter(d_xs, d_ys, s=30, c=color_for_type("DRIFT"), label="DRIFT="+str(len(d_xs)), alpha=1)
    #axs[1].set_title("BENDs and DRIFTs")
    axs[1].set_ylabel("Y (m)")
    axs[1].legend(markerscale=2)
    axs[1].grid(True, linestyle='--', alpha=0.5)
    #axs[1].set_aspect('equal', adjustable='box')
    axs[1].set_xlim(xlim)
    axs[1].set_ylim(ylim)

    # Panel 3: BEND
    axs[2].scatter(b_xs, b_ys, s=30, c=color_for_type("BEND"), label="BEND="+str(len(b_xs)), alpha=1)
    #axs[2].set_title("BENDs")
    axs[2].set_ylabel("Y (m)")
    axs[2].legend(markerscale=2)
    axs[2].grid(True, linestyle='--', alpha=0.5)
    #axs[2].set_aspect('equal', adjustable='box')
    axs[2].set_xlim(xlim)
    axs[2].set_ylim(ylim)

    # Panel 4: Multipoles
    axs[3].scatter(m_xs, m_ys, s=20, c=color_for_type("QUAD"), label="Multipoles="+str(len(m_xs)), alpha=1)
    #axs[3].set_title("Multipoles (QUAD, SEXT, MULT)")
    axs[3].set_ylabel("Y (m)")
    axs[3].set_xlabel("X (m)")
    axs[3].legend(markerscale=2)
    axs[3].grid(True, linestyle='--', alpha=0.5)
    #axs[3].set_aspect('equal', adjustable='box')
    axs[3].set_xlim(xlim)
    axs[3].set_ylim(ylim)

    # Panel 5: for CAVIs
    axs[4].scatter(r_xs, r_ys, s=20, c=color_for_type("CAVI"), label="CAVI="+str(len(r_xs)), alpha=1)
    #axs[4].set_title("CAVIs")
    axs[4].set_ylabel("Y (m)")
    axs[4].set_xlabel("X (m)")
    axs[4].legend(markerscale=2)
    axs[4].grid(True, linestyle='--', alpha=0.5)
    #axs[4].set_aspect('equal', adjustable='box')
    axs[4].set_xlim(xlim)
    axs[4].set_ylim(ylim)

    # Panel 6: MARKs
    axs[5].scatter(k_xs, k_ys, s=20, c=color_for_type("MARK"), label="MARK="+str(len(k_xs)), alpha=1)
    #axs[5].set_title("MARKs")
    axs[5].set_ylabel("Y (m)")
    axs[5].set_xlabel("X (m)")
    axs[5].legend(markerscale=2)
    axs[5].grid(True, linestyle='--', alpha=0.5)
    #axs[5].set_aspect('equal', adjustable='box')
    axs[5].set_xlim(xlim)
    axs[5].set_ylim(ylim)

    # Panel 7: SOLs
    axs[6].scatter(s_xs, s_ys, s=20, c=color_for_type("SOL"), label="SOL="+str(len(s_xs)), alpha=1)
    #axs[6].set_title("SOLs")
    axs[6].set_ylabel("Y (m)")
    axs[6].set_xlabel("X (m)")
    axs[6].legend(markerscale=2)
    axs[6].grid(True, linestyle='--', alpha=0.5)
    #axs[6].set_aspect('equal', adjustable='box')
    axs[6].set_xlim(xlim)
    axs[6].set_ylim(ylim)
    if len(o_xs) > 0:
        print(f"Warning: There are {len(o_xs)} elements of other types not plotted separately.")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.clf()
    '''

def convert(el: Element) -> str:
    t = el.sad_type.upper()
    name = el.name.lower()
    if t == "DRIFT":
        L = el.params.get("L", 0.0)
        return f"{name}: drift, l={L}*m;"
    elif t == "BEND":
        L = el.params.get("L", 0.0)
        ANGLE = el.params.get("ANGLE", 0.0)
        E1 = el.params.get("E1", 0.0)
        E2 = el.params.get("E2", 0.0)
        return f"{name}: sbend, l={L}*m, angle={ANGLE}, e1={E1}, e2={E2};"
    elif t == "QUAD":
        L = el.params.get("L", 0.0)
        K1 = el.params.get("K1", 0.0)
        return f"{name}: quadrupole, l={L}*m, k1={K1/L if L != 0 else 0};"
    elif t == "SEXT":
        L = el.params.get("L", 0.0)
        K2 = el.params.get("K2", 0.0)
        return f"{name}: sextupole, l={L}*m, k2={K2/L if L != 0 else 0};"
    elif t == "MULT":
        L = el.params.get("L", 0.0)
        knl_str = ','.join(str(el.params.get(f'K{i}', 0.0)/L) for i in range(0, 5))
        ksl_str = ','.join(str(el.params.get(f'SK{i}', 0.0)/L) for i in range(0, 5))
        return f"{name}: multipole, l={L}*m, knl={{{knl_str}}}, ksl={{{ksl_str}}};"
    elif t == "CAVI":
        L = el.params.get("L", 0.0)
        VOLTAGE = el.params.get("VOLTAGE", 0.0)
        FREQUENCY = el.params.get("FREQUENCY", 0.0)
        LAG = el.params.get("LAG", 0.0)
        return f"{name}: rfcavity, l={L}*m, voltage={VOLTAGE}*MV, frequency={FREQUENCY}*MHz, lag={LAG};"
    elif t == "SOL":
        L = el.params.get("L", 0.0)
        BZ = el.params.get("BZ", 0.0)
        return f"{name}: solenoid, l={L}*m, B={BZ}*T;"
    elif t == "MARK":
        return f"! {name}: marker; "+f" ! at s={el.start_s:.3f} m"
    else:
        return f"! {name} of type {t} not converted"


def to_gmad(seq: List[Element], path: str) -> None:
    if path is None:
        return
    if path.endswith('.gmad'):
        path = path.replace('.gmad', '')
    readyname = []
    with open(path+'_components.gmad', 'w') as f:
        for el in seq:
            readyname.append(el.name.lower())
            if el.sad_type == 'MARK':
                f.write(f"! {el.name.lower()}: marker; ! at s={el.start_s:.3f} m\n")
            if el.gmad_string:
                if el.name.lower() in readyname:
                    f.write(f"! {el.gmad_string} ! Duplicate name, skipped\n")
                else:
                    f.write(el.gmad_string + "\n")
            else:
                f.write(f"! {el.name} of type {el.sad_type} not converted\n")
        f.write("\n")
    with open(path+'_beam.gmad', 'w') as f:
        f.write(f"beam, particle = \"e-\", energy = 120.0*GeV, distrType = \"reference\", ! "+el.name+", "+str(el.start_s)+" m\n")
        #for el in seq:
        #    if el.sad_type == 'MARK':
        #        f.write(f"  {el.name.lower()};\n")
        #    else:
        #        continue
    with open(path+'_line.gmad', 'w') as f:
        line_str = ", ".join(el.name.lower() for el in seq)
        f.write(f"ring: line = ({line_str});\n")
        f.write("use, period = ring;\n")
    with open(path+'_options.gmad', 'w') as f:
        f.write("! to be filled as needed\n")
    with open(path+'.gmad', 'w') as f:
        f.write(f"include {path}_components.gmad;\n")
        f.write(f"include {path}_beam.gmad;\n")
        f.write(f"include {path}_line.gmad;\n")
        f.write(f"include {path}_options.gmad;\n")
        f.write(f"sample, all;\n")

def process_arc(seq: List[Element], out_path: str) -> List[Element]:
    
    # the near mark
    near_distance = 99285.752 # full ring length
    near_idx = -1
    far_distance = 0.0
    far_idx = -1
    for i, el in enumerate(seq):
        if el.sad_type == 'MARK' and el.start_s < near_distance and el.start_s > 100:# the ip is mark sad type
            near_distance = el.start_s
            near_idx = i
        if el.sad_type == 'MARK' and el.start_s > far_distance and el.start_s < 99285.752-300:
            far_distance = el.start_s
            far_idx = i

    if near_idx != -1 and far_idx != -1:
        print(f"Near marker: {seq[near_idx].name} is {seq[near_idx].sad_type} at {seq[near_idx].start_s} m")
        print(f"Far marker: {seq[far_idx].name} is {seq[far_idx].sad_type} at {99285.752 - seq[far_idx].start_s} m")
    else:
        print("Warning: No markers found in the sequence.")
    
    arc_seq = seq[far_idx+1:] + seq[:near_idx+1]

    plt.scatter(
        [el.global_x for el in arc_seq],
        [el.global_y for el in arc_seq],
        s=1,
        c="black",
        alpha=0.5,
        label="Elements"
    )
    #plt.xlim(-100, 100)
    #plt.ylim(-10, 10)
    plt.savefig('debug_arc.png', dpi=300)
    plt.clf()
    return arc_seq
    

def main():
    ap = argparse.ArgumentParser(description="Plot a circular ring layout from a SAD lattice LINE.")
    ap.add_argument("--input", "-i", required=True, help="Path to SAD lattice file")
    ap.add_argument("--line", "-l", default="RINGH", help="LINE to plot (e.g. RINGH, SlicedRing, APTRing)")
    ap.add_argument("--to-json", "-s", default=None, help="Save parsed model to JSON file")
    ap.add_argument("--to-gmad", "-g", default=None, help="Convert to GMAD format and save to file")
    args = ap.parse_args()

    model = parse_sad(args.input)

    seq = process_ring(model, args.line, "ring_layout.png")

    arc = process_arc(seq, "arc_layout.png")
    
    to_gmad(arc, args.to_gmad)

    if args.to_json:
        save_model(model, args.to_json)

    print("Processing complete.")

if __name__ == "__main__":
    main()