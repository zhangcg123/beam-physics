#!/usr/bin/env python3
import argparse
import math
import re
import json
from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

with open('beam_param_map.json', 'r') as f:
    beam_param_map = json.load(f)

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

def beamparams_convt(param_name: str) -> str:
    return beam_param_map[param_name]
    
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
    # highlight marks
    mrk_x = [el.global_x for el in seq if el.sad_type.upper() == "MARK"]
    mrk_y = [el.global_y for el in seq if el.sad_type.upper() == "MARK"]
    mrk_n = [el.name for el in seq if el.sad_type.upper() == "MARK"]
    plt.scatter(
        mrk_x,
        mrk_y,
        s=10,
        c="red",
        alpha=1,
        label="MARK"
    )
    for i, name in enumerate(mrk_n):
        plt.text(mrk_x[i], mrk_y[i], name, fontsize=6, ha='right', va='bottom', rotation=30)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    #plt.title(f"Ring Layout: {line_name}")
    ax = plt.gca()
    ax.text(
        0.02, 0.98,
        f"NElE={len(seq)}  NBEND={sum(1 for el in seq if el.sad_type.upper()=='BEND')}\n"
        f"NQUA={sum(1 for el in seq if el.sad_type.upper()=='QUAD')}\n"
        f"NSEXT={sum(1 for el in seq if el.sad_type.upper()=='SEXT')}\n"
        f"NDRIFT={sum(1 for el in seq if el.sad_type.upper()=='DRIFT')}\n"
        f"NMULT={sum(1 for el in seq if el.sad_type.upper()=='MULT')}\n"
        f"NCAVI={sum(1 for el in seq if el.sad_type.upper()=='CAVI')}\n"
        f"NSOL={sum(1 for el in seq if el.sad_type.upper()=='SOL')}\n"
        f"NMARK={sum(1 for el in seq if el.sad_type.upper()=='MARK')}\n"
        f"DEBUG: Tot. L={sum(el.params.get('L', 0.0) for el in seq):.3f}  Tot. angle/2={sum(el.params.get('ANGLE', 0.0)/2. for el in seq):.6f} rad\n"
        f"          Begin: {seq[0].name}, {seq[0].sad_type}, s={seq[0].start_s:.3f}, x={seq[0].global_x:.3f}, y={seq[0].global_y:.3f}\n"
        f"          End: {seq[-1].name}, {seq[-1].sad_type}, s={seq[-1].start_s:.3f}, x={seq[-1].global_x:.3f}, y={seq[-1].global_y:.3f}",
        transform=ax.transAxes, fontsize=8, va="top", ha="left"
    )
    plt.savefig('debug_ring.png', dpi=300)
    plt.clf()
    print ( 'The starting point name: ', seq[0].name, ' type: ', seq[0].sad_type, ' at s= ', seq[0].start_s, ' x= ', seq[0].global_x, ' y= ', seq[0].global_y )
    print ( 'The ending point name: ', seq[-1].name, ' type: ', seq[-1].sad_type, ' at s= ', seq[-1].start_s, ' x= ', seq[-1].global_x, ' y= ', seq[-1].global_y )
    # slice by mark and cout the number of other elements types between two marks
    for i, el in enumerate(seq):
        if el.sad_type.upper() == "MARK":
            next_mark_index = [ j for j in range(i+1, len(seq)) if seq[j].sad_type.upper() == "MARK" ]
            if len(next_mark_index) == 0:
                next_mark_index = [ jj for jj in range(len(seq)) if seq[jj].sad_type.upper() == "MARK" ][0] # if the i is the last mark, loop back to the first mark
            else:
                next_mark_index = next_mark_index[0]
            count_bend = sum(1 for e in seq[i+1:next_mark_index] if e.sad_type.upper() == "BEND")
            count_quad = sum(1 for e in seq[i+1:next_mark_index] if e.sad_type.upper() == "QUAD")
            count_sext = sum(1 for e in seq[i+1:next_mark_index] if e.sad_type.upper() == "SEXT")
            count_drift = sum(1 for e in seq[i+1:next_mark_index] if e.sad_type.upper() == "DRIFT")
            count_mult = sum(1 for e in seq[i+1:next_mark_index] if e.sad_type.upper() == "MULT")
            count_cavi = sum(1 for e in seq[i+1:next_mark_index] if e.sad_type.upper() == "CAVI")
            count_sol = sum(1 for e in seq[i+1:next_mark_index] if e.sad_type.upper() == "SOL")
            count_mark = next_mark_index - (i + 1)
            print(f"From MARK {el.name} at s={el.start_s:.3f} m, to next MARK {seq[next_mark_index].name} at s={seq[next_mark_index].start_s:.3f} m: "
                  f"{count_bend} BEND, {count_quad} QUAD, {count_sext} SEXT, {count_drift} DRIFT, {count_mult} MULT, "
                  f"{count_cavi} CAVI, {count_sol} SOL, total {count_mark} elements.")
    '''
    # debug plot for IP region
    plt.scatter(
        [el.global_x for el in seq],
        [el.global_y for el in seq],
        s=1,
        c="blue",
        alpha=0.5,
    )
    plt.scatter(
        [-1*el.global_x for el in seq],
        [el.global_y for el in seq],
        s=1,
        c="red",
        alpha=0.5,
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.xlim(-1500, 1500)
    plt.ylim(-5, 20)
    plt.savefig('debug_ring_ip.png', dpi=300)
    plt.clf()
    '''
    return seq # the full sequence 

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
            if el.gmad_string:
                if el.name.lower() in readyname or el.sad_type.upper() == "MARK":
                    f.write(f"! {el.gmad_string} ! at s={el.start_s:.3f} m -- skip duplicate or skip mark\n")
                else:
                    f.write(f"{el.gmad_string} ! at s={el.start_s:.3f} m\n")
                    readyname.append(el.name.lower())
            else:
                f.write(f"! {el.name} of type {el.sad_type} not converted\n")
        f.write("\n")
    # the beam has been defined properly in the beginning mark
    with open(path+'_beam.gmad', 'w') as f:
        for el in seq:
            if el.sad_type == 'MARK' and seq.index(el) == 0:
                f.write(f"beam, particle=\"e-\", energy = 120.0*GeV, distrType = \"gausstwiss\", ! {el.name}\n")
                mark_params_str = ',\n'.join(f"    {beam_param_map[k]}={v}" for k, v in el.params.items() if beam_param_map[k] != "")
                f.write(f"{mark_params_str};\n")
            elif el.sad_type == 'MARK':
                f.write(f"!beam, particle=\"e-\", energy = 120.0*GeV, distrType = \"gausstwiss\", ! {el.name}\n")
                mark_params_str = ',\n'.join(f"!    {beam_param_map[k]}={v}" for k, v in el.params.items() if beam_param_map[k] != "")
                f.write(f"{mark_params_str};\n")
            else:
                continue
    
    with open(path+'_line.gmad', 'w') as f:
        line_str = ", ".join(el.name.lower() for el in seq if el.sad_type.upper() != "MARK" and el.sad_type.upper() != "SOL" and el.params.get("L", 0.0) != 0.0)
        f.write(f"ring: line = ({line_str});\n")
        f.write("use, period = ring;\n")
    with open(path+'_options.gmad', 'w') as f:
        f.write("! to be filled as needed\n")
        f.write("option, ngenerate=100, physicsList=\"synch_rad em\";\n")
    with open(path+'.gmad', 'w') as f:
        f.write(f"include {path}_components.gmad;\n")
        f.write(f"include {path}_beam.gmad;\n")
        f.write(f"include {path}_line.gmad;\n")
        f.write(f"include {path}_options.gmad;\n")
        f.write(f"sample, all;\n")

def process_arc(seq: List[Element], out_path: str) -> List[Element]:
    
    # arc by names
    arc_start_name = "MIRD"
    arc_end_name = "MSTRRDO"

    start_index = next((i for i, el in enumerate(seq) if el.name == arc_start_name), None)
    end_index = next((i for i, el in enumerate(seq) if el.name == arc_end_name), None)
    
    if start_index is None or end_index is None:
        raise RuntimeError(f"Arc start or end markers '{arc_start_name}' or '{arc_end_name}' not found in the sequence.")
    if start_index >= end_index:
        raise RuntimeError(f"Arc start marker '{arc_start_name}' occurs after end marker '{arc_end_name}' in the sequence.")
    arc_seq = seq[start_index:end_index + 1]
    for i, el in enumerate(arc_seq):
        el.start_s = sum(e.params.get("L", 0.0) for e in arc_seq[:i])
    print(f"Arc segment from '{arc_start_name}' to '{arc_end_name}' contains {len(arc_seq)} elements.")
    plt.scatter(
        [el.global_x for el in arc_seq],
        [el.global_y for el in arc_seq],
        s=1,
        c="black",
        alpha=0.5,
        label="Elements"
    )
    # highlight marks
    mrk_x = [el.global_x for el in arc_seq if el.sad_type.upper() == "MARK"]
    mrk_y = [el.global_y for el in arc_seq if el.sad_type.upper() == "MARK"]
    mrk_n = [el.name for el in arc_seq if el.sad_type.upper() == "MARK"]
    plt.scatter(
        mrk_x,
        mrk_y,
        s=10,
        c="red",
        alpha=1,
        label="MARK"
    )
    for i, name in enumerate(mrk_n):
        plt.text(mrk_x[i], mrk_y[i], name, fontsize=6, ha='right', va='bottom', rotation=30)
    plt.text(0.02, 0.98,
        f"NElE={len(arc_seq)}  NBEND={sum(1 for el in arc_seq if el.sad_type.upper()=='BEND')}\n"
        f"NQUA={sum(1 for el in arc_seq if el.sad_type.upper()=='QUAD')}\n"
        f"NSEXT={sum(1 for el in arc_seq if el.sad_type.upper()=='SEXT')}\n"
        f"NDRIFT={sum(1 for el in arc_seq if el.sad_type.upper()=='DRIFT')}\n"
        f"NMULT={sum(1 for el in arc_seq if el.sad_type.upper()=='MULT')}\n"
        f"NCAVI={sum(1 for el in arc_seq if el.sad_type.upper()=='CAVI')}\n"
        f"NSOL={sum(1 for el in arc_seq if el.sad_type.upper()=='SOL')}\n"
        f"NMARK={sum(1 for el in arc_seq if el.sad_type.upper()=='MARK')}\n"
        f"DEBUG: Tot. L={sum(el.params.get('L', 0.0) for el in arc_seq):.3f}  Tot. angle/2={sum(el.params.get('ANGLE', 0.0)/2. for el in arc_seq):.6f} rad\n"
        f"          Begin: {arc_seq[0].name}, {arc_seq[0].sad_type}, s={arc_seq[0].start_s:.3f}, x={arc_seq[0].global_x:.3f}, y={arc_seq[0].global_y:.3f}\n"
        f"          End: {arc_seq[-1].name}, {arc_seq[-1].sad_type}, s={arc_seq[-1].start_s:.3f}, x={arc_seq[-1].global_x:.3f}, y={arc_seq[-1].global_y:.3f}\n"
        , transform=plt.gca().transAxes, fontsize=8, va="top", ha="left"
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    #plt.title(f"Arc Layout")
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