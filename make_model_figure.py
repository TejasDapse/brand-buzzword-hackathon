#!/usr/bin/env python3
"""
make_model_figure.py
====================

Regenerates the DETAILED model figure for HangmanNet — the network only, at the
level of individual layers, tensor shapes and masks. The companion script
`make_architecture_figure.py` draws the whole pipeline (data, training,
decision rule, inference); this one zooms into `class HangmanNet`.

Every shape and parameter count is derived from `hangman_kaggle_notebook.ipynb`
cell 5 (`hangman.py`) under the `Config` built in cell 8.

Usage:
    python make_model_figure.py

Writes:
    hangman_model_architecture.png   (220 dpi)
    hangman_model_architecture.svg
    hangman_model_architecture.pdf

Dependencies: matplotlib + numpy only. Deterministic, no network, no assets.

NOTE: the drawing helpers below are kept byte-identical to those in
`make_architecture_figure.py` so the two figures share one visual language.
Each script stays self-contained on purpose — they are figure sources, not a
library, and a shared import would couple two deliverables that are meant to
be runnable on their own.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle, Wedge,
)
from matplotlib.path import Path

# --------------------------------------------------------------------------
# Canvas constants and palette
# --------------------------------------------------------------------------

W, H = 201.0, 134.0          # abstract unit coordinate space (aspect 1.5)
FIG_W, FIG_H = 24.0, 16.0    # inches (aspect 1.5)

INK = "#12161F"              # element titles
MUTED = "#5A6478"            # sub-lines
RULE = "#B9C0CC"             # hairlines
LEGEND_BG = "#F1F3F6"

HUE = {
    "input":     "#1F3864",  # navy
    "encode":    "#2E5FA3",  # blue
    "sample":    "#E8A33D",  # amber
    "backbone":  "#C0492B",  # rust
    "heads":     "#5B3E90",  # purple
    "loss":      "#8E1B2E",  # dark red
    "train":     "#2E7D4F",  # green
    "decide":    "#187B6E",  # teal
    "infer":     "#44546A",  # slate
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "mathtext.fontset": "dejavuserif",
    "mathtext.default": "it",
    "path.simplify": False,
})

# font sizes
FS_PANEL = 12.0
FS_PANEL_SUB = 7.6
FS_TITLE = 8.6
FS_SUB = 6.6
FS_TINY = 6.0
FS_TAG = 6.4
FS_MATH = 8.2


def tint(hex_color: str, alpha: float) -> tuple:
    """Blend a hue toward white; returns an opaque RGB so overlaps never darken."""
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple(rgb * alpha + np.ones(3) * (1.0 - alpha))


# --------------------------------------------------------------------------
# Primitive helpers
# --------------------------------------------------------------------------


def txt(ax, x, y, s, size=FS_SUB, color=MUTED, weight="normal", ha="left",
        va="center", style="normal", family=None, zorder=6):
    kw = {}
    if family:
        kw["family"] = family
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, style=style, zorder=zorder, **kw)


def panel(ax, x0, y0, x1, y1, number, title, hue, subtitle=None, icon=None):
    """Draw panel chrome. (x0,y0) bottom-left, (x1,y1) top-right.

    Returns the inner body rectangle (x0, y0, x1, y_body_top) for content.
    """
    head_h = 6.2 if subtitle else 4.6
    body_top = y1 - head_h

    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=1.2",
        linewidth=1.0, edgecolor=hue, facecolor="#FFFFFF", zorder=1))

    ax.add_patch(FancyBboxPatch(
        (x0, body_top), x1 - x0, head_h,
        boxstyle="round,pad=0,rounding_size=1.2",
        linewidth=0, facecolor=hue, zorder=2))
    # square off the bottom edge of the header bar
    ax.add_patch(Rectangle((x0, body_top), x1 - x0, head_h * 0.45,
                           linewidth=0, facecolor=hue, zorder=2))

    tx = x0 + 2.0
    if icon is not None:
        icon(ax, x0 + 2.6, y1 - head_h + head_h * (0.72 if subtitle else 0.5), 2.5)
        tx = x0 + 5.8

    ty = y1 - head_h * (0.33 if subtitle else 0.5)
    txt(ax, tx, ty, f"{number}.  {title}", size=FS_PANEL, color="#FFFFFF",
        weight="bold", va="center", zorder=7)
    if subtitle:
        txt(ax, tx, y1 - head_h * 0.74, subtitle, size=FS_PANEL_SUB,
            color="#E6EAF2", weight="bold", va="center", zorder=7)

    return x0, y0, x1, body_top


FIT_WARNINGS = []

PAD = 0.9          # inner padding of an element box
TITLE_DROP = 0.7   # title baseline below the box top padding
FIRST_GAP = 1.50   # title -> first sub-line
LINE_STEP = 1.30   # sub-line pitch


def box_height(n_lines: int, has_title: bool = True) -> float:
    """Minimum box height that fits a title plus `n_lines` sub-lines."""
    if not has_title:
        return 2 * PAD + max(1, n_lines) * LINE_STEP + 0.2
    if n_lines == 0:
        return 2 * PAD + 1.6
    return 2 * PAD + TITLE_DROP + 0.45 + FIRST_GAP + (n_lines - 1) * LINE_STEP


def box(ax, x, y, w, h, title=None, lines=(), hue="#44546A", dashed=False,
        fill=0.10, title_size=FS_TITLE, sub_size=FS_SUB, align="left",
        title_color=None, pad=PAD, tag=""):
    """Element box. Returns a dict of edge anchors so connectors are derived
    from geometry instead of being hard-coded twice.

    Records a warning if the requested height cannot hold the text, so layout
    bugs surface as console output rather than as silent overflow.
    """
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.7",
        linewidth=1.0, edgecolor=hue, facecolor=tint(hue, fill),
        linestyle=(0, (2.2, 1.6)) if dashed else "solid", zorder=3))

    lines = [s for s in (lines or ()) if s]
    n = len(lines)
    need = box_height(n, has_title=title is not None)
    if h < need - 1e-6:
        FIT_WARNINGS.append(
            f"box{'' if not tag else ' ' + tag} at ({x:.1f},{y:.1f}) h={h:.2f} "
            f"needs {need:.2f} for {n} sub-line(s)"
            + (f" — {title[:34]}" if title else ""))

    tx, ha = (x + w / 2.0, "center") if align == "center" else (x + pad, "left")

    if title is not None:
        t_y = y + h - pad - TITLE_DROP
        txt(ax, tx, t_y, title, size=title_size, color=title_color or INK,
            weight="bold", ha=ha, va="center")
        for i, line in enumerate(lines):
            txt(ax, tx, t_y - FIRST_GAP - i * LINE_STEP, line, size=sub_size,
                color=MUTED, ha=ha, va="center")
    else:
        top = y + h / 2.0 + (n - 1) * LINE_STEP / 2.0
        for i, line in enumerate(lines):
            txt(ax, tx, top - i * LINE_STEP, line, size=sub_size,
                color=MUTED, ha=ha, va="center")

    return {"x": x, "y": y, "w": w, "h": h,
            "l": (x, y + h / 2.0), "r": (x + w, y + h / 2.0),
            "t": (x + w / 2.0, y + h), "b": (x + w / 2.0, y),
            "c": (x + w / 2.0, y + h / 2.0),
            "x1": x + w, "y1": y + h}


def cuboid(ax, x, y, w, h, depth, hue, label=None, channels=None,
           label_size=FS_TINY):
    """3-D isometric tensor block: width to spatial size, depth to channels."""
    dx, dy = depth * 0.62, depth * 0.42
    face = tint(hue, 0.22)
    top = tint(hue, 0.40)
    side = tint(hue, 0.55)

    ax.add_patch(Polygon([(x, y + h), (x + dx, y + h + dy),
                          (x + w + dx, y + h + dy), (x + w, y + h)],
                         closed=True, facecolor=top, edgecolor=hue,
                         linewidth=0.8, zorder=3))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy),
                          (x + w + dx, y + h + dy), (x + w, y + h)],
                         closed=True, facecolor=side, edgecolor=hue,
                         linewidth=0.8, zorder=3))
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=hue,
                           linewidth=0.9, zorder=4))

    cx = x + (w + dx) / 2.0
    if label:
        txt(ax, cx, y + h + dy + 1.3, label, size=label_size, color=INK,
            weight="bold", ha="center")
    if channels:
        txt(ax, cx, y - 1.5, channels, size=label_size, color=MUTED,
            ha="center")

    return {"x": x, "y": y, "w": w, "h": h,
            "l": (x, y + h / 2.0), "r": (x + w + dx, y + h / 2.0 + dy / 2.0),
            "t": (cx, y + h + dy), "b": (x + w / 2.0, y),
            "c": (cx, y + h / 2.0)}


def arrow(ax, p0, p1, color="#12161F", lw=1.0, dashed=False, mutation=7.0,
          zorder=5, shrink=0.0):
    style = (0, (2.6, 1.8)) if dashed else "solid"
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=mutation,
        linewidth=lw, color=color, linestyle=style,
        shrinkA=shrink, shrinkB=shrink, zorder=zorder,
        joinstyle="miter", capstyle="butt"))


def elbow(ax, p0, p1, lift=None, top=None, color=INK, lw=0.9, r=1.1,
          dashed=False, arrowhead=True, zorder=5):
    """Rounded elbow: up out of p0, across, then down into p1.

    Give either `top` (the absolute height of the crossing lane) or `lift`
    (height above the higher endpoint). `top` is preferred, so several elbows
    can share one lane regardless of where their endpoints sit.
    """
    (x0, y0), (x1, y1) = p0, p1
    ytop = top if top is not None else max(y0, y1) + lift
    sx = 1.0 if x1 > x0 else -1.0
    verts = [(x0, y0),
             (x0, ytop - r), (x0, ytop), (x0 + sx * r, ytop),
             (x1 - sx * r, ytop), (x1, ytop), (x1, ytop - r),
             (x1, y1)]
    codes = [Path.MOVETO,
             Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO]
    path = Path(verts, codes)
    style = (0, (2.6, 1.8)) if dashed else "solid"
    if arrowhead:
        ax.add_patch(FancyArrowPatch(
            path=path, arrowstyle="-|>", mutation_scale=7.0, linewidth=lw,
            color=color, linestyle=style, zorder=zorder,
            shrinkA=0, shrinkB=0))
    else:
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color,
                               linewidth=lw, linestyle=style, zorder=zorder))
    return path


def op(ax, x, y, symbol, hue=INK, r=1.35, size=8.0):
    """Small circled operator: add, sigmoid, product."""
    ax.add_patch(Circle((x, y), r, facecolor="#FFFFFF", edgecolor=hue,
                        linewidth=1.0, zorder=6))
    ax.text(x, y, symbol, fontsize=size, color=hue, ha="center", va="center",
            zorder=7, fontweight="bold")
    return {"c": (x, y), "l": (x - r, y), "r": (x + r, y),
            "t": (x, y + r), "b": (x, y - r), "r_": r}


def shape_tag(ax, x, y, s, ha="center", va="bottom", size=FS_TAG):
    """Italic serif tensor-shape tag sitting on a connector."""
    ax.text(x, y, s, fontsize=size, color="#33405A", ha=ha, va=va,
            zorder=7, style="italic", family="serif")


def rule_label(ax, x0, x1, label, y, hue):
    """Small-caps sub-branch header with a thin horizontal rule."""
    txt(ax, (x0 + x1) / 2.0, y, label, size=7.0, color=hue, weight="bold",
        ha="center")
    ax.plot([x0, x1], [y - 1.5, y - 1.5], color=tint(hue, 0.45), linewidth=0.8,
            zorder=3)


# --------------------------------------------------------------------------
# Header pictograms (flat, built from matplotlib primitives)
# --------------------------------------------------------------------------


def _ic_kw(s=1.0):
    return dict(color="#FFFFFF", linewidth=1.0 * s, zorder=8,
                solid_capstyle="round")


def icon_book(ax, x, y, s):
    ax.plot([x - s * .6, x - s * .6], [y - s * .55, y + s * .55], **_ic_kw())
    for dy in (-.3, 0.0, .3):
        ax.plot([x - s * .35, x + s * .6], [y + s * dy, y + s * dy], **_ic_kw(.8))


def icon_grid(ax, x, y, s):
    for i in range(4):
        gx = x - s * .65 + i * s * .43
        ax.add_patch(Rectangle((gx, y - s * .35), s * .3, s * .7,
                               facecolor="none" if i % 2 else "#FFFFFF",
                               edgecolor="#FFFFFF", linewidth=0.9, zorder=8))


def icon_dice(ax, x, y, s):
    ax.add_patch(FancyBboxPatch((x - s * .5, y - s * .5), s, s,
                                boxstyle="round,pad=0,rounding_size=0.25",
                                facecolor="none", edgecolor="#FFFFFF",
                                linewidth=1.0, zorder=8))
    for dx, dy in ((-.24, .24), (.24, -.24), (0, 0)):
        ax.add_patch(Circle((x + s * dx, y + s * dy), s * .09,
                            facecolor="#FFFFFF", edgecolor="none", zorder=8))


def icon_stack(ax, x, y, s):
    for i, dy in enumerate((-.42, 0.0, .42)):
        ax.add_patch(Polygon([(x - s * .6, y + s * dy), (x, y + s * (dy + .24)),
                              (x + s * .6, y + s * dy), (x, y + s * (dy - .24))],
                             closed=True, facecolor="none", edgecolor="#FFFFFF",
                             linewidth=0.9, zorder=8))


def icon_fork(ax, x, y, s):
    ax.plot([x - s * .6, x], [y, y], **_ic_kw())
    ax.plot([x, x + s * .55], [y, y + s * .5], **_ic_kw())
    ax.plot([x, x + s * .55], [y, y], **_ic_kw())
    ax.plot([x, x + s * .55], [y, y - s * .5], **_ic_kw())
    for dy in (.5, 0.0, -.5):
        ax.add_patch(Circle((x + s * .62, y + s * dy), s * .12,
                            facecolor="#FFFFFF", edgecolor="none", zorder=8))


def icon_target(ax, x, y, s):
    ax.add_patch(Circle((x, y), s * .55, facecolor="none", edgecolor="#FFFFFF",
                        linewidth=1.0, zorder=8))
    ax.add_patch(Circle((x, y), s * .22, facecolor="#FFFFFF", edgecolor="none",
                        zorder=8))


def icon_calendar(ax, x, y, s):
    ax.add_patch(Rectangle((x - s * .55, y - s * .5), s * 1.1, s * .95,
                           facecolor="none", edgecolor="#FFFFFF",
                           linewidth=1.0, zorder=8))
    ax.plot([x - s * .55, x + s * .55], [y + s * .18, y + s * .18], **_ic_kw(.8))
    ax.plot([x - s * .28, x - s * .28], [y + s * .45, y + s * .62], **_ic_kw(.8))
    ax.plot([x + s * .28, x + s * .28], [y + s * .45, y + s * .62], **_ic_kw(.8))


def icon_scale(ax, x, y, s):
    ax.plot([x, x], [y - s * .5, y + s * .5], **_ic_kw())
    ax.plot([x - s * .6, x + s * .6], [y + s * .35, y + s * .35], **_ic_kw())
    for sx in (-1, 1):
        ax.plot([x + sx * s * .6, x + sx * s * .6], [y + s * .35, y + s * .05],
                **_ic_kw(.8))
        ax.plot([x + sx * s * .82, x + sx * s * .38], [y + s * .05, y + s * .05],
                **_ic_kw(.8))


def icon_play(ax, x, y, s):
    ax.add_patch(Polygon([(x - s * .42, y - s * .5), (x - s * .42, y + s * .5),
                          (x + s * .5, y)], closed=True, facecolor="#FFFFFF",
                         edgecolor="none", zorder=8))

def icon_sum(ax, x, y, s):
    ax.plot([x - s * .5, x + s * .5], [y, y], **_ic_kw())
    ax.plot([x, x], [y - s * .5, y + s * .5], **_ic_kw())
    ax.add_patch(Circle((x, y), s * .62, facecolor="none", edgecolor="#FFFFFF",
                        linewidth=0.9, zorder=8))


def icon_kernels(ax, x, y, s):
    for i, w in enumerate((.25, .42, .6)):
        ax.add_patch(Rectangle((x - s * w, y + s * (.34 - i * .34)), s * w * 2,
                               s * .22, facecolor="none", edgecolor="#FFFFFF",
                               linewidth=0.8, zorder=8))


def icon_chain(ax, x, y, s):
    for dx in (-.34, .34):
        ax.add_patch(Circle((x + s * dx, y), s * .3, facecolor="none",
                            edgecolor="#FFFFFF", linewidth=1.0, zorder=8))
    ax.plot([x - s * .06, x + s * .06], [y, y], **_ic_kw())
    ax.plot([x - s * .64, x - s * .64], [y - s * .3, y + s * .3], **_ic_kw(.8))
    ax.plot([x + s * .64, x + s * .64], [y - s * .3, y + s * .3], **_ic_kw(.8))


def icon_lattice(ax, x, y, s):
    for i in range(3):
        for j in range(3):
            ax.add_patch(Rectangle((x - s * .6 + i * s * .42,
                                    y - s * .6 + j * s * .42),
                                   s * .3, s * .3,
                                   facecolor="#FFFFFF" if (i + j) % 2 == 0 else "none",
                                   edgecolor="#FFFFFF", linewidth=0.6, zorder=8))


def icon_layers(ax, x, y, s):
    for i, dy in enumerate((-.36, 0.0, .36)):
        ax.add_patch(Rectangle((x - s * .6, y + s * dy - s * .11), s * 1.2,
                               s * .22, facecolor="none", edgecolor="#FFFFFF",
                               linewidth=0.8, zorder=8))


# --------------------------------------------------------------------------
# Model constants (all traced to hangman.py under the cell-8 Config)
# --------------------------------------------------------------------------

D_MODEL = 320
N_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS          # 40
FFN_DIM = 4 * D_MODEL                  # 1280
CONV_OUT = D_MODEL // 3                # 106
CONV_CAT = CONV_OUT * 3                # 318
LSTM_HIDDEN = D_MODEL // 2             # 160
TOTAL_PARAMS = 6_022_131

LEDGER = [
    ("char_embedding(29, 320)", "9,280"),
    ("position_embedding(48, 320)", "15,360"),
    ("length_embedding(49, 320)", "15,680"),
    ("guessed_projection  26 → 320", "8,640"),
    ("input LayerNorm(320)", "640"),
    ("Conv1d  320 → 106,  k = 3", "101,866"),
    ("Conv1d  320 → 106,  k = 5", "169,706"),
    ("Conv1d  320 → 106,  k = 7", "237,546"),
    ("conv_projection  318 → 320", "102,080"),
    ("conv LayerNorm(320)", "640"),
    ("BiLSTM,  2 layers,  hidden 160", "1,233,920"),
    ("lstm LayerNorm(320)", "640"),
    ("TransformerEncoderLayer × 3", "3,698,880"),
    ("position_head  320 → 26", "8,346"),
    ("set_head  640 → 320 → 26", "213,466"),
    ("value_head  640 → 320 → 1", "205,441"),
]

HUE.update({
    "spine":  "#1F3864",
    "inputs": "#2E5FA3",
    "compose": "#5B3E90",
    "conv":   "#C0492B",
    "lstm":   "#E8A33D",
    "trans":  "#2E7D4F",
    "attn":   "#187B6E",
    "heads":  "#8E1B2E",
})

# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#FFFFFF")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
ax.set_facecolor("#FFFFFF")

txt(ax, W / 2, 130.6,
    "HANGMANNET — DETAILED MODEL ARCHITECTURE",
    size=17.0, color=INK, weight="bold", ha="center")
txt(ax, W / 2, 127.0,
    "d_model = 320  ·  dropout = 0.15  ·  6,022,131 parameters   ·   "
    "B = batch size, L = padded word length in the batch   ·   "
    "shapes and counts traced to class HangmanNet",
    size=8.4, color=MUTED, ha="center")
ax.plot([3, 198], [125.0, 125.0], color=RULE, linewidth=0.9, zorder=1)

# ==========================================================================
# 1.  FORWARD SPINE  +  PARAMETER LEDGER
# ==========================================================================
panel(ax, 3, 15, 48, 124, 1, "FORWARD SPINE", HUE["spine"],
      subtitle="(ONE PASS, TOP TO BOTTOM)", icon=icon_layers)
hs = HUE["spine"]
SX0, SXW = 5, 37                 # spine box geometry
SCX = SX0 + SXW / 2              # 23.5 — every spine element is centred here


CUB_W, CUB_H, CUB_D = 9.0, 5.5, 3.5
CUB_PEAK = CUB_H + CUB_D * 0.42          # front-face top + top-face depth


def spine_cuboid(y, tag):
    """A tensor block on the spine. Returns (peak_y, base_y) so connectors
    attach to the drawn silhouette rather than to the front face alone."""
    cuboid(ax, 19.0, y, CUB_W, CUB_H, CUB_D, hs)
    shape_tag(ax, 33.5, y + 1.6, tag, ha="left")
    return y + CUB_PEAK, y


e1 = box(ax, SX0, 110.75, SXW, 7.05, "INPUTS",
         ["chars   (B, L)  int64", "guessed   (B, 26)  float32",
          "lengths   (B,)  int64"], hs, tag="e1")
e2 = box(ax, SX0, 105.05, SXW, 4.45, "①  INPUT COMPOSITION   →  panel 3",
         ["4 additive streams → LayerNorm → × valid"], hs, sub_size=FS_TINY,
         tag="e2")
arrow(ax, (SCX, e1["y"]), (SCX, e2["y1"]), lw=0.9)
c1_top, c1_bot = spine_cuboid(96.8, r"$B \times L \times 320$")
arrow(ax, (SCX, e2["y"]), (SCX, c1_top), lw=0.9)

e3 = box(ax, SX0, 90.8, SXW, 4.45, "②  CONV FRONT-END   →  panel 4",
         ["multi-scale conv, residual + LayerNorm"], hs, sub_size=FS_TINY,
         tag="e3")
arrow(ax, (SCX, c1_bot), (SCX, e3["y1"]), lw=0.9)
c2_top, c2_bot = spine_cuboid(82.55, r"$B \times L \times 320$")
arrow(ax, (SCX, e3["y"]), (SCX, c2_top), lw=0.9)

e4 = box(ax, SX0, 76.55, SXW, 4.45, "③  BiLSTM   →  panel 5",
         ["2 bidirectional layers, residual + LayerNorm"], hs, sub_size=FS_TINY,
         tag="e4")
arrow(ax, (SCX, c2_bot), (SCX, e4["y1"]), lw=0.9)
c3_top, c3_bot = spine_cuboid(68.3, r"$B \times L \times 320$")
arrow(ax, (SCX, e4["y"]), (SCX, c3_top), lw=0.9)

e5 = box(ax, SX0, 62.3, SXW, 4.45, "④  TRANSFORMER  × 3   →  panels 6 · 7",
         ["pre-norm self-attention + position-wise FFN"], hs, sub_size=FS_TINY,
         tag="e5")
arrow(ax, (SCX, c3_bot), (SCX, e5["y1"]), lw=0.9)
c4_top, c4_bot = spine_cuboid(54.05, r"$B \times L \times 320$")
arrow(ax, (SCX, e5["y"]), (SCX, c4_top), lw=0.9)

e6 = box(ax, SX0, 45.45, SXW, 7.05, "⑤  THREE HEADS   →  panel 8",
         ["positional  →  (B, L, 26)", "set  →  (B, 26)", "value  →  (B,)"],
         hs, tag="e6")
arrow(ax, (SCX, c4_bot), (SCX, e6["y1"]), lw=0.9)

# ---- parameter ledger ----
rule_label(ax, 5, 46, "PARAMETER LEDGER", 43.5, hs)
_ly = 41.5
txt(ax, 5.6, _ly, "LAYER", size=5.8, color=INK, weight="bold")
txt(ax, 45.4, _ly, "PARAMETERS", size=5.8, color=INK, weight="bold", ha="right")
for i, (name, count) in enumerate(LEDGER):
    ry = _ly - 1.45 * (i + 1)
    if i % 2 == 0:
        ax.add_patch(Rectangle((5, ry - 0.68), 41, 1.36, linewidth=0,
                               facecolor=tint(hs, 0.055), zorder=2))
    txt(ax, 5.6, ry, name, size=5.6, color=MUTED)
    txt(ax, 45.4, ry, count, size=5.6, color=INK, ha="right")
_ty = _ly - 1.45 * (len(LEDGER) + 1)
ax.add_patch(Rectangle((5, _ty - 0.78), 41, 1.56, linewidth=0,
                       facecolor=tint(hs, 0.20), zorder=2))
txt(ax, 5.6, _ty, "TOTAL", size=6.0, color=INK, weight="bold")
txt(ax, 45.4, _ty, f"{TOTAL_PARAMS:,}", size=6.0, color=INK, weight="bold",
    ha="right")

# ==========================================================================
# 2.  INPUT TENSORS & MASKING
# ==========================================================================
panel(ax, 51, 99, 123, 124, 2, "INPUT TENSORS & MASKING", HUE["inputs"],
      icon=icon_grid)
hi = HUE["inputs"]
box(ax, 53, 114.95, 68, 4.45, "chars   (B, L)   int64",
    ["token ids: a–z → 0…25,  BLANK = 26 marks a hidden slot,  "
     "PAD = 28 fills past the word"], hi, sub_size=FS_TINY, tag="chars")
box(ax, 53, 109.63, 68, 4.45, "guessed   (B, 26)   float32",
    ["multi-hot over every letter already guessed this game — hits and misses alike"],
    hi, sub_size=FS_TINY, tag="guessed")
box(ax, 53, 104.31, 68, 4.45, "lengths   (B,)   int64",
    ["the true word length, clamped to max_len = 48 before the embedding lookup"],
    hi, sub_size=FS_TINY, tag="lengths")
box(ax, 53, 98.99, 68, 4.45, "THREE MASKS, ALL DERIVED FROM chars",
    ["pad_mask = chars == 28    ·    valid = (¬pad_mask).float()    ·    "
     "the transformer receives src_key_padding_mask = pad_mask"],
    hi, sub_size=FS_TINY, tag="masks")

# ==========================================================================
# 3.  INPUT COMPOSITION
# ==========================================================================
panel(ax, 126, 99, 198, 124, 3, "INPUT COMPOSITION", HUE["compose"],
      icon=icon_sum)
hc = HUE["compose"]
streams = [
    ("char_embedding(29 → 320)",
     "(B, L, 320)   ·   padding_idx = 28 keeps this table's PAD row at zero"),
    ("position_embedding(48 → 320)",
     "(1, L, 320)   ·   broadcast over the batch"),
    ("length_embedding(49 → 320)",
     "(B, 1, 320)   ·   broadcast over the sequence"),
    ("guessed_projection   Linear(26 → 320)",
     "(B, 1, 320)   ·   broadcast over the sequence"),
]
oc = op(ax, 167, 109.2, "⊕", hue=hc)
for i, (title, line) in enumerate(streams):
    by = 114.95 - i * 5.3
    b = box(ax, 128, by, 34, 4.45, title, [line], hc, title_size=7.4,
            sub_size=5.6, tag=f"stream{i}")
    arrow(ax, (b["x1"], b["c"][1]), (oc["c"][0] - oc["r_"] - 0.15, oc["c"][1]),
          lw=0.8, mutation=6.0)
b_ln = box(ax, 171, 106.7, 25, 5.0, None,
           ["LayerNorm(320)", "× valid  ⇒  PAD rows zeroed"], hc,
           sub_size=6.0, align="center", tag="compose_ln")
arrow(ax, (oc["r"][0] + 0.15, oc["c"][1]), (b_ln["x"], oc["c"][1]), lw=0.9)
shape_tag(ax, 183.5, 104.4, r"$B \times L \times 320$")

# ==========================================================================
# 4.  MULTI-SCALE CONV FRONT-END   (full width)
# ==========================================================================
panel(ax, 51, 72, 198, 96, 4, "MULTI-SCALE CONV FRONT-END", HUE["conv"],
      icon=icon_kernels)
hv = HUE["conv"]
CY = 82.125                      # the conv row's centre line

shape_tag(ax, 53, 85.2, r"$B \times L \times 320$", ha="left")
arrow(ax, (51, CY), (54, CY), lw=0.9)
s1 = box(ax, 54, 79.9, 16, 4.45, "transpose(1, 2)", ["→  (B, 320, L)"], hv,
         title_size=7.4, sub_size=5.8, align="center", tag="tr1")
arrow(ax, (s1["x1"], CY), (72.5, CY), lw=0.9, mutation=0.1)
ax.plot([72.5, 72.5], [77.5, 86.7], color=INK, linewidth=0.9, zorder=5)

s_cat = box(ax, 101, 76.5, 12, 11.0, None,
            ["concat on", "the channel axis", "(B, 318, L)"], hv,
            sub_size=5.6, align="center", tag="cat")
for i, k in enumerate((3, 5, 7)):
    by = 85.0 - i * 4.6
    b = box(ax, 75, by, 24, 3.4, f"Conv1d(320 → 106)   k = {k},  pad = {k // 2}",
            (), hv, title_size=6.6, tag=f"k{k}")
    arrow(ax, (72.5, b["c"][1]), (b["x"], b["c"][1]), lw=0.8, mutation=6.0)
    arrow(ax, (b["x1"], b["c"][1]), (s_cat["x"], b["c"][1]), lw=0.8, mutation=6.0)

s4 = box(ax, 115, 79.9, 16, 4.45, "transpose(1, 2)", ["→  (B, L, 318)"], hv,
         title_size=7.4, sub_size=5.8, align="center", tag="tr2")
arrow(ax, (s_cat["x1"], CY), (s4["x"], CY), lw=0.9)
s5 = box(ax, 133, 79.9, 22, 4.45, "conv_projection   Linear(318 → 320)",
         ["GELU   →   Dropout 0.15"], hv, title_size=7.0, sub_size=5.8,
         align="center", tag="proj")
arrow(ax, (s4["x1"], CY), (s5["x"], CY), lw=0.9)
ov = op(ax, 159, CY, "⊕", hue=hv)
arrow(ax, (s5["x1"], CY), (ov["c"][0] - ov["r_"] - 0.15, CY), lw=0.9)
s7 = box(ax, 163, 79.9, 22, 4.45, "conv_norm   LayerNorm(320)",
         ["→  (B, L, 320)"], hv, title_size=7.0, sub_size=5.8, align="center",
         tag="convnorm")
arrow(ax, (ov["r"][0] + 0.15, CY), (s7["x"], CY), lw=0.9)
arrow(ax, (s7["x1"], CY), (194, CY), lw=0.9)
shape_tag(ax, 189.5, 83.2, r"$B \times L \times 320$")
elbow(ax, (52.5, CY), (159, ov["t"][1] + 0.15), top=89.8, color=hv, lw=0.9,
      r=1.2)
txt(ax, 54, 73.8, "padding = k // 2 keeps the sequence at length L, so the three "
    "branches concatenate cleanly on the channel axis;  the trailing [:, :, :L] "
    "slice is a no-op for odd kernels and a guard for even ones.",
    size=5.8, color=MUTED)

# ==========================================================================
# 5.  BiLSTM STACK
# ==========================================================================
panel(ax, 51, 45, 123, 69, 5, "BiLSTM STACK", HUE["lstm"],
      subtitle="(THE FULL SEQUENCE, NOT THE FINAL HIDDEN STATE)",
      icon=icon_chain)
hl = HUE["lstm"]
l1 = box(ax, 53, 57.05, 68, 5.75, "LAYER 1  —  bidirectional",
         ["forward LSTM 320 → 160    ‖    backward LSTM 320 → 160    ⇒    "
          "concat per step → (B, L, 320)",
          "Dropout 0.15 is applied between the two layers (the nn.LSTM dropout argument)"],
         hl, sub_size=FS_TINY, tag="l1")
l2 = box(ax, 53, 51.4, 68, 4.45, "LAYER 2  —  bidirectional",
         ["forward LSTM 320 → 160    ‖    backward LSTM 320 → 160    ⇒    "
          "concat per step → (B, L, 320)"], hl, sub_size=FS_TINY, tag="l2")
arrow(ax, (87, l1["y"]), (87, l2["y1"]), lw=0.9)
l3 = box(ax, 53, 45.75, 68, 4.45, "RESIDUAL  +  NORM",
         ["x  ←  lstm_norm( x + Dropout 0.15(lstm_out) )    ⇒    (B, L, 320)"],
         hl, sub_size=FS_TINY, tag="l3")
arrow(ax, (87, l2["y"]), (87, l3["y1"]), lw=0.9)

# ==========================================================================
# 6.  TRANSFORMER ENCODER LAYER
# ==========================================================================
panel(ax, 126, 45, 198, 69, 6, "TRANSFORMER ENCODER LAYER", HUE["trans"],
      subtitle="(PRE-NORM  ·  REPEATED × 3  ·  1,232,960 PARAMS EACH)",
      icon=icon_layers)
ht = HUE["trans"]
t1 = box(ax, 128, 58.35, 68, 4.45, "SUB-BLOCK 1  —  SELF-ATTENTION",
         ["x  ←  x + Dropout 0.15( MHA( LayerNorm(x) ) )    ·    8 heads    ·    "
          "expanded in panel 7"], ht, sub_size=FS_TINY, tag="t1")
t2 = box(ax, 128, 51.4, 68, 5.75, "SUB-BLOCK 2  —  POSITION-WISE FFN",
         ["h = LayerNorm(x)  →  Linear(320 → 1280)  →  GELU  →  Dropout 0.15",
          "→  Linear(1280 → 320)  →  Dropout 0.15,   then   x  ←  x + h"],
         ht, sub_size=FS_TINY, tag="t2")
arrow(ax, (162, t1["y"]), (162, t2["y1"]), lw=0.9)
box(ax, 128, 45.75, 68, 4.45, "NO FINAL NORM AFTER THE STACK",
    ["nn.TransformerEncoder is built without a norm= argument, so this pre-norm "
     "stack ends unnormalised"], ht, sub_size=FS_TINY, tag="t3")

# ==========================================================================
# 7.  MULTI-HEAD SELF-ATTENTION
# ==========================================================================
panel(ax, 51, 15, 123, 42, 7, "MULTI-HEAD SELF-ATTENTION", HUE["attn"],
      subtitle="(d = 320  ·  8 HEADS  ·  40 DIMS PER HEAD)", icon=icon_lattice)
ha_ = HUE["attn"]
a1 = box(ax, 53, 31.35, 68, 4.45, "①  PROJECT, THEN SPLIT INTO HEADS",
         ["in_proj Linear(320 → 960)  ⇒  Q, K, V each (B, L, 320)  →  "
          "reshape to (B, 8, L, 40)"], ha_, sub_size=FS_TINY, tag="a1")
a2 = box(ax, 53, 25.7, 68, 4.45, "②  SCALED DOT-PRODUCT SCORES",
         ["Q Kᵀ / √40   ⇒   (B, 8, L, L);   src_key_padding_mask sets every PAD "
          "column to −∞"], ha_, sub_size=FS_TINY, tag="a2")
a3 = box(ax, 53, 20.05, 68, 4.45, "③  ATTEND, THEN MERGE THE HEADS",
         ["softmax over keys  →  Dropout 0.15  →  A V   ⇒   (B, 8, L, 40)  →  "
          "merge  ⇒  (B, L, 320)"], ha_, sub_size=FS_TINY, tag="a3")
a4 = box(ax, 53, 15.4, 68, 3.4,
         "④  out_proj   Linear(320 → 320)    ⇒    (B, L, 320)", (), ha_,
         title_size=7.4, tag="a4")
for u, v in zip([a1, a2, a3], [a2, a3, a4]):
    arrow(ax, (87, u["y"]), (87, v["y1"]), lw=0.9)

# ==========================================================================
# 8.  MASKED POOLING & THE THREE HEADS
# ==========================================================================
panel(ax, 126, 15, 198, 42, 8, "MASKED POOLING & THE THREE HEADS",
      HUE["heads"], subtitle="(x ← x × valid FIRST, SO PAD SLOTS CONTRIBUTE NOTHING)",
      icon=icon_fork)
hh = HUE["heads"]
box(ax, 128, 31.35, 68, 4.45, "(A)  POSITIONAL HEAD  —  one prediction per slot",
    ["Dropout 0.15  →  Linear(320 → 26)   ⇒   (B, L, 26),  softmax taken per slot"],
    hh, sub_size=FS_TINY, tag="h_pos")
b_pool = box(ax, 128, 24.4, 68, 5.75, "(B)  MASKED POOLING",
             ["mean = x.sum(1) / valid.sum(1).clamp(min=1)    ·    "
              "max over valid slots, PAD filled with −1e4",
              "concat[ mean, max ]   ⇒   (B, 640) — the input to both global heads"],
             hh, sub_size=FS_TINY, tag="h_pool")
ax.plot([162, 162], [b_pool["y"], 23.0], color=INK, linewidth=0.9, zorder=5)
ax.plot([144.5, 179.5], [23.0, 23.0], color=INK, linewidth=0.9, zorder=5)
arrow(ax, (144.5, 23.0), (144.5, 21.95), lw=0.9)
arrow(ax, (179.5, 23.0), (179.5, 21.95), lw=0.9)
box(ax, 128, 17.5, 33, 4.45, "(C)  SET HEAD",
    ["640 → 320 → GELU → Dropout → 26,  then sigmoid"], hh, sub_size=FS_TINY,
    tag="h_set")
box(ax, 163, 17.5, 33, 4.45, "(D)  VALUE HEAD",
    ["640 → 320 → GELU → Dropout → 1,  then sigmoid"], hh, sub_size=FS_TINY,
    tag="h_val")
txt(ax, 128, 16.2, "The positional head reads the per-slot features directly; only "
    "the set and value heads go through the pooled summary.",
    size=5.8, color=MUTED)

# ==========================================================================
# LEGEND STRIP
# ==========================================================================
ax.add_patch(FancyBboxPatch((3, 2.0), 195, 11.0,
                            boxstyle="round,pad=0,rounding_size=1.0",
                            linewidth=0.9, edgecolor=RULE,
                            facecolor=LEGEND_BG, zorder=1))
ax.add_patch(FancyBboxPatch((5.0, 5.6), 12.0, 3.8,
                            boxstyle="round,pad=0,rounding_size=0.6",
                            linewidth=0, facecolor=INK, zorder=3))
txt(ax, 11.0, 7.5, "LEGEND", size=8.6, color="#FFFFFF", weight="bold",
    ha="center")

LEG_X0, LEG_W = 20.0, 19.5
LG = "#3B4557"


def _leg(i, caption_a, caption_b, draw):
    ex = LEG_X0 + i * LEG_W
    draw(ex + 3.4, 7.9)
    txt(ax, ex + 7.0, 8.7, caption_a, size=6.0, color=INK, weight="bold")
    txt(ax, ex + 7.0, 6.6, caption_b, size=5.8, color=MUTED)


def _g_header(x, y):
    ax.add_patch(FancyBboxPatch((x - 3.0, y - 2.2), 6.0, 4.4,
                                boxstyle="round,pad=0,rounding_size=0.5",
                                linewidth=0.9, edgecolor=LG,
                                facecolor="#FFFFFF", zorder=3))
    ax.add_patch(Rectangle((x - 3.0, y + 0.4), 6.0, 1.8, linewidth=0,
                           facecolor=LG, zorder=4))
    icon_layers(ax, x, y + 1.3, 1.2)


def _g_box(x, y):
    box(ax, x - 3.0, y - 2.0, 6.0, 4.0, None, (), LG, fill=0.10)
    ax.plot([x - 2.0, x + 1.4], [y + 0.9, y + 0.9], color=INK, linewidth=1.0,
            zorder=5)
    ax.plot([x - 2.0, x + 2.0], [y - 0.4, y - 0.4], color=RULE, linewidth=0.8,
            zorder=5)
    ax.plot([x - 2.0, x + 0.6], [y - 1.2, y - 1.2], color=RULE, linewidth=0.8,
            zorder=5)


def _g_cuboid(x, y):
    cuboid(ax, x - 2.8, y - 2.0, 3.4, 4.0, 2.4, LG)


def _g_arrow(x, y):
    arrow(ax, (x - 3.2, y), (x + 3.2, y), lw=1.0)


def _g_bus(x, y):
    ax.plot([x - 3.2, x - 1.4], [y, y], color=INK, linewidth=1.0, zorder=5)
    ax.plot([x - 1.4, x - 1.4], [y - 1.8, y + 1.8], color=INK, linewidth=1.0,
            zorder=5)
    for dy in (-1.8, 0.0, 1.8):
        arrow(ax, (x - 1.4, y + dy), (x + 3.0, y + dy), lw=0.8, mutation=5.0)


def _g_elbow(x, y):
    elbow(ax, (x - 3.0, y - 1.6), (x + 3.0, y - 1.6), lift=3.2, color=LG,
          lw=1.0, r=0.8)


def _g_plus(x, y):
    op(ax, x, y, "⊕", hue=LG, r=1.5, size=8.0)


def _g_tag(x, y):
    ax.plot([x - 3.2, x + 3.2], [y - 1.8, y - 1.8], color=INK, linewidth=1.0,
            zorder=5)
    shape_tag(ax, x, y - 1.2, r"$B \times L \times d$", size=6.2)


def _g_ledger(x, y):
    for i, dy in enumerate((1.4, 0.0, -1.4)):
        if i % 2 == 0:
            ax.add_patch(Rectangle((x - 3.2, y + dy - 0.65), 6.4, 1.3,
                                   linewidth=0, facecolor=tint(LG, 0.10),
                                   zorder=3))
        ax.plot([x - 2.7, x + 0.2], [y + dy, y + dy], color=MUTED,
                linewidth=0.7, zorder=5)
        ax.plot([x + 1.4, x + 2.7], [y + dy, y + dy], color=INK,
                linewidth=0.7, zorder=5)


_leg(0, "NUMBERED STAGE", "panel header + pictogram", _g_header)
_leg(1, "ELEMENT BOX", "one layer or operation", _g_box)
_leg(2, "TENSOR CUBOID", "depth ∝ channel count", _g_cuboid)
_leg(3, "SOLID ARROW", "forward data path", _g_arrow)
_leg(4, "BRANCH BUS", "one tensor feeding several ops", _g_bus)
_leg(5, "ELBOW ARC", "residual (skip) connection", _g_elbow)
_leg(6, "⊕ OPERATOR", "element-wise addition", _g_plus)
_leg(7, "SHAPE TAG", "tensor shape on a path", _g_tag)
_leg(8, "LEDGER ROW", "layer → parameter count", _g_ledger)

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------

if FIT_WARNINGS:
    print(f"\n{len(FIT_WARNINGS)} text-fit warning(s):")
    for w in FIT_WARNINGS:
        print("  !", w)
else:
    print("text-fit check: all boxes fit their contents")

_ledger_total = sum(int(c.replace(",", "")) for _, c in LEDGER)
assert _ledger_total == TOTAL_PARAMS, (
    f"ledger sums to {_ledger_total:,}, expected {TOTAL_PARAMS:,}")
print(f"ledger check: rows sum to {_ledger_total:,}")

for ext, kw in (("png", {"dpi": 220}), ("svg", {}), ("pdf", {})):
    fig.savefig(f"hangman_model_architecture.{ext}", facecolor="#FFFFFF", **kw)
    print(f"wrote hangman_model_architecture.{ext}")
plt.close(fig)
