#!/usr/bin/env python3
"""
make_architecture_figure.py
===========================

Regenerates the main architecture figure for the Neural Hangman Solver (v2).

Every number drawn here is traceable to `hangman_kaggle_notebook.ipynb`
(cell 5 defines `hangman.py`; cell 8 defines the run `Config`) or to a direct
measurement of `train.txt` / `test.txt`.

Usage (from anywhere -- output paths are resolved from this file's location):
    python figures/make_architecture_figure.py

Writes:
    <repo root>/hangman_architecture.png   (220 dpi, the one the README embeds)
    figures/hangman_architecture.svg
    figures/hangman_architecture.pdf

Dependencies: matplotlib + numpy only. Deterministic, no network, no assets.
"""

from __future__ import annotations

import os

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

def loop_right(ax, p_from, p_to, corridor_x, color, lw=0.9, r=1.0,
               dashed=True):
    """C-shaped feedback path: out of `p_from`, up the corridor, back
    into `p_to`. Used for the on-policy self-play loop."""
    (x0, y0), (x1, y1) = p_from, p_to
    verts = [(x0, y0),
             (corridor_x - r, y0), (corridor_x, y0), (corridor_x, y0 + r),
             (corridor_x, y1 - r), (corridor_x, y1), (corridor_x - r, y1),
             (x1, y1)]
    codes = [Path.MOVETO,
             Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO]
    ax.add_patch(FancyArrowPatch(
        path=Path(verts, codes), arrowstyle="-|>", mutation_scale=7.0,
        linewidth=lw, color=color, zorder=5, shrinkA=0, shrinkB=0,
        linestyle=(0, (2.6, 1.8)) if dashed else "solid"))


# --------------------------------------------------------------------------
# Measured constants (from train.txt / test.txt; see README)
# --------------------------------------------------------------------------

N_TRAIN_WORDS = 225_300
N_TEST_WORDS = 250_000
N_HOLDOUT = 4_506          # int(225300 * 0.02)
N_TRAIN_SPLIT = 220_794
TRAIN_LEN_HIST = [17, 263, 2184, 5235, 11177, 19369, 25722, 30185, 30633,
                  26715, 22575, 18027, 12845, 8626, 5165, 3120, 1757, 856,
                  436, 225, 97, 42, 14, 9, 3, 0, 0, 1, 2]   # lengths 1..29


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#FFFFFF")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
ax.set_facecolor("#FFFFFF")

TRUNK_Y = 74.0          # the forward path through the encoder runs at this height
LIFT_Y = 86.2           # residual elbows arc across at this height

# ---- title band ----------------------------------------------------------
txt(ax, W / 2, 130.6,
    "NEURAL HANGMAN SOLVER (v2) — CHARACTER ENCODER WITH THREE HEADS AND ON-POLICY SELF-PLAY",
    size=17.0, color=INK, weight="bold", ha="center")
txt(ax, W / 2, 127.0,
    "6,022,131 parameters   ·   trained only on train.txt (225,300 words)   ·   "
    "250,000 test games   ·   every value below is traced to hangman.py / Config",
    size=8.4, color=MUTED, ha="center")
ax.plot([3, 198], [125.0, 125.0], color=RULE, linewidth=0.9, zorder=1)

# ==========================================================================
# 1.  INPUT & VOCABULARY
# ==========================================================================
panel(ax, 3, 96, 66, 124.5, 1, "INPUT & VOCABULARY", HUE["input"],
      subtitle="(NO WORD LIST ENTERS THE DECISION PATH)", icon=icon_book)
h1 = HUE["input"]

box(ax, 5, 110.2, 26, 7.1, "train.txt  —  225,300 words",
    ["a–z only · length 1–29 · mean 9.35",
     "the sole source of supervision",
     "shares 0 words with test.txt"], h1, tag="train")
box(ax, 38, 110.2, 26, 7.1, "test.txt  —  250,000 words",
    ["a–z only · length 2–29 · mean 9.43",
     "read once, to write the submission",
     "never seen during training"], h1, dashed=True, tag="test")

txt(ax, 34.5, 108.9, "split_holdout(frac = 0.02, seed = 1337)", size=FS_TAG,
    color=h1, weight="bold", ha="center")
ax.plot([18, 18], [110.2, 107.5], color=INK, linewidth=0.9, zorder=5)
ax.plot([18, 51], [107.5, 107.5], color=INK, linewidth=0.9, zorder=5)
arrow(ax, (18, 107.5), (18, 106.4), lw=0.9)
arrow(ax, (51, 107.5), (51, 106.4), lw=0.9)

box(ax, 5, 100.6, 26, 5.8, "TRAIN SPLIT  —  220,794 (98%)",
    ["every board state the model ever trains on",
     "and every word replayed during self-play"], h1, tag="split")
box(ax, 38, 100.6, 26, 5.8, "HOLDOUT  —  4,506 (2%)",
    ["never trained on; the only honest local",
     "estimate of private-set win rate"], h1, dashed=True, tag="holdout")

box(ax, 5, 96.4, 59, 3.4, None,
    ["INTEGRITY:  train.txt ∩ test.txt = ∅  (verified: 0 shared words), so no dictionary "
     "lookup or candidate filter could generalise."],
    h1, fill=0.05, sub_size=FS_TINY, tag="integrity")

# ==========================================================================
# 2.  BOARD ENCODING
# ==========================================================================
panel(ax, 69, 96, 132, 124.5, 2, "BOARD ENCODING", HUE["encode"],
      subtitle="(THE SECRET WORD IS MASKED BEFORE EVERY FORWARD PASS)",
      icon=icon_grid)
h2 = HUE["encode"]

box(ax, 71, 113.3, 59, 4.5, "TOKEN VOCABULARY  —  V = 29",
    ["a–z → 0…25      BLANK '_' → 26      OTHER (non-letter) → 27      PAD → 28"],
    h2, tag="vocab")

rule_label(ax, 71, 130,
           "(A)  WORKED EXAMPLE — secret = “hangman”,  correctly guessed = { a, n }",
           111.7, h2)

_chars = ["_", "a", "n", "_", "_", "a", "n"]
_ids = ["26", "0", "13", "26", "26", "0", "13"]
_tgt = ["7", "·", "·", "6", "12", "·", "·"]
_cw, _cx0, _cy = 4.8, 82.0, 106.3
for i, ch in enumerate(_chars):
    cx = _cx0 + i * (_cw + 0.7)
    hidden = ch == "_"
    ax.add_patch(FancyBboxPatch(
        (cx, _cy), _cw, 3.6, boxstyle="round,pad=0,rounding_size=0.5",
        linewidth=1.0, edgecolor=h2,
        facecolor=tint(h2, 0.22) if hidden else "#FFFFFF", zorder=3))
    txt(ax, cx + _cw / 2, _cy + 1.8, ch, size=10.0, color=INK, weight="bold",
        ha="center")
    txt(ax, cx + _cw / 2, 104.8, _ids[i], size=FS_TINY, color=MUTED, ha="center")
    txt(ax, cx + _cw / 2, 103.1, _tgt[i], size=FS_TINY, color=MUTED, ha="center")

txt(ax, 80.6, _cy + 1.8, "board", size=FS_SUB, color=INK, weight="bold", ha="right")
txt(ax, 80.6, 104.8, "input ids", size=FS_TINY, color=MUTED, ha="right")
txt(ax, 80.6, 103.1, "pos_target", size=FS_TINY, color=MUTED, ha="right")
txt(ax, 121.5, 103.1, "· = −100, ignored", size=FS_TINY, color=MUTED, ha="left")

box(ax, 71, 96.3, 29, 5.8, "MODEL INPUTS",
    ["chars (B, L) · guessed (B, 26) · lengths (B,)",
     "right-padded to max L in batch, PAD = 28"], h2, sub_size=FS_TINY,
    tag="inputs")
box(ax, 101, 96.3, 29, 5.8, "TRAINING TARGETS",
    ["pos_target (B, L) · set_target (B, 26) = { g, h, m }",
     "value_target (B,) ∈ {0, 1},  or −1 = no label"], h2, sub_size=FS_TINY,
    tag="targets")

# ==========================================================================
# 3.  PHASE-1 STATE SAMPLING
# ==========================================================================
panel(ax, 135, 96, 198, 124.5, 3, "PHASE-1 STATE SAMPLING", HUE["sample"],
      subtitle="(8 SYNTHETIC BOARDS PER WORD → 1,766,352 STATES)",
      icon=icon_dice)
h3 = HUE["sample"]

box(ax, 137, 112.1, 59, 5.8, "REVEAL A RANDOM SUBSET OF THE LETTERS",
    ["k = # distinct letters;  n_revealed ~ Uniform{0 … k−1}  ⇒  at least one always stays hidden",
     "reveal_frac = n_revealed / (k − 1),  the within-word difficulty of this board"],
    h3, sub_size=FS_TINY, tag="reveal")
arrow(ax, (166.5, 112.1), (166.5, 110.7), lw=0.9)
box(ax, 137, 103.6, 59, 7.1, "ADD PLAUSIBLE WRONG GUESSES",
    ["cap = min(# letters absent from the word,  max_wrong − 1 = 5)",
     "n_wrong = min(cap,  max(0,  int(Gauss(reveal_frac · cap,  1.2))))",
     "drawn ∝ the Laplace-smoothed corpus letter prior, restricted to absent letters — "
     "the prior shapes misses only, it never scores a guess at inference"],
    h3, sub_size=FS_TINY, tag="wrong")
arrow(ax, (166.5, 103.6), (166.5, 102.2), lw=0.9)

box(ax, 137, 96.4, 59, 5.8, "CURRICULUM WEIGHTING  (curriculum = True)",
    ["word weight  w = 1 + epoch_progress · norm_len   —   uniform at epoch 1, "
     "up to 2× on the longest words by epoch 40"],
    h3, sub_size=FS_TINY, tag="curriculum")
_hmax = max(TRAIN_LEN_HIST)
_hx0, _hx1, _hy0, _hh = 139.0, 194.0, 97.3, 1.6
_bw = (_hx1 - _hx0) / len(TRAIN_LEN_HIST)
for i, c in enumerate(TRAIN_LEN_HIST):
    ax.add_patch(Rectangle((_hx0 + i * _bw, _hy0), _bw * 0.76,
                           _hh * c / _hmax, facecolor=tint(h3, 0.75),
                           edgecolor="none", zorder=4))
ax.plot([_hx0, _hx1], [_hy0, _hy0], color=tint(h3, 0.5), linewidth=0.7, zorder=4)
txt(ax, _hx0, 96.85, "len 1", size=5.6, color=MUTED, ha="left")
txt(ax, _hx1, 96.85, "29", size=5.6, color=MUTED, ha="right")
txt(ax, 166.5, 96.85, "train.txt word-length histogram", size=5.6,
    color=MUTED, ha="center")

# ==========================================================================
# 4.  HANGMANNET — SHARED CHARACTER ENCODER
# ==========================================================================
panel(ax, 3, 56, 137, 93.5, 4, "HANGMANNET — SHARED CHARACTER ENCODER",
      HUE["backbone"],
      subtitle="(6,022,131 PARAMETERS · d_model = 320 · dropout = 0.15)",
      icon=icon_stack)
h4 = HUE["backbone"]

# (a) four additive embedding streams
_emb = ["char_embedding    29 → 320",
        "position_embedding    48 → 320",
        "length_embedding    49 → 320",
        "guessed_projection    26 → 320"]
emb_boxes = []
for i, label in enumerate(_emb):
    by = 82.3 - (i + 1) * 3.4 - i * 1.0
    emb_boxes.append(box(ax, 5, by, 17, 3.4, label, (), h4, title_size=7.0,
                         tag=f"emb{i}"))
txt(ax, 13.5, 64.6, "all four are summed, then normalised", size=5.6,
    color=MUTED, ha="center")

o1 = op(ax, 24.4, TRUNK_Y, "⊕", hue=h4)
for b in emb_boxes:
    arrow(ax, (b["x1"], b["c"][1]), (o1["c"][0] - o1["r_"] - 0.15, TRUNK_Y),
          lw=0.8, mutation=6.0)

b_ln0 = box(ax, 26.6, 71.7, 8.6, 4.6, None, ["LayerNorm(320)", "× valid mask"],
            h4, sub_size=6.2, align="center", tag="ln0")
arrow(ax, (o1["r"][0] + 0.15, TRUNK_Y), (b_ln0["x"], TRUNK_Y), lw=0.9)

c1 = cuboid(ax, 37.0, 68.7, 4.6, 9.0, 3.6, h4, channels="320 ch")
arrow(ax, (b_ln0["x1"], TRUNK_Y), (36.9, TRUNK_Y), lw=0.9)
shape_tag(ax, 40.5, 81.4, r"$B \times L \times 320$")

# (b) multi-scale convolution front-end
CONV_X0, CONV_X1 = 45.5, 71.5
box(ax, CONV_X0, 64.0, CONV_X1 - CONV_X0, 19.4, None, (), h4, fill=0.04)
rule_label(ax, CONV_X0 + 1.5, CONV_X1 - 1.5,
           "(A)  MULTI-SCALE CONV FRONT-END", 81.8, h4)
arrow(ax, (c1["r"][0] + 0.1, TRUNK_Y), (47.2, TRUNK_Y), lw=0.9, mutation=0.1)
ax.plot([47.2, 47.2], [68.9, 77.7], color=INK, linewidth=0.9, zorder=5)
kb = []
for i, k in enumerate((3, 5, 7)):
    by = 76.0 - i * 4.4
    b = box(ax, 48.6, by, 13.2, 3.4, f"Conv1d  k = {k}  →  106 ch", (), h4,
            title_size=6.8, tag=f"conv{k}")
    kb.append(b)
    arrow(ax, (47.2, b["c"][1]), (b["x"], b["c"][1]), lw=0.8, mutation=6.0)
txt(ax, 48.6, 65.6, "padding = k // 2, so length L is preserved", size=5.6,
    color=MUTED)
b_cat = box(ax, 63.3, 68.5, 7.0, 11.0, None,
            ["concat", "318 ch", "Linear", "318→320", "GELU", "drop .15"],
            h4, sub_size=5.6, align="center", tag="concat")
for b in kb:
    arrow(ax, (b["x1"], b["c"][1]), (b_cat["x"], b["c"][1]), lw=0.8, mutation=6.0)

o2 = op(ax, 74.0, TRUNK_Y, "⊕", hue=h4)
txt(ax, 74.0, 70.7, "LayerNorm", size=5.8, color=MUTED, ha="center")
arrow(ax, (b_cat["x1"], TRUNK_Y), (o2["c"][0] - o2["r_"] - 0.15, TRUNK_Y), lw=0.9)
elbow(ax, (44.3, TRUNK_Y), (74.0, o2["t"][1] + 0.15), top=LIFT_Y,
      color=h4, lw=0.9, r=1.2)

# (c) BiLSTM, drawn once with a x2 multiplicity marker
LSTM_X0, LSTM_X1 = 77.0, 95.0
box(ax, LSTM_X0, 68.0, LSTM_X1 - LSTM_X0, 15.4, None, (), h4, fill=0.04)
rule_label(ax, LSTM_X0 + 1.5, LSTM_X1 - 1.5, "(B)  BiLSTM", 81.8, h4)
box(ax, 79.9, 71.9, 14.0, 7.0, None, (), h4, fill=0.05)
box(ax, 78.5, 70.5, 14.0, 7.0, "BiLSTM LAYER",
    ["hidden 160 per direction", "concat → 320, dropout .15"], h4,
    title_size=7.0, sub_size=5.6, tag="lstm")
txt(ax, 93.5, 69.2, "× 2", size=9.0, color=h4, weight="bold", ha="right")
arrow(ax, (o2["r"][0] + 0.15, TRUNK_Y), (78.5, TRUNK_Y), lw=0.9)

o3 = op(ax, 97.5, TRUNK_Y, "⊕", hue=h4)
txt(ax, 97.5, 70.7, "LayerNorm", size=5.8, color=MUTED, ha="center")
arrow(ax, (92.5, TRUNK_Y), (o3["c"][0] - o3["r_"] - 0.15, TRUNK_Y), lw=0.9)
elbow(ax, (75.7, TRUNK_Y), (97.5, o3["t"][1] + 0.15), top=LIFT_Y,
      color=h4, lw=0.9, r=1.2)

# (d) transformer encoder, drawn once with a x3 multiplicity marker
TR_X0, TR_X1 = 100.0, 122.0
box(ax, TR_X0, 64.0, TR_X1 - TR_X0, 19.4, None, (), h4, fill=0.04)
rule_label(ax, TR_X0 + 1.5, TR_X1 - 1.5, "(C)  TRANSFORMER ENCODER", 81.8, h4)
for off in (2.2, 1.1):
    box(ax, 102.2 + off, 70.2 + off, 14.0, 7.6, None, (), h4, fill=0.05)
box(ax, 102.2, 70.2, 14.0, 7.6, "ENCODER LAYER",
    ["pre-norm · 8 heads · d = 320", "FFN 1280 · GELU · PAD masked"], h4,
    title_size=7.0, sub_size=5.6, tag="encoder")
txt(ax, 120.0, 65.6, "× 3", size=9.0, color=h4, weight="bold", ha="right")
arrow(ax, (o3["r"][0] + 0.15, TRUNK_Y), (102.2, TRUNK_Y), lw=0.9)

c3 = cuboid(ax, 124.5, 68.7, 4.6, 9.0, 3.6, h4, channels="320 ch")
arrow(ax, (TR_X1, TRUNK_Y), (124.4, TRUNK_Y), lw=0.9)

# (e) parameter budget
rule_label(ax, 5, 93, "(D)  PARAMETER BUDGET", 62.6, h4)
_budget = [("embeddings + input norm", "49,600"),
           ("conv front-end", "611,838"),
           ("BiLSTM (2 layers)", "1,234,560"),
           ("transformer × 3", "3,698,880"),
           ("three heads", "427,253"),
           ("TOTAL", "6,022,131")]
for i, (name, val) in enumerate(_budget):
    bx = 5 + i * 14.8
    box(ax, bx, 56.6, 13.4, 4.0, None, (), h4,
        fill=0.20 if name == "TOTAL" else 0.08)
    txt(ax, bx + 6.7, 59.4, name, size=5.6, color=MUTED, ha="center")
    txt(ax, bx + 6.7, 57.6, val, size=7.6, color=INK, weight="bold", ha="center")

box(ax, 98.0, 56.6, 36.0, 5.8, "ENCODER LAYER  (× 3, PRE-NORM)",
    ["LayerNorm → 8-head self-attention (d = 320) → ⊕",
     "LayerNorm → 320 → 1280 → GELU → drop → 320 → ⊕"],
    h4, title_size=7.0, sub_size=5.8, tag="inset")

# ==========================================================================
# 5.  THREE PREDICTION HEADS
# ==========================================================================
panel(ax, 140, 56, 198, 93.5, 5, "THREE PREDICTION HEADS", HUE["heads"],
      subtitle="(ONE SHARED TRUNK, THREE OBJECTIVES)", icon=icon_fork)
h5 = HUE["heads"]

b_in5 = box(ax, 142, 72.3, 50, 3.4,
            "TRUNK OUTPUT  —  contextual features, one vector per slot", (), h5,
            align="center", title_size=7.6, tag="trunkout")
arrow(ax, (c3["r"][0] + 0.4, TRUNK_Y), (b_in5["x"], TRUNK_Y), lw=1.0)
shape_tag(ax, 134.0, 76.2, r"$B \times L \times 320$")

arrow(ax, (167, b_in5["y1"]), (167, 80.6), lw=0.9)
box(ax, 143, 80.6, 52, 5.8, "(A)  POSITIONAL HEAD  —  per-slot softmax",
    ["Linear(320 → 26) applied at every slot  →  (B, L, 26)",
     "already-guessed letters forced to −1e4 before the softmax"],
    h5, sub_size=FS_TINY, tag="poshead")

arrow(ax, (167, b_in5["y"]), (167, 70.8), lw=0.9)
box(ax, 143, 65.0, 52, 5.8, "(B)  MASKED POOLING  —  fixed-size state summary",
    ["concat[ mean-pool over valid slots ,  max-pool over valid slots ]",
     "→  (B, 640),  the shared input to both global heads"],
    h5, sub_size=FS_TINY, tag="pool")

ax.plot([167, 167], [65.0, 63.8], color=INK, linewidth=0.9, zorder=5)
ax.plot([155, 179], [63.8, 63.8], color=INK, linewidth=0.9, zorder=5)
arrow(ax, (155, 63.8), (155, 62.9), lw=0.9)
arrow(ax, (179, 63.8), (179, 62.9), lw=0.9)

box(ax, 143, 56.6, 24, 6.3, "(C)  SET HEAD",
    ["640 → 320 → GELU → drop → 26"], h5, sub_size=FS_TINY, tag="sethead")
op(ax, 145.8, 58.1, "σ", hue=h5, r=1.1, size=7.0)
txt(ax, 147.6, 58.1, "P(letter still hidden)", size=FS_TINY, color=MUTED)

box(ax, 171, 56.6, 24, 6.3, "(D)  VALUE HEAD",
    ["640 → 320 → GELU → drop → 1"], h5, sub_size=FS_TINY, tag="valhead")
op(ax, 173.8, 58.1, "σ", hue=h5, r=1.1, size=7.0)
txt(ax, 175.6, 58.1, "P(win | board state)", size=FS_TINY, color=MUTED)

# ---- band B -> band C connectors -----------------------------------------
arrow(ax, (75, 96.3), (75, 93.7), lw=1.0)
txt(ax, 76.6, 95.0, "one padded batch of 512 boards", size=FS_TINY, color=MUTED)
arrow(ax, (150, 56.6), (150, 53.7), lw=1.0)
txt(ax, 151.6, 54.9, "head outputs  →  decision rule", size=FS_TINY, color=MUTED)

# ==========================================================================
# 6.  LOSS & OPTIMISATION
# ==========================================================================
panel(ax, 3, 15.5, 47, 53.5, 6, "LOSS & OPTIMISATION", HUE["loss"],
      icon=icon_scale)
h6 = HUE["loss"]

box(ax, 5, 45.2, 40, 3.6, None, (), h6, fill=0.16)
ax.text(25, 47.0,
        r"$\mathcal{L}\;=\;\mathrm{CE}_{pos}\;+\;0.5\cdot\mathrm{BCE}_{set}"
        r"\;+\;0.5\cdot\mathrm{BCE}_{val}$",
        fontsize=10.0, color=INK, ha="center", va="center", zorder=6,
        family="serif")

box(ax, 5, 38.5, 40, 5.8, "POSITIONAL CROSS-ENTROPY  (weight 1.00)",
    ["supervised on hidden slots only (ignore_index = −100)",
     "logits of already-guessed letters masked to −1e4 first"],
    h6, sub_size=FS_TINY, tag="posce")
box(ax, 5, 31.8, 40, 5.8, "SET BCE  (set_loss_weight = 0.50)",
    ["binary_cross_entropy_with_logits over all 26 letters",
     "target = multi-hot of the letters still hidden"],
    h6, sub_size=FS_TINY, tag="setbce")
box(ax, 5, 25.1, 40, 5.8, "VALUE BCE  (value_loss_weight = 0.50)",
    ["applied only to rows whose label ≥ 0",
     "phase-1 rows carry the −1 sentinel ⇒ term is 0"],
    h6, dashed=True, sub_size=FS_TINY, tag="valbce")
box(ax, 5, 15.8, 40, 8.4, "OPTIMISER & SCHEDULE",
    ["AdamW · weight_decay 0.01 · grad-clip 1.0",
     "batch 512 · AMP float16 with GradScaler",
     "lr 3e-4 (phase 1)  /  1e-4 (phase 2)",
     "3% linear warmup → cosine to 0 · 3,449 steps/epoch"],
    h6, sub_size=FS_TINY, tag="optim")

# ==========================================================================
# 7.  TWO-PHASE TRAINING
# ==========================================================================
panel(ax, 50, 15.5, 102, 53.5, 7, "TWO-PHASE TRAINING", HUE["train"],
      subtitle="(PRETRAIN, THEN AN ON-POLICY SELF-PLAY LOOP)",
      icon=icon_calendar)
h7 = HUE["train"]

box(ax, 52, 41.4, 43, 5.8, "PHASE 1  —  RANDOM-STATE PRETRAINING",
    ["40 epochs over 1,766,352 synthetic states, lr 3e-4, curriculum on",
     "the value head receives no label in this phase"],
    h7, sub_size=FS_TINY, tag="phase1")
arrow(ax, (73.5, 41.4), (73.5, 40.1), lw=0.9)
rule_label(ax, 52, 95, "PHASE 2  —  ON-POLICY ROUND, REPEATED × 5", 39.7, h7)

steps = [
    (33.4, 4.5, "① SAMPLE",
     ["100,000 words drawn from the 220,794-word train split"]),
    (27.9, 4.5, "② SELF-PLAY",
     ["the current model plays each game out, logging every state visited"]),
    (22.4, 4.5, "③ BACK-FILL THE OUTCOME",
     ["the game's final win/loss ∈ {0, 1} labels every state from that game"]),
    (15.8, 5.8, "④ RE-TRAIN",
     ["harvested states mixed 50/50 with fresh random states, 2 epochs @ 1e-4",
      "then evaluate on the holdout — the only unbiased pre-submission signal"]),
]
step_boxes = []
for y, h, title, lines in steps:
    step_boxes.append(box(ax, 52, y, 43, h, title, lines, h7, title_size=7.4,
                          sub_size=5.6, tag=title[:9]))
for a, b in zip(step_boxes, step_boxes[1:]):
    arrow(ax, (73.5, a["y"]), (73.5, b["y1"]), lw=0.9)

loop_right(ax, (95, step_boxes[3]["c"][1]), (95, step_boxes[1]["c"][1]),
           corridor_x=98.6, color=h7, lw=0.9)
ax.text(99.9, 24.5, "improved policy  ⇒  new state distribution",
        fontsize=5.6, color=h7, ha="center", va="center", rotation=90,
        zorder=6, fontweight="bold")

# ==========================================================================
# 8.  DECISION RULE
# ==========================================================================
panel(ax, 105, 15.5, 157, 53.5, 8, "DECISION RULE", HUE["decide"],
      subtitle="(HOW THE NEXT LETTER IS CHOSEN, EVERY TURN)", icon=icon_target)
h8 = HUE["decide"]
DX, DW = 107, 48
DC = DX + DW / 2


def dmath(y, s, size=7.8):
    ax.text(DC, y, s, fontsize=size, color=INK, ha="center", va="center",
            zorder=6, family="serif")


d1 = box(ax, DX, 41.4, DW, 5.8, "①  MASK, THEN PER-SLOT SOFTMAX",
         ["guessed letters → −1e4, softmax over 26, then zeroed on revealed slots",
          "giving  p[i, ℓ] = P(slot i holds letter ℓ)"], h8, sub_size=FS_TINY,
         tag="d1")
d2 = box(ax, DX, 31.8, DW, 8.6, "②  PRESENCE, BLENDED WITH THE SET HEAD", (), h8)
dmath(36.2, r"$\mathrm{presence}_{\ell}=1-\prod_{i\,\in\,\mathrm{hidden}}"
            r"\left(1-p_{i\ell}\right)$")
dmath(33.4, r"$\mathrm{comb}_{\ell}=0.5\,\sigma(\mathrm{set}_{\ell})"
            r"+0.5\,\mathrm{presence}_{\ell}$")
d3 = box(ax, DX, 24.2, DW, 6.8, "③  EXPECTED-YIELD BONUS, ANNEALED BY LIVES",
         (), h8)
dmath(27.0, r"$\mathrm{score}_{\ell}=\mathrm{comb}_{\ell}"
            r"+b[\mathrm{lives}]\cdot(\sum_i p_{i\ell})\,/\,|\mathrm{hidden}|$")
txt(ax, DC, 24.9, "b = (0.10, 0.10, 0.06, 0.03, 0, 0, 0), indexed by lives left",
    size=5.6, color=MUTED, ha="center")
d4 = box(ax, DX, 15.8, DW, 7.4, "④  RISK ADJUSTMENT, THEN ARGMAX", (), h8)
dmath(19.4, r"$\mathrm{final}_{\ell}=0.65\,\mathrm{score}_{\ell}"
            r"+0.35\left[\,v\,\mathrm{comb}_{\ell}"
            r"+(1-v)\,\mathrm{comb}_{\ell}^{2}\,\right]$")
txt(ax, DC, 17.4, "v = σ(value head);  low v squares the scores, sharpening "
    "onto the safest letter", size=5.6, color=MUTED, ha="center")
txt(ax, DC, 16.4, "guessed letters set to −1e9,  then  guess = argmax over the "
    "26 letters", size=5.6, color=MUTED, ha="center")
for a, b in zip([d1, d2, d3], [d2, d3, d4]):
    arrow(ax, (DC, a["y"]), (DC, b["y1"]), lw=0.9)

# ==========================================================================
# 9.  INFERENCE & SUBMISSION
# ==========================================================================
panel(ax, 160, 15.5, 198, 53.5, 9, "INFERENCE & SUBMISSION", HUE["infer"],
      subtitle="(250,000 GAMES, BATCHED ON GPU)", icon=icon_play)
h9 = HUE["infer"]
IX, IW = 162, 34

i1 = box(ax, IX, 42.7, IW, 4.5, "LENGTH-BUCKETED BATCHES",
         ["sorted by length, ≤ 2,048 games/batch ⇒ no padding"], h9,
         sub_size=FS_TINY, tag="i1")
i2 = box(ax, IX, 34.7, IW, 7.1, "TURN LOOP  —  AT MOST 26 TURNS",
         ["one forward pass advances every active board",
          "hit → reveal all matching slots;  miss → lives − 1",
          "a game ends on a win or the 6th miss (max_wrong = 6)"], h9,
         sub_size=FS_TINY, tag="i2")
i3 = box(ax, IX, 29.3, IW, 4.5, "ENSEMBLE  (TRAIN_SECOND_SEED = False)",
         ["average scores across seeds; off by default"], h9, dashed=True,
         sub_size=FS_TINY, tag="i3")
i4 = box(ax, IX, 22.6, IW, 5.8, "submission.csv  —  250,000 rows",
         ["word_id: 0 … 249,999, in order",
          "guessed_letters_string: the guesses played"], h9, sub_size=FS_TINY,
         tag="i4")
i5 = box(ax, IX, 15.9, IW, 5.8, "verify_submission()",
         ["column names and order, row count, a–z only,",
          "no letter guessed twice — asserted before upload"], h9,
         sub_size=FS_TINY, tag="i5")
for a, b in zip([i1, i2, i3, i4], [i2, i3, i4, i5]):
    arrow(ax, (IX + IW / 2, a["y"]), (IX + IW / 2, b["y1"]), lw=0.9)

loop_right(ax, (d4["x1"], 19.5), (IX, 38.0), corridor_x=158.5, color=INK,
           lw=0.9, dashed=False)
ax.text(159.7, 29.0, "one letter per turn", fontsize=5.6, color=MUTED,
        ha="center", va="center", rotation=90, zorder=6)

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

LEG_X0, LEG_W = 20.0, 17.6
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
    icon_stack(ax, x, y + 1.3, 1.2)


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


def _g_dashed(x, y):
    box(ax, x - 3.0, y - 2.0, 6.0, 4.0, None, (), LG, fill=0.06, dashed=True)


def _g_arrow(x, y):
    arrow(ax, (x - 3.2, y), (x + 3.2, y), lw=1.0)


def _g_dasharrow(x, y):
    arrow(ax, (x - 3.2, y), (x + 3.2, y), lw=1.0, dashed=True,
          color=HUE["train"])


def _g_elbow(x, y):
    elbow(ax, (x - 3.0, y - 1.6), (x + 3.0, y - 1.6), lift=3.2, color=LG,
          lw=1.0, r=0.8)


def _g_plus(x, y):
    op(ax, x, y, "⊕", hue=LG, r=1.5, size=8.0)


def _g_sigma(x, y):
    op(ax, x, y, "σ", hue=LG, r=1.5, size=8.0)


def _g_tag(x, y):
    ax.plot([x - 3.2, x + 3.2], [y - 1.8, y - 1.8], color=INK, linewidth=1.0,
            zorder=5)
    shape_tag(ax, x, y - 1.2, r"$B \times L \times d$", size=6.2)


_leg(0, "NUMBERED STAGE", "panel header + pictogram", _g_header)
_leg(1, "ELEMENT BOX", "one layer or operation", _g_box)
_leg(2, "TENSOR CUBOID", "depth ∝ channel count", _g_cuboid)
_leg(3, "DASHED BOX", "optional / off by default", _g_dashed)
_leg(4, "SOLID ARROW", "forward data path", _g_arrow)
_leg(5, "DASHED ARROW", "self-play feedback loop", _g_dasharrow)
_leg(6, "ELBOW ARC", "residual (skip) connection", _g_elbow)
_leg(7, "⊕ OPERATOR", "element-wise addition", _g_plus)
_leg(8, "σ OPERATOR", "sigmoid squashing to (0, 1)", _g_sigma)
_leg(9, "SHAPE TAG", "tensor shape on a path", _g_tag)

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------

if FIT_WARNINGS:
    print(f"\n{len(FIT_WARNINGS)} text-fit warning(s):")
    for w in FIT_WARNINGS:
        print("  !", w)
else:
    print("text-fit check: all boxes fit their contents")

# The PNG lives at the repo root because the README embeds it; the vector
# exports stay next to this script. Paths are resolved from __file__ so the
# script behaves the same whichever directory it is invoked from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_OUTPUTS = (
    ("png", os.path.join(_ROOT, "hangman_architecture.png"), {"dpi": 220}),
    ("svg", os.path.join(_HERE, "hangman_architecture.svg"), {}),
    ("pdf", os.path.join(_HERE, "hangman_architecture.pdf"), {}),
)
for _ext, _path, _kw in _OUTPUTS:
    fig.savefig(_path, facecolor="#FFFFFF", **_kw)
    print(f"wrote {os.path.relpath(_path, _ROOT)}")
plt.close(fig)
