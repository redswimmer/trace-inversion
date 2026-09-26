#!/usr/bin/env python3
"""Render the README's figures as light/dark SVG pairs.

Reads the committed records — bench/results/phase6/summary.json (every student evaluation) — plus
the cited rows that were never re-run in Phase 6 (Phase 0's no-think baseline, the paper's
Table 3). No dependencies.

    python3 docs/assets/make_charts.py      # writes docs/assets/*-{light,dark}.svg

Figures:
    headline     the paper's core claim (forged beats plain distillation), paper vs here
    pipeline     how the attack is wired, and where the oracle comes from
    results      every condition's student accuracy on both benchmarks
    termination  why: accuracy vs the share of answers that never finished
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets"
S = json.loads((ROOT / "bench" / "results" / "phase6" / "summary.json").read_text())

BENCHES = ("MATH500", "JEEBench")
NOTHINK = {"MATH500": 79.0, "JEEBench": 47.8}          # Phase 0, template-default (no-think) render
THINK = {b: S["baseline-think"]["bench"][b]["acc"] for b in BENCHES}

# Paper Table 3, Qwen2.5-7B student, victim R1, R1-Weak (= R1-Distill-Qwen-1.5B) surrogate (docs/02)
PAPER = {"MATH500": {"surr": 63.2, "synth": 71.8}, "JEEBench": {"surr": 19.7, "synth": 36.3}}

SEEDS = ["synth-7b-sum", "synth-7b-sum-seed1235", "synth-7b-sum-seed1236"]

FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'
# Categorical slots 1–3 of the dataviz reference palette (validated all-pairs in both modes) carry
# the same meaning in every figure: blue = forged traces (the attack), orange = the surrogate's own
# traces, aqua = the victim's real traces. Blue/red is the diverging pair. Neutral gray = no trace.
THEME = {
    "light": dict(ink="#0b0b0b", ink2="#52514e", muted="#6f6e69", grid="#e7e6e1", axis="#c3c2b7",
                  page="#ffffff", panel="#f6f5f2", box="#ffffff", forged="#2a78d6",
                  distilled="#eb6834", oracle="#1baf7a", ref="#a8a69f", up="#2a78d6",
                  down="#e34948", band="#ecebe6"),
    "dark": dict(ink="#f0f0ee", ink2="#c3c2b7", muted="#9a9992", grid="#2a2a28", axis="#45443f",
                 page="#0d1117", panel="#161b22", box="#0d1117", forged="#3987e5",
                 distilled="#d95926", oracle="#199e70", ref="#6e6d68", up="#3987e5",
                 down="#e66767", band="#23282f"),
}


def acc(run, bench):
    return S[run]["bench"][bench]["acc"]


def trunc(run, bench):
    b = S[run]["bench"][bench]
    return 100.0 * b["truncated"] / b["n"]


def seed_range(bench):
    vals = [acc(r, bench) for r in SEEDS]
    return max(vals) - min(vals)


# --------------------------------------------------------------------------- SVG primitives
def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, t, size=12, fill=None, anchor="start", weight="normal", tabular=False,
         halo=False, italic=False):
    style = "font-variant-numeric: tabular-nums;" if tabular else ""
    if italic:
        style += "font-style: italic;"
    # halo = page-coloured stroke behind the glyphs so a label crossing a line stays legible
    halo_attr = (f' paint-order="stroke" stroke="{t["page"]}" stroke-width="3" '
                 f'stroke-linejoin="round"') if halo else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family=\'{FONT}\' font-size="{size}" '
            f'fill="{fill or t["ink2"]}" text-anchor="{anchor}" font-weight="{weight}" '
            f'style="{style}"{halo_attr}>{esc(s)}</text>')


def line(x1, y1, x2, y2, stroke, w=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
            f'stroke-width="{w}"{d}/>')


def rect(x, y, w, h, fill, r=0, stroke=None, sw=1.0, dash=None):
    s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" '
            f'fill="{fill}"{s}{d}/>')


def hbar(x0, y, w, h, fill, r=4):
    """Horizontal bar growing right from x0; rounded on the data end only."""
    if w <= 0:
        return ""
    r = min(r, w / 2, h / 2)
    x1 = x0 + w
    return (f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} a{r},{r} 0 0 1 {r},{r} V{y + h - r:.1f} '
            f'a{r},{r} 0 0 1 -{r},{r} H{x0:.1f} Z" fill="{fill}"/>')


def hbar_left(x0, y, w, h, fill, r=4):
    """Bar growing LEFT from x0 (negative values); rounded on the data end only."""
    if w <= 0:
        return ""
    r = min(r, w / 2, h / 2)
    x1 = x0 - w
    return (f'<path d="M{x0:.1f},{y:.1f} H{x1 + r:.1f} a{r},{r} 0 0 0 -{r},{r} V{y + h - r:.1f} '
            f'a{r},{r} 0 0 0 {r},{r} H{x0:.1f} Z" fill="{fill}"/>')


def arrow_defs(t):
    marks = []
    for name, col in (("ink", t["ink2"]), ("forged", t["forged"]), ("oracle", t["oracle"]),
                      ("distilled", t["distilled"])):
        marks.append(f'<marker id="ah-{name}" viewBox="0 0 10 10" refX="9" refY="5" '
                     f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                     f'<path d="M0,1 L9,5 L0,9 Z" fill="{col}"/></marker>')
    return "<defs>" + "".join(marks) + "</defs>"


def svg(w, h, body, t, title):
    # explicit page background so the figure reads the same on any README surface
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">\n'
            f'<title>{esc(title)}</title>\n'
            f'{rect(0, 0, w, h, t["page"], r=10)}\n{body}\n</svg>\n')


def swatch_legend(x, y, items, t, size=12):
    """Row of (colour, label) swatches; widths estimated from label length."""
    out = []
    for col, lab in items:
        out.append(rect(x, y - 9, 12, 12, col, r=3))
        out.append(text(x + 18, y + 1, lab, t, size, t["ink2"]))
        x += 18 + 0.58 * size * len(lab) + 22
    return out


# --------------------------------------------------------------------------- 1. headline
HEAD_ROWS = [  # label, sublabel, {bench: delta}
    ("Paper", "R1-Distill-1.5B surrogate",
     {b: PAPER[b]["synth"] - PAPER[b]["surr"] for b in BENCHES}),
    ("This reproduction", "same 1.5B surrogate",
     {b: acc("synth-1.5b-sum", b) - acc("surr-1.5b", b) for b in BENCHES}),
    ("This reproduction", "7B surrogate",
     {b: acc("synth-7b-sum", b) - acc("surr-7b", b) for b in BENCHES}),
]


def chart_headline(mode):
    t = THEME[mode]
    W, LABEL_W, PANEL_W, GAP, TOP, ROW, BAR = 900, 220, 300, 44, 104, 46, 22
    LO, HI = -15.0, 20.0
    px = PANEL_W / (HI - LO)
    H = TOP + len(HEAD_ROWS) * ROW + 70
    out = [text(24, 36, "Does forging the hidden reasoning beat just copying the surrogate?", t,
                18, t["ink"], weight="700"),
           text(24, 58, "Student accuracy when trained on forged victim traces, minus accuracy "
                "when trained on the surrogate's own traces (percentage points)", t, 12.5,
                t["muted"])]
    for i, bench in enumerate(BENCHES):
        x0 = LABEL_W + i * (PANEL_W + GAP)
        zero = x0 + (0 - LO) * px
        band = seed_range(bench)
        out.append(text(x0, TOP - 16, bench, t, 13, t["ink"], weight="700"))
        # noise band: ± the range across 3 evaluation seeds of one cell
        out.append(rect(zero - band * px, TOP - 4, 2 * band * px, len(HEAD_ROWS) * ROW + 4,
                        t["band"]))
        for v in range(-10, 21, 10):
            gx = x0 + (v - LO) * px
            out.append(line(gx, TOP - 4, gx, TOP + len(HEAD_ROWS) * ROW,
                            t["axis"] if v == 0 else t["grid"], 1.2 if v == 0 else 1))
            out.append(text(gx, TOP + len(HEAD_ROWS) * ROW + 16, f"{v:+d}" if v else "0", t, 11,
                            t["muted"], anchor="middle", tabular=True))
        for r, (_, _, deltas) in enumerate(HEAD_ROWS):
            d = deltas[bench]
            ry = TOP + r * ROW + (ROW - BAR) / 2
            if d >= 0:
                out.append(hbar(zero, ry, d * px, BAR, t["up"]))
                out.append(text(zero + d * px + 7, ry + BAR / 2 + 5, f"+{d:.1f}", t, 13,
                                t["ink"], weight="600", tabular=True, halo=True))
            else:
                out.append(hbar_left(zero, ry, -d * px, BAR, t["down"]))
                out.append(text(zero + d * px - 7, ry + BAR / 2 + 5, f"−{-d:.1f}", t, 13,
                                t["ink"], anchor="end", weight="600", tabular=True, halo=True))
        yb = TOP + len(HEAD_ROWS) * ROW + 34
        out.append(text(zero - 6, yb, "← copying wins", t, 11, t["down"], anchor="end",
                        weight="600"))
        out.append(text(zero + 6, yb, "forging wins →", t, 11, t["up"], weight="600"))
    for r, (lab, sub, _) in enumerate(HEAD_ROWS):
        cy = TOP + r * ROW + ROW / 2
        out.append(text(24, cy - 2, lab, t, 13, t["ink"], weight="600"))
        out.append(text(24, cy + 14, sub, t, 11.5, t["muted"]))
    out.append(line(24, TOP + ROW, W - 24, TOP + ROW, t["grid"], 1))  # paper | ours
    out.append(rect(24, H - 22, 12, 10, t["band"], r=2))
    out.append(text(42, H - 13, f"shaded: within evaluation-seed noise (range across 3 seeds: "
                    f"MATH500 ±{seed_range('MATH500'):.1f}, JEEBench ±{seed_range('JEEBench'):.1f})"
                    " · the paper reports single runs", t, 11, t["muted"]))
    return svg(W, H, "\n".join(out), t,
               "Forged traces minus plain distillation: paper positive, this reproduction negative")


# --------------------------------------------------------------------------- 2. pipeline
def chart_pipeline(mode):
    t = THEME[mode]
    W = 900
    out = [arrow_defs(t)]
    BH = 98

    def box(x, y, w, title, lines, accent=None, emphasis=False):
        o = [rect(x, y, w, BH, t["box"], r=10, stroke=accent or t["axis"],
                  sw=2 if emphasis else 1.2)]
        if accent:
            o.append(f'<path d="M{x + 1:.1f},{y + 10:.1f} a9,9 0 0 1 9,-9 H{x + 12:.1f} '
                     f'V{y + BH - 1:.1f} H{x + 10:.1f} a9,9 0 0 1 -9,-9 Z" fill="{accent}"/>')
        tx = x + (22 if accent else 14)
        o.append(text(tx, y + 27, title, t, 15, t["ink"], weight="700"))
        for i, s in enumerate(lines):
            o.append(text(tx, y + 50 + i * 18, s, t, 12.5, t["ink2"]))
        return o

    def arrow(x1, y, x2, kind="ink", w=1.8):
        col = t[kind] if kind != "ink" else t["ink2"]
        return (f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="{col}" '
                f'stroke-width="{w}" marker-end="url(#ah-{kind})"/>')

    # four columns shared by both lanes
    XS, WS = (28, 196, 426, 662), (128, 190, 188, 216)
    L1, LH = 16, 154
    L2 = L1 + LH + 46
    for y, n, lab in ((L1, "1", "Learn to forge, on a surrogate whose reasoning is visible"),
                      (L2, "2", "Attack: the victim shows only its answer and a summary")):
        out.append(rect(16, y, W - 32, LH, t["panel"], r=12))
        out.append(f'<circle cx="40" cy="{y + 22}" r="11" fill="{t["ink"]}"/>')
        out.append(text(40, y + 26.5, n, t, 13, t["page"], anchor="middle", weight="700"))
        out.append(text(60, y + 27, lab, t, 14, t["ink"], weight="600"))
    BY1, BY2 = L1 + 44, L2 + 44
    lanes = (
        (BY1, [("Problems A", ["OpenThoughts", "5,000 prompts"], None, False),
               ("Surrogate", ["R1-Distill 7B · 1.5B", "reasons in the open"], t["distilled"],
                False),
               ("Compressor", ["Qwen3.5-4B, zero-shot", "summarizes each trace"], None, False),
               ("Inverter: train", ["Qwen3.5-4B + LoRA (×4)", "problem, answer, summary",
                                    "→ reasoning trace"], t["forged"], True)]),
        (BY2, [("Problems B", ["OpenThoughts", "5,000 prompts"], None, False),
               ("Victim", ["Qwen3.8-27B, 4-bit", "shows answer + summary", "hides its trace"],
                None, False),
               ("Inverter: apply", ["same 4 adapters", "forges the hidden trace"], t["forged"],
                False),
               ("Student", ["Qwen3.5-2B, full fine-tune", "one per training condition",
                            "scored: MATH500, JEEBench"], None, True)]),
    )
    for by, boxes in lanes:
        for (title, lines, accent, emph), x, w in zip(boxes, XS, WS):
            out += box(x, by, w, title, lines, accent, emph)
        for i in range(3):
            last = by == BY2 and i == 2
            out.append(arrow(XS[i] + WS[i], by + BH / 2, XS[i + 1] - 2,
                             "forged" if last else "ink", 2.6 if last else 1.8))
    fx = (XS[2] + WS[2] + XS[3]) / 2
    out.append(text(fx, BY2 + BH / 2 - 9, "forged", t, 12, t["forged"], anchor="middle",
                    weight="700"))
    out.append(text(fx, BY2 + BH / 2 + 20, "traces", t, 12, t["forged"], anchor="middle",
                    weight="700"))

    # the trained adapters move from lane 1 to lane 2
    ax, ix, my = XS[3] + WS[3] / 2, XS[2] + WS[2] - 44, (L1 + LH + L2) / 2
    out.append(f'<path d="M{ax},{BY1 + BH} V{my} H{ix} V{BY2 - 2}" fill="none" '
               f'stroke="{t["forged"]}" stroke-width="1.8" marker-end="url(#ah-forged)"/>')
    out.append(text((ax + ix) / 2, my - 7, "trained adapters", t, 12, t["forged"],
                    anchor="middle", weight="700", halo=True))

    # oracle path: the victim's real trace, withheld from the attack
    vx, sx_ = XS[1] + WS[1] / 2, XS[3] + WS[3] / 2
    oy = L2 + LH + 24
    out.append(f'<path d="M{vx},{BY2 + BH} V{oy} H{sx_} V{BY2 + BH + 2}" fill="none" '
               f'stroke="{t["oracle"]}" stroke-width="1.8" stroke-dasharray="6 4" '
               f'marker-end="url(#ah-oracle)"/>')
    out.append(text((vx + sx_) / 2, oy + 20, "the victim's real traces: never shown to the "
                    "attack, used only to train the oracle student", t, 12.5, t["oracle"],
                    anchor="middle", weight="700"))
    H = oy + 62
    out.append(text(W / 2, H - 12, "Baseline students on the same problems: victim's answers "
                    "only · summaries + answers · the surrogate's own traces", t, 12, t["muted"],
                    anchor="middle"))
    return svg(W, H, "\n".join(out), t, "Pipeline: surrogate, compressor, inverter, victim, student")


# --------------------------------------------------------------------------- 3. results
RES_ROWS = [  # label, run, colour role
    ("Victim's answers only", "answer-only", "ref"),
    ("Victim's summaries + answers", "summary-answer", "ref"),
    ("Forged traces · 1.5B surrogate", "synth-1.5b-sum", "forged"),
    ("Forged traces · 7B surrogate", "synth-7b-sum", "forged"),
    ("Surrogate's own traces · 1.5B", "surr-1.5b", "distilled"),
    ("Surrogate's own traces · 7B", "surr-7b", "distilled"),
    ("Victim's real traces (oracle)", "oracle", "oracle"),
]


def chart_results(mode):
    t = THEME[mode]
    W, LABEL_W, PANEL_W, GAP, TOP, ROW, BAR = 900, 232, 290, 44, 132, 32, 18
    XMAX = 80.0
    px = PANEL_W / XMAX
    PLOT_H = len(RES_ROWS) * ROW
    H = TOP + PLOT_H + 92
    out = [text(24, 36, "What the student learned from each kind of training data", t, 18,
                t["ink"], weight="700"),
           text(24, 58, "Qwen3.5-2B accuracy (%) after fine-tuning on 3,616 problems · 1,015 "
                "test problems · dashed lines: the same model untrained", t, 12.5, t["muted"])]
    out += swatch_legend(24, 86, [(t["ref"], "no trace"), (t["forged"], "forged traces (the "
                                  "attack)"), (t["distilled"], "surrogate's traces (plain "
                                  "distillation)"), (t["oracle"], "victim's real traces")], t)
    for i, bench in enumerate(BENCHES):
        x0 = LABEL_W + i * (PANEL_W + GAP)
        out.append(text(x0, TOP - 12, bench, t, 13, t["ink"], weight="700"))
        for v in range(0, int(XMAX) + 1, 20):
            gx = x0 + v * px
            out.append(line(gx, TOP, gx, TOP + PLOT_H, t["axis"] if v == 0 else t["grid"]))
            if min(abs(v - THINK[bench]), abs(v - NOTHINK[bench])) > 3:  # keep clear of refs
                out.append(text(gx, TOP + PLOT_H + 15, str(v), t, 10.5, t["muted"],
                                anchor="middle", tabular=True))
        # untrained reference lines, labelled beneath the axis so they never meet a value label
        for k, (val, lab, col) in enumerate((
                (THINK[bench], f"thinking on · {THINK[bench]:.1f}", t["muted"]),
                (NOTHINK[bench], f"thinking off · {NOTHINK[bench]:.1f}", t["ink"]))):
            gx = x0 + val * px
            out.append(line(gx, TOP - 2, gx, TOP + PLOT_H + 24 + k * 15, col, 1.5, "4 3"))
            out.append(text(gx - 4, TOP + PLOT_H + 36 + k * 15, lab, t, 10.5, col, anchor="end",
                            weight="600", halo=True))
        for r, (_, run, role) in enumerate(RES_ROWS):  # bars drawn over the reference lines
            v = acc(run, bench)
            out.append(hbar(x0, TOP + r * ROW + (ROW - BAR) / 2, v * px, BAR, t[role]))
        for r, (_, run, role) in enumerate(RES_ROWS):
            v = acc(run, bench)
            ry = TOP + r * ROW + ROW / 2
            out.append(text(x0 + v * px + 6, ry + 4.5, f"{v:.1f}", t, 12, t["ink"],
                            weight="600" if role in ("forged", "oracle") or run == "surr-7b"
                            else "normal", tabular=True, halo=True))
    for r, (label, run, role) in enumerate(RES_ROWS):
        out.append(text(24, TOP + r * ROW + ROW / 2 + 4.5, label, t, 12.5, t["ink"]))
    out.append(text(24, H - 12, f"Evaluation-seed noise on one cell: ±{seed_range('MATH500'):.1f} "
                    f"MATH500, ±{seed_range('JEEBench'):.1f} JEEBench. Forged rows use the "
                    "with-summary inverter; all 13 runs are tabulated below.", t, 11, t["muted"]))
    return svg(W, H, "\n".join(out), t, "Student accuracy by training condition")


# --------------------------------------------------------------------------- 4. termination
POINTS = [  # run, label, role, label dx, dy, anchor
    ("baseline-think", "untrained, thinking on", "ref", 0, -16, "middle"),
    ("oracle", "victim's real traces", "oracle", 0, -15, "middle"),
    ("surr-7b", "surrogate's traces · 7B", "distilled", 12, 4, "start"),
    ("surr-1.5b", "surrogate's traces · 1.5B", "distilled", 12, 16, "start"),
    ("synth-7b-sum", "forged · 7B surrogate", "forged", 12, -8, "start"),
    ("synth-1.5b-sum", "forged · 1.5B surrogate", "forged", 12, 4, "start"),
]
FLOORS = ("answer-only", "summary-answer")
VARIANTS = ("synth-7b-nosum", "synth-1.5b-nosum")    # plotted small, unlabelled
TRACE_RUNS = [p[0] for p in POINTS if p[0] != "baseline-think"] + list(VARIANTS)


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    b = sxy / sxx
    return my - b * mx, b, sxy / math.sqrt(sxx * syy)


def chart_termination(mode):
    t = THEME[mode]
    W, H, L, R, TOP, BOT = 900, 470, 72, 40, 112, 56
    XLO, XHI, YLO, YHI = 0.0, 70.0, 15.0, 50.0
    pw, ph = W - L - R, H - TOP - BOT
    sx = lambda v: L + (v - XLO) / (XHI - XLO) * pw
    sy = lambda v: TOP + (YHI - v) / (YHI - YLO) * ph
    xs = [trunc(r, "JEEBench") for r in TRACE_RUNS]
    ys = [acc(r, "JEEBench") for r in TRACE_RUNS]
    a, b, r = fit(xs, ys)
    out = [text(24, 36, "Students that learned to stop scored; students that looped did not", t,
                18, t["ink"], weight="700"),
           text(24, 58, f"JEEBench accuracy vs share of answers cut off at the 32k-token limit · "
                f"r = {r:.2f}".replace("-", "−") + " across the 7 trace-trained students", t, 12.5, t["muted"])]
    out += swatch_legend(24, 86, [(t["forged"], "forged traces"), (t["distilled"],
                                  "surrogate's traces"), (t["oracle"], "victim's real traces"),
                                  (t["ref"], "untrained / no trace")], t)
    for v in range(0, 71, 10):
        out.append(line(sx(v), TOP, sx(v), TOP + ph, t["axis"] if v == 0 else t["grid"]))
        out.append(text(sx(v), TOP + ph + 17, f"{v}%", t, 10.5, t["muted"], anchor="middle",
                        tabular=True))
    for v in range(15, 51, 5):
        out.append(line(L, sy(v), L + pw, sy(v), t["grid"]))
        out.append(text(L - 8, sy(v) + 4, str(v), t, 10.5, t["muted"], anchor="end",
                        tabular=True))
    out.append(text(L + pw / 2 - 60, H - 12, "JEEBench answers still unfinished at the 32k-token "
                    "limit →", t, 11.5, t["ink2"], anchor="middle"))
    out.append(f'<text x="20" y="{TOP + ph / 2:.1f}" font-family=\'{FONT}\' font-size="11.5" '
               f'fill="{t["ink2"]}" text-anchor="middle" '
               f'transform="rotate(-90 20 {TOP + ph / 2:.1f})">JEEBench accuracy % →</text>')
    # least-squares line through the trace-trained students
    x1, x2 = 4.0, 54.0
    out.append(line(sx(x1), sy(a + b * x1), sx(x2), sy(a + b * x2), t["axis"], 2, "6 4"))
    # the untrained model's cure: an arrow from it towards the trained cluster
    bx, by = sx(trunc("baseline-think", "JEEBench")), sy(acc("baseline-think", "JEEBench"))
    out.append(arrow_defs(t))
    out.append(f'<line x1="{bx - 14:.1f}" y1="{by - 6:.1f}" x2="{sx(26):.1f}" y2="{sy(42):.1f}" '
               f'stroke="{t["ink2"]}" stroke-width="1.4" stroke-dasharray="2 4" '
               f'marker-end="url(#ah-ink)"/>')
    out.append(text(sx(44), sy(43.2), "fine-tuning mostly teaches", t, 11.5, t["ink2"],
                    anchor="middle", italic=True, halo=True))
    out.append(text(sx(44), sy(43.2) + 15, "the model when to stop", t, 11.5, t["ink2"],
                    anchor="middle", italic=True, halo=True))
    # no-trace floors: they stop, but never learned to reason
    fx = [sx(trunc(f, "JEEBench")) for f in FLOORS]
    fy = [sy(acc(f, "JEEBench")) for f in FLOORS]
    for x, y in zip(fx, fy):
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{t["ref"]}" '
                   f'stroke="{t["page"]}" stroke-width="2"/>')
    out.append(text(max(fx) + 12, sum(fy) / 2 + 4, "answers / summaries only — stop at once, "
                    "never learned to reason", t, 11.5, t["ink2"], halo=True))
    for run in VARIANTS:
        out.append(f'<circle cx="{sx(trunc(run, "JEEBench")):.1f}" '
                   f'cy="{sy(acc(run, "JEEBench")):.1f}" r="5" fill="{t["forged"]}" '
                   f'stroke="{t["page"]}" stroke-width="1.5" opacity="0.75"/>')
    out.append(text(W - R, H - 12, "small dots: forged, no-summary variant", t, 10.5,
                    t["muted"], anchor="end"))
    for run, label, role, dx, dy, anchor in POINTS:
        x, y = sx(trunc(run, "JEEBench")), sy(acc(run, "JEEBench"))
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{t[role]}" '
                   f'stroke="{t["page"]}" stroke-width="2"/>')
        out.append(text(x + dx, y + dy, label, t, 12, t["ink"], anchor=anchor, weight="600",
                        halo=True))
    return svg(W, H, "\n".join(out), t, "JEEBench accuracy vs truncation rate by student")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("headline", chart_headline), ("pipeline", chart_pipeline),
                     ("results", chart_results), ("termination", chart_termination)):
        for mode in ("light", "dark"):
            p = OUT / f"{name}-{mode}.svg"
            p.write_text(fn(mode))
            print("wrote", p.relative_to(ROOT))
    # self-check: the numbers the README quotes are the ones in the record
    assert acc("oracle", "JEEBench") == 45.6 and acc("surr-7b", "JEEBench") == 45.4
    assert abs(trunc("baseline-think", "JEEBench") - 65.6) < 0.1
    assert [round(d["JEEBench"], 1) for _, _, d in HEAD_ROWS] == [16.6, -7.5, -9.9]
    assert [round(d["MATH500"], 1) for _, _, d in HEAD_ROWS] == [8.6, -1.2, -7.6]
    assert round(seed_range("MATH500"), 1) == 3.4 and round(seed_range("JEEBench"), 1) == 5.3
    print("self-check passed")
