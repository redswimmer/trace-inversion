#!/usr/bin/env python3
"""Render the README's figures as light/dark SVG pairs.

Reads the committed records — bench/results/phase6/summary.json (every student evaluation) — plus
the cited rows that were never re-run in Phase 6 (Phase 0's no-think baseline, the paper's
Table 3). No dependencies.

    python3 docs/assets/make_charts.py      # writes docs/assets/*-{light,dark}.svg

Figures:
    headline     the paper's core claim (forged beats plain distillation), paper vs here
    pipeline     how the attack is wired, and where the victim's real traces go
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
PAPER = {"MATH500": {"base": 71.2, "surr": 63.2, "synth": 71.8},
         "JEEBench": {"base": 28.3, "surr": 19.7, "synth": 36.3}}

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
# For each setting: the untrained student, then plain distillation, then forged traces, so each
# method's effect reads directly as a change from the untrained student.
HEAD_GROUPS = [  # label, sublabel, {bench: (untrained, plain distillation, forged)}
    ("Paper", "R1-Distill-1.5B surrogate",
     {b: (PAPER[b]["base"], PAPER[b]["surr"], PAPER[b]["synth"]) for b in BENCHES}),
    ("Our recreation", "7B surrogate",
     {b: (THINK[b], acc("surr-7b", b), acc("synth-7b-sum", b)) for b in BENCHES}),
]


def chart_headline(mode):
    t = THEME[mode]
    W, LABEL_W, PANEL_W, GAP, TOP = 900, 210, 290, 60, 132
    BAR, STEP, GROUP_GAP = 15, 20, 26
    XMAX = 80.0
    px = PANEL_W / XMAX
    GH = 3 * STEP
    PLOT_H = len(HEAD_GROUPS) * GH + (len(HEAD_GROUPS) - 1) * GROUP_GAP
    H = TOP + PLOT_H + 104
    out = [text(24, 36, "Forged traces lost to plain distillation: the paper's gap reversed", t, 18,
                t["ink"], weight="700"),
           text(24, 58, "Student accuracy (%) before and after training · (±) = change from the "
                "untrained student", t, 12.5, t["muted"])]
    out += swatch_legend(24, 88, [(t["ref"], "untrained student"),
                                  (t["distilled"], "trained on the surrogate's traces (plain distillation)"),
                                  (t["forged"], "trained on forged traces (the attack)")], t, 12.5)
    for i, bench in enumerate(BENCHES):
        x0 = LABEL_W + i * (PANEL_W + GAP)
        out.append(text(x0, TOP - 12, bench, t, 13, t["ink"], weight="700"))
        for v in range(0, int(XMAX) + 1, 20):
            gx = x0 + v * px
            out.append(line(gx, TOP - 2, gx, TOP + PLOT_H, t["axis"] if v == 0 else t["grid"]))
            out.append(text(gx, TOP + PLOT_H + 16, str(v), t, 11.5, t["muted"], anchor="middle",
                            tabular=True))
        for g, (_, _, vals) in enumerate(HEAD_GROUPS):
            gy = TOP + g * (GH + GROUP_GAP)
            base = vals[bench][0]
            for k, (v, role) in enumerate(zip(vals[bench], ("ref", "distilled", "forged"))):
                y = gy + k * STEP + (STEP - BAR) / 2
                out.append(hbar(x0, y, v * px, BAR, t[role]))
                d = v - base
                lab = f"{v:.1f}" if k == 0 else f"{v:.1f} ({'+' if d >= 0 else '−'}{abs(d):.1f})"
                out.append(text(x0 + v * px + 6, y + BAR - 3, lab, t, 12.5, t["ink"],
                                weight="400" if k == 0 else "600", tabular=True, halo=True))
    for g, (lab, sub, _) in enumerate(HEAD_GROUPS):
        gy = TOP + g * (GH + GROUP_GAP)
        out.append(text(24, gy + GH / 2 - 2, lab, t, 13, t["ink"], weight="600"))
        out.append(text(24, gy + GH / 2 + 14, sub, t, 11.5, t["muted"]))
        if g:
            out.append(line(24, gy - GROUP_GAP / 2, W - 24, gy - GROUP_GAP / 2, t["grid"]))
    for k, note in enumerate((
            f"Noise: 3 evaluation seeds moved the score of the student trained on forged traces by "
            f"{seed_range('MATH500'):.1f} (MATH500) and {seed_range('JEEBench'):.1f} (JEEBench) points.",
            "The paper reports single runs. Our untrained student is scored with thinking on, as every "
            "student is;",
            f"with thinking off it scores {NOTHINK['MATH500']} / {NOTHINK['JEEBench']}, above every "
            "trained student.")):
        out.append(text(24, H - 52 + k * 20, note, t, 12.5, t["muted"]))
    return svg(W, H, "\n".join(out), t,
               "Untrained student vs plain distillation vs forged traces, paper and our recreation")


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
    L2 = L1 + LH + 72
    for y, n, lab in ((L1, "1", "Learn to forge, on a surrogate whose reasoning is visible"),
                      (L2, "2", "Attack: the victim shows only its answer and a summary")):
        out.append(rect(16, y, W - 32, LH, t["panel"], r=12))
        out.append(f'<circle cx="40" cy="{y + 22}" r="11" fill="{t["ink"]}"/>')
        out.append(text(40, y + 26.5, n, t, 13, t["page"], anchor="middle", weight="700"))
        out.append(text(60, y + 27, lab, t, 14, t["ink"], weight="600"))
    BY1, BY2 = L1 + 44, L2 + 44
    lanes = (
        (BY1, [("Problems A", ["OpenThoughts", "5,000 prompts"], None, False),
               ("Surrogate", ["R1-Distill-Qwen-7B", "reasons in the open"], t["distilled"],
                False),
               ("Compressor", ["Qwen3.5-4B, zero-shot", "summarizes each trace"], None, False),
               ("Inverter: train", ["Qwen3.5-4B + LoRA", "problem, answer, summary",
                                       "→ reasoning trace"], t["forged"], True)]),
        (BY2, [("Problems B", ["OpenThoughts", "5,045 prompts answered"], None, False),
               ("Victim", ["Qwen3.8-27B, 4-bit", "shows answer + summary*", "hides its trace"],
                None, False),
               ("Inverter: apply", ["the trained adapter", "forges the hidden trace"], t["forged"],
                False),
               ("Student", ["Qwen3.5-2B, full fine-tune", "one per kind of training data",
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
    ax, ix, my = XS[3] + WS[3] / 2, XS[2] + WS[2] - 44, L1 + LH + 20
    out.append(f'<path d="M{ax},{BY1 + BH} V{my} H{ix} V{BY2 - 2}" fill="none" '
               f'stroke="{t["forged"]}" stroke-width="1.8" marker-end="url(#ah-forged)"/>')
    out.append(text((ax + ix) / 2, my - 7, "trained adapter", t, 12, t["forged"],
                    anchor="middle", weight="700", halo=True))

    # plain distillation: the surrogate's own traces train a student directly (the winning
    # baseline). It must cross the adapter link; a page-coloured underlay makes it a bridge.
    dx0, dy, dx1 = XS[1] + WS[1] / 2, L1 + LH + 50, XS[3] + 58
    dpath = f"M{dx0},{BY1 + BH} V{dy} H{dx1} V{BY2 - 2}"
    out.append(f'<path d="{dpath}" fill="none" stroke="{t["page"]}" stroke-width="7"/>')
    out.append(f'<path d="{dpath}" fill="none" stroke="{t["distilled"]}" stroke-width="1.8" '
               f'stroke-dasharray="6 4" marker-end="url(#ah-distilled)"/>')
    out.append(text(dx0 + 10, dy - 7, "its own traces → a student (plain distillation)", t, 12,
                    t["distilled"], weight="700", halo=True))

    # the victim's real traces: withheld from the attack, used for one comparison student
    vx, sx_ = XS[1] + WS[1] / 2, XS[3] + WS[3] / 2
    oy = L2 + LH + 24
    out.append(f'<path d="M{vx},{BY2 + BH} V{oy} H{sx_} V{BY2 + BH + 2}" fill="none" '
               f'stroke="{t["oracle"]}" stroke-width="1.8" stroke-dasharray="6 4" '
               f'marker-end="url(#ah-oracle)"/>')
    out.append(text((vx + sx_) / 2, oy + 20, "the victim's real traces: never shown to the "
                    "attack, used only to train one comparison student", t, 12.5, t["oracle"],
                    anchor="middle", weight="700"))
    H = oy + 84
    for k, note in enumerate((
            "* The victim has no summary API, so the compressor summarizes its hidden trace, as "
            "the paper presumably did for R1.",
            "Also trained, not drawn: students trained on the victim's answers only, or its "
            "summaries + answers.")):
        out.append(text(24, H - 32 + k * 20, note, t, 12, t["muted"]))
    return svg(W, H, "\n".join(out), t, "Pipeline: surrogate, compressor, inverter, victim, student")


# --------------------------------------------------------------------------- 3. students vs the victim
# The victim's and the 7B surrogate's own scores, cited from records rather than re-run (docs/results/phase6.md,
# reference rows): the victim as the attack queried it (medium effort, 250-problem subset,
# phase3.md §6) and the 7B surrogate (baselines.md, Phase 0). Both ran on llama.cpp, the students on
# vLLM, so these two rows are approximate.
VICTIM = {"MATH500": 97.2, "JEEBench": 82.0}
SURR7 = {"MATH500": 92.6, "JEEBench": 60.6}

LADDER = [  # label, key, colour role, share-of-the-way label?, is a reference (outlined) row
    ("Victim (Qwen3.8-27B)", "victim", "oracle", False, True),
    ("7B surrogate", "surrogate", "distilled", False, True),
    ("the victim's real traces", "oracle", "oracle", True, False),
    ("the surrogate's traces", "surr-7b", "distilled", True, False),
    ("forged traces", "synth-7b-sum", "forged", True, False),
    ("nothing (the untrained student)", "baseline-think", "ref", False, False),
    ("the victim's summaries + answers", "summary-answer", "ref", False, False),
    ("the victim's answers only", "answer-only", "ref", False, False),
]


def ladder_score(key, bench):
    return {"victim": VICTIM, "surrogate": SURR7}[key][bench] if key in ("victim", "surrogate") \
        else acc(key, bench)


def share_toward_victim(key, bench):
    """Share of the gap from the untrained student to the victim that a student closed, in %."""
    start = THINK[bench]
    return 100 * (acc(key, bench) - start) / (VICTIM[bench] - start)


def chart_results(mode):
    t = THEME[mode]
    W, LABEL_W, PANEL_W, GAP, TOP, ROW, BAR = 900, 272, 272, 46, 156, 30, 16
    XMAX = 100.0
    px = PANEL_W / XMAX
    SEP = 34  # gap between the reference rows (victim, surrogate) and the students
    PLOT_H = len(LADDER) * ROW + SEP
    H = TOP + PLOT_H + 110
    ry = lambda r: TOP + r * ROW + (SEP if r >= 2 else 0)
    out = [text(24, 36, "The victim's real traces got our student no closer to it than the surrogate's did",
                t, 18, t["ink"], weight="700"),
           text(24, 58, "Our recreation, 7B surrogate · accuracy (%) · outlined bars: the victim "
                "and the surrogate themselves", t, 12.5, t["muted"]),
           text(24, 77, "(%) = share of the way from the untrained student to the victim", t, 12.5,
                t["muted"])]
    out += swatch_legend(24, 108, [(t["oracle"], "victim's real traces"),
                                   (t["distilled"], "surrogate's traces"),
                                   (t["forged"], "forged traces (the attack)"),
                                   (t["ref"], "untrained / no reasoning")], t, 13)
    for i, bench in enumerate(BENCHES):
        x0 = LABEL_W + i * (PANEL_W + GAP)
        out.append(text(x0, TOP - 12, bench, t, 13, t["ink"], weight="700"))
        for v in range(0, int(XMAX) + 1, 20):
            gx = x0 + v * px
            out.append(line(gx, TOP, gx, TOP + PLOT_H, t["axis"] if v == 0 else t["grid"]))
            out.append(text(gx, TOP + PLOT_H + 15, str(v), t, 12, t["muted"],
                            anchor="middle", tabular=True))
        for r, (_, key, role, show_share, is_reference) in enumerate(LADDER):
            v = ladder_score(key, bench)
            y = ry(r) + (ROW - BAR) / 2
            if is_reference:  # outlined: a reference point, not a trained student
                out.append(rect(x0, y + 1, v * px, BAR - 2, t["page"], r=3, stroke=t[role], sw=2))
            else:
                out.append(hbar(x0, y, v * px, BAR, t[role]))
            cy = ry(r) + ROW / 2
            end = v
            if key == SEEDS[0]:  # range across 3 evaluation seeds, drawn where it was measured
                vals = [acc(k, bench) for k in SEEDS]
                lo, hi = x0 + min(vals) * px, x0 + max(vals) * px
                out.append(line(lo, cy, hi, cy, t["ink"], 1.6))
                out.append(line(lo, cy - 5, lo, cy + 5, t["ink"], 1.6))
                out.append(line(hi, cy - 5, hi, cy + 5, t["ink"], 1.6))
                end = max(vals)
            lab = f"{v:.1f}"
            if show_share:
                sh = share_toward_victim(key, bench)
                lab += f" ({sh:.0f} %)" if sh >= 0 else f" (−{-sh:.0f} %)"
            out.append(text(x0 + end * px + 6, cy + 4.5, lab, t, 13, t["ink"],
                            weight="600" if show_share else "400", tabular=True, halo=True))
    for r, (label, _, _, _, is_reference) in enumerate(LADDER):
        out.append(text(24, ry(r) + ROW / 2 + 4.5, label, t, 13.5, t["ink"],
                        weight="600" if is_reference else "400"))
    out.append(line(24, TOP + 2 * ROW + 8, W - 24, TOP + 2 * ROW + 8, t["grid"]))
    out.append(text(24, TOP + 2 * ROW + SEP - 2, "Student trained on:", t, 13, t["muted"],
                    weight="600"))
    out.append(text(24, H - 52, f"⊢⊣ range across 3 evaluation seeds: {seed_range('MATH500'):.1f} "
                    f"(MATH500), {seed_range('JEEBench'):.1f} (JEEBench) · every student is scored "
                    "with thinking on", t, 12, t["muted"]))
    out.append(text(24, H - 32, "Victim and surrogate were scored before the study on llama.cpp "
                    "(students on vLLM); the victim on a 250-problem", t, 12, t["muted"]))
    out.append(text(24, H - 12, "subset at a 14k-token limit, as the attack queried it. Their "
                    "positions are approximate.", t, 12, t["muted"]))
    return svg(W, H, "\n".join(out), t, "Students, the victim and the surrogate on one scale, our recreation")


# --------------------------------------------------------------------------- 4. termination
POINTS = [  # run, label, role, label dx, dy, anchor
    ("baseline-think", "untrained student", "ref", 10, -14, "end"),
    ("oracle", "victim's real traces", "oracle", 0, 24, "middle"),
    ("surr-7b", "surrogate's traces", "distilled", 12, 4, "start"),
    ("synth-7b-sum", "forged traces", "forged", 12, -8, "start"),
]
FLOORS = ("answer-only", "summary-answer")
VARIANTS = ("synth-7b-nosum",)    # plotted small, unlabelled
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
    out = [text(24, 36, "Accuracy tracked how often a student finished its answer", t, 18,
                t["ink"], weight="700"),
           text(24, 58, "JEEBench accuracy vs answers cut off at the 32k-token limit · cut-off "
                "answers score zero, so the link is partly by construction", t, 12.5,
                t["muted"])]
    out.append(text(24, 91, "Student trained on:", t, 13, t["muted"], weight="600"))
    out += swatch_legend(186, 90, [(t["oracle"], "the victim's real traces"),
                                   (t["distilled"], "the surrogate's traces"),
                                   (t["forged"], "forged traces")], t, 13)
    for v in range(0, 71, 10):
        out.append(line(sx(v), TOP, sx(v), TOP + ph, t["axis"] if v == 0 else t["grid"]))
        out.append(text(sx(v), TOP + ph + 17, f"{v}%", t, 12, t["muted"], anchor="middle",
                        tabular=True))
    for v in range(15, 51, 5):
        out.append(line(L, sy(v), L + pw, sy(v), t["grid"]))
        out.append(text(L - 8, sy(v) + 4, str(v), t, 12, t["muted"], anchor="end",
                        tabular=True))
    out.append(text(L + pw / 2, H - 12, "JEEBench answers cut off at the 32k-token limit →", t,
                    11.5, t["ink2"], anchor="middle"))
    out.append(f'<text x="20" y="{TOP + ph / 2:.1f}" font-family=\'{FONT}\' font-size="11.5" '
               f'fill="{t["ink2"]}" text-anchor="middle" '
               f'transform="rotate(-90 20 {TOP + ph / 2:.1f})">JEEBench accuracy % →</text>')
    # least-squares line through the trace-trained students, over their own x-range only
    x1, x2 = min(xs), max(xs)
    out.append(line(sx(x1), sy(a + b * x1), sx(x2), sy(a + b * x2), t["axis"], 2, "6 4"))
    out.append(text(sx(x2) + 6, sy(a + b * x2) + 4, f"fit, r = −{-r:.2f}", t, 10.5, t["muted"], halo=True))
    # the untrained model with thinking off: the bar no student reached
    ny = sy(NOTHINK["JEEBench"])
    out.append(line(L, ny, L + pw, ny, t["ink2"], 1.4, "4 4"))
    out.append(text(L + pw - 6, ny - 6, f"untrained student, thinking off · {NOTHINK['JEEBench']:.1f}: "
                    "no student reached it", t, 13, t["ink2"], anchor="end", weight="600",
                    halo=True))
    # the untrained thinking-on point sits above the fit: when it does finish, it is mostly right
    bx, by = sx(trunc("baseline-think", "JEEBench")), sy(acc("baseline-think", "JEEBench"))
    out.append(text(bx + 10, by - 54, "thinking on · cut off on 66 %;", t, 12.5, t["ink2"],
                    anchor="end", halo=True))
    out.append(text(bx + 10, by - 37, "its median answer runs to the limit", t, 12.5, t["ink2"], anchor="end",
                    halo=True))
    # no-trace floors: they stop, but never learned to reason
    fx = [sx(trunc(f, "JEEBench")) for f in FLOORS]
    fy = [sy(acc(f, "JEEBench")) for f in FLOORS]
    for x, y in zip(fx, fy):
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{t["ref"]}" '
                   f'stroke="{t["page"]}" stroke-width="2"/>')
    out.append(text(max(fx) + 12, sum(fy) / 2 + 4, "the victim's answers only / summaries + answers: "
                    "stop at once, but score low", t, 13, t["ink2"], halo=True))
    # same-length training data, about half the cut-offs: surrogate 7B vs forged 7B
    F = json.loads((ROOT / "bench" / "results" / "phase5" / "format-stats.json").read_text())
    med = lambda k: F["conditions"][k]["completion"]["median"] / 1000
    ax_, ay_ = sx(trunc("surr-7b", "JEEBench")), sy(acc("surr-7b", "JEEBench"))
    bx_, by_ = sx(trunc("synth-7b-sum", "JEEBench")), sy(acc("synth-7b-sum", "JEEBench"))
    out.append(line(ax_ + 9, ay_ + 5, bx_ - 7, by_ - 7, t["distilled"], 1.4, "3 3"))
    out.append(text(sx(31.5), sy(43.0), f"similar-length training data (~{med('surr-7b'):.1f}k vs "
                    f"~{med('synth-7b-sum'):.1f}k tokens),", t, 13, t["ink2"], italic=True,
                    halo=True))
    out.append(text(sx(31.5), sy(43.0) + 18, "about half the cut-offs", t, 13, t["ink2"],
                    italic=True, halo=True))
    for run in VARIANTS:
        out.append(text(sx(trunc(run, "JEEBench")) - 10, sy(acc(run, "JEEBench")) + 4,
                        "forged traces, without summary", t, 12, t["ink2"], anchor="end", halo=True))
        out.append(f'<circle cx="{sx(trunc(run, "JEEBench")):.1f}" '
                   f'cy="{sy(acc(run, "JEEBench")):.1f}" r="5" fill="{t["forged"]}" '
                   f'stroke="{t["page"]}" stroke-width="1.5" opacity="0.75"/>')
    for run, label, role, dx, dy, anchor in POINTS:
        x, y = sx(trunc(run, "JEEBench")), sy(acc(run, "JEEBench"))
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{t[role]}" '
                   f'stroke="{t["page"]}" stroke-width="2"/>')
        out.append(text(x + dx, y + dy, label, t, 13.5, t["ink"], anchor=anchor, weight="600",
                        halo=True))
    return svg(W, H, "\n".join(out), t, "JEEBench accuracy vs share of answers cut off, by student")


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
    gap = lambda b: [round(v[b][2] - v[b][1], 1) for _, _, v in HEAD_GROUPS]  # forged − distillation
    assert gap("JEEBench") == [16.6, -9.9] and gap("MATH500") == [8.6, -7.6]
    assert round(seed_range("MATH500"), 1) == 3.4 and round(seed_range("JEEBench"), 1) == 5.3
    shares = [round(share_toward_victim(k, "JEEBench")) for k in ("surr-7b", "oracle", "synth-7b-sum")]
    assert shares == [24, 24, 4], shares
    print("self-check passed")
