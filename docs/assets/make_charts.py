#!/usr/bin/env python3
"""Render the README's three result charts as light/dark SVG pairs.

Reads bench/results/phase6/summary.json (the committed Phase 6 record) plus the cited rows that
were never re-run in Phase 6 (Phase 0's no-think baseline, the paper's Table 3). No dependencies.

    python3 docs/assets/make_charts.py      # writes docs/assets/*-{light,dark}.svg
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets"
S = json.loads((ROOT / "bench" / "results" / "phase6" / "summary.json").read_text())

NOTHINK = {"MATH500": 79.0, "JEEBench": 47.8}          # Phase 0, template-default (no-think) render
THINK = {b: S["baseline-think"]["bench"][b]["acc"] for b in NOTHINK}

# Paper Table 3, Qwen2.5-7B student, R1-Weak surrogate, victim R1 (docs/02): JEEBench
PAPER_JEE = {"base": 28.3, "answer-only": 21.6, "summary-answer": 24.0, "surr": 19.7,
             "synth": 36.3, "oracle": 43.7}
PAPER_MATH = {"base": 71.2, "answer-only": 61.0, "summary-answer": 63.0, "surr": 63.2,
              "synth": 71.8, "oracle": 72.2}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
# validated with the dataviz validator: 3 categorical slots pass all-pairs in both modes;
# blue/red is the diverging pair. Reference rows (baselines, floors) are neutral gray.
THEME = {
    "light": dict(ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  page="#ffffff", forged="#2a78d6", distilled="#eb6834", oracle="#1baf7a",
                  ref="#a8a69f", up="#2a78d6", down="#e34948"),
    "dark": dict(ink="#f0f0ee", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 page="#0d1117", forged="#3987e5", distilled="#d95926", oracle="#199e70",
                 ref="#6e6d68", up="#3987e5", down="#e66767"),
}


def acc(run, bench):
    return S[run]["bench"][bench]["acc"]


def trunc(run, bench):
    b = S[run]["bench"][bench]
    return 100.0 * b["truncated"] / b["n"]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, t, size=12, fill=None, anchor="start", weight="normal", tabular=False,
         halo=False):
    style = "font-variant-numeric: tabular-nums;" if tabular else ""
    # halo = page-coloured stroke behind the glyphs so a label crossing a line stays legible
    halo_attr = (f' paint-order="stroke" stroke="{t["page"]}" stroke-width="3" '
                 f'stroke-linejoin="round"') if halo else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family=\'{FONT}\' font-size="{size}" '
            f'fill="{fill or t["ink2"]}" text-anchor="{anchor}" font-weight="{weight}" '
            f'style="{style}"{halo_attr}>{esc(s)}</text>')


def hbar(x0, y, w, h, fill, r=4):
    """Horizontal bar growing right from x0; rounded on the data end only."""
    if w <= 0:
        return ""
    r = min(r, w / 2, h / 2)
    x1 = x0 + w
    return (f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} a{r},{r} 0 0 1 {r},{r} V{y + h - r:.1f} '
            f'a{r},{r} 0 0 1 -{r},{r} H{x0:.1f} Z" fill="{fill}"/>')


def hbar_left(x0, y, w, h, fill, r=4):
    """Bar growing LEFT from x0 (negative deltas); rounded on the data end only."""
    if w <= 0:
        return ""
    r = min(r, w / 2, h / 2)
    x1 = x0 - w
    return (f'<path d="M{x0:.1f},{y:.1f} H{x1 + r:.1f} a{r},{r} 0 0 0 -{r},{r} V{y + h - r:.1f} '
            f'a{r},{r} 0 0 0 {r},{r} H{x0:.1f} Z" fill="{fill}"/>')


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img">\n{body}\n</svg>\n')


# --------------------------------------------------------------------------- chart 1: results
ROWS = [  # (label, run key or None for a group header, colour role)
    ("Trained on what the victim shows", None, None),
    ("answers only", "answer-only", "ref"),
    ("summaries + answers", "summary-answer", "ref"),
    ("Trained on forged traces (the attack)", None, None),
    ("7B surrogate arm, with summary", "synth-7b-sum", "forged"),
    ("7B arm, no summary", "synth-7b-nosum", "forged"),
    ("7B arm, with summary — LoRA twin", "synth-7b-sum-lora", "forged"),
    ("1.5B surrogate arm, with summary", "synth-1.5b-sum", "forged"),
    ("1.5B arm, no summary", "synth-1.5b-nosum", "forged"),
    ("Trained on the surrogate's own traces (plain distillation)", None, None),
    ("7B surrogate", "surr-7b", "distilled"),
    ("1.5B surrogate", "surr-1.5b", "distilled"),
    ("Trained on the victim's real, withheld traces", None, None),
    ("oracle", "oracle", "oracle"),
]
SEEDS = ["synth-7b-sum", "synth-7b-sum-seed1235", "synth-7b-sum-seed1236"]


def chart_results(mode):
    t = THEME[mode]
    W, LABEL_W, PANEL_W, GAP, TOP = 900, 300, 260, 50, 84
    ROW, HEAD, BAR = 24, 30, 14
    px = PANEL_W / 100.0
    out = []
    # layout rows
    y = TOP
    ys = []
    for label, run, role in ROWS:
        if run is None:
            y += 8
            ys.append((y, "head"))
            y += HEAD - 8
        else:
            ys.append((y, "row"))
            y += ROW
    H = y + 46
    panels = [("MATH500", LABEL_W), ("JEEBench", LABEL_W + PANEL_W + GAP)]
    out.append(text(16, 24, "Student accuracy after training, by what it was trained on", t, 15,
                    t["ink"], weight="600"))
    out.append(text(16, 42, "Qwen3.5-2B, 1,015 problems, one protocol · dashed lines = the same "
                    "model untrained", t, 12, t["muted"]))
    for bench, x0 in panels:
        out.append(text(x0, TOP - 22, bench, t, 12, t["ink"], weight="600"))
        # gridlines every 25
        for v in (0, 25, 50, 75, 100):
            gx = x0 + v * px
            out.append(f'<line x1="{gx:.1f}" y1="{TOP - 2}" x2="{gx:.1f}" y2="{H - 40}" '
                       f'stroke="{t["grid"]}" stroke-width="1"/>')
            out.append(text(gx, H - 26, str(v), t, 10, t["muted"], anchor="middle", tabular=True))
        # reference lines, labelled at the top: "thinking on" to the left of its line,
        # "no-think" to the right, so the two labels never meet
        for val, lab, col, anchor, dx in (
                (THINK[bench], f"thinking on {THINK[bench]:.1f}", t["muted"], "end", -4),
                (NOTHINK[bench], f"no-think {NOTHINK[bench]:.1f}", t["ink"], "start", 4)):
            gx = x0 + val * px
            out.append(f'<line x1="{gx:.1f}" y1="{TOP - 2}" x2="{gx:.1f}" y2="{H - 40}" '
                       f'stroke="{col}" stroke-width="1.5" stroke-dasharray="4 3"/>')
            out.append(text(gx + dx, TOP - 7, lab, t, 10, col, anchor=anchor, halo=True))
        for (label, run, role), (ry, kind) in zip(ROWS, ys):
            if kind == "head":
                continue
            v = acc(run, bench)
            out.append(hbar(x0, ry + (ROW - BAR) / 2, v * px, BAR, t[role]))
            out.append(text(x0 + v * px + 6, ry + ROW / 2 + 4, f"{v:.1f}", t, 11, t["ink2"],
                            tabular=True, halo=True))
            if run == "synth-7b-sum":  # seed range bracket, 3 eval seeds
                vals = [acc(r, bench) for r in SEEDS]
                lo, hi = x0 + min(vals) * px, x0 + max(vals) * px
                cy = ry + ROW / 2
                out.append(f'<line x1="{lo:.1f}" y1="{cy - BAR / 2 - 4}" x2="{hi:.1f}" '
                           f'y2="{cy - BAR / 2 - 4}" stroke="{t["ink"]}" stroke-width="1.5"/>')
                for gx in (lo, hi):
                    out.append(f'<line x1="{gx:.1f}" y1="{cy - BAR / 2 - 7}" x2="{gx:.1f}" '
                               f'y2="{cy - BAR / 2 - 1}" stroke="{t["ink"]}" stroke-width="1.5"/>')
    for (label, run, role), (ry, kind) in zip(ROWS, ys):
        if kind == "head":
            out.append(text(16, ry + 14, label, t, 11, t["muted"], weight="600"))
        else:
            out.append(text(28, ry + ROW / 2 + 4, label, t, 12, t["ink"]))
    out.append(text(16, H - 12, "⊢⊣ range across 3 evaluation seeds on the 7B-arm forged cell "
                    "(MATH500 3.4 pts, JEEBench 5.3 pts)", t, 10, t["muted"]))
    return svg(W, H, "\n".join(out))


# --------------------------------------------------------------------------- chart 2: paper vs ours
DROWS = [("answers only", "answer-only"), ("summaries + answers", "summary-answer"),
         ("surrogate's own traces", "surr"), ("forged traces", "synth"),
         ("victim's real traces (oracle)", "oracle")]


def ours_delta(key, bench):
    run = {"answer-only": "answer-only", "summary-answer": "summary-answer", "surr": "surr-7b",
           "synth": "synth-7b-sum", "oracle": "oracle"}[key]
    return acc(run, bench) - NOTHINK[bench]


def chart_paper_vs_ours(mode):
    t = THEME[mode]
    W, LABEL_W, PANEL_W, GAP, TOP, ROW, BAR = 900, 210, 300, 40, 74, 30, 14
    LO, HI = -30.0, 20.0
    px = PANEL_W / (HI - LO)
    H = TOP + len(DROWS) * ROW + 44
    out = [text(16, 24, "Change in JEEBench accuracy vs the untrained student", t, 15, t["ink"],
                weight="600"),
           text(16, 42, "percentage points · the paper's headline setting beside the same five "
                "conditions here (7B surrogate arm)", t, 12, t["muted"])]
    panels = [("Paper · Qwen2.5-7B student, 1.5B surrogate, victim R1",
               {r: PAPER_JEE[r] - PAPER_JEE["base"] for _, r in DROWS}, LABEL_W),
              ("Here · Qwen3.5-2B student, 7B surrogate, victim 27B",
               {r: ours_delta(r, "JEEBench") for _, r in DROWS}, LABEL_W + PANEL_W + GAP)]
    for title, deltas, x0 in panels:
        out.append(text(x0, TOP - 10, title, t, 11, t["ink"], weight="600"))
        zero = x0 + (0 - LO) * px
        for v in (-30, -20, -10, 0, 10, 20):
            gx = x0 + (v - LO) * px
            out.append(f'<line x1="{gx:.1f}" y1="{TOP}" x2="{gx:.1f}" y2="{H - 36}" '
                       f'stroke="{t["axis"] if v == 0 else t["grid"]}" stroke-width="1"/>')
            out.append(text(gx, H - 22, f"{v:+d}" if v else "0", t, 10, t["muted"],
                            anchor="middle", tabular=True))
        for i, (label, key) in enumerate(DROWS):
            d = deltas[key]
            ry = TOP + i * ROW + (ROW - BAR) / 2
            if d >= 0:
                out.append(hbar(zero + 1, ry, d * px - 1, BAR, t["up"]))
                out.append(text(zero + d * px + 6, ry + BAR - 3, f"+{d:.1f}", t, 11, t["ink2"],
                                tabular=True))
            else:
                out.append(hbar_left(zero - 1, ry, -d * px - 1, BAR, t["down"]))
                out.append(text(zero + d * px - 6, ry + BAR - 3, f"{d:.1f}", t, 11, t["ink2"],
                                anchor="end", tabular=True))
    for i, (label, key) in enumerate(DROWS):
        out.append(text(16, TOP + i * ROW + ROW / 2 + 4, label, t, 12, t["ink"]))
    out.append(text(16, H - 6, "paper: Table 3 (docs/02) · here: measured against the no-think "
                    "baseline 47.8 — every cell lands below it; the ordering is the finding",
                    t, 10, t["muted"]))
    return svg(W, H, "\n".join(out))


# --------------------------------------------------------------------------- chart 3: termination
POINTS = [  # run, short label, role, label dx, dy, anchor
    ("baseline-think", "untrained, thinking on", "ref", -10, 4, "end"),
    ("answer-only", "answers only", "ref", 10, -4, "start"),
    ("summary-answer", "summaries + answers", "ref", 10, 12, "start"),
    ("oracle", "oracle (real traces)", "oracle", 10, -6, "start"),
    ("surr-7b", "distilled 7B surrogate", "distilled", 10, 12, "start"),
    ("surr-1.5b", "distilled 1.5B surrogate", "distilled", 10, 4, "start"),
    ("synth-7b-sum", "forged, 7B arm (sum)", "forged", 10, -6, "start"),
    ("synth-7b-sum-lora", "forged, 7B arm (LoRA)", "forged", -10, 14, "end"),
    ("synth-7b-nosum", "forged, 7B arm (no sum)", "forged", 10, 12, "start"),
    ("synth-1.5b-sum", "forged, 1.5B arm (sum)", "forged", 10, 4, "start"),
    ("synth-1.5b-nosum", "forged, 1.5B arm (no sum)", "forged", 10, 12, "start"),
]


def chart_termination(mode):
    t = THEME[mode]
    W, H, L, R, TOP, BOT = 760, 400, 64, 40, 70, 48
    XLO, XHI, YLO, YHI = 0.0, 70.0, 15.0, 50.0
    pw, ph = W - L - R, H - TOP - BOT
    sx = lambda v: L + (v - XLO) / (XHI - XLO) * pw
    sy = lambda v: TOP + (YHI - v) / (YHI - YLO) * ph
    out = [text(16, 24, "Accuracy tracks termination: JEEBench score vs share of answers "
                "cut off at the 32k-token cap", t, 15, t["ink"], weight="600"),
           text(16, 42, "every trained student loops less than the untrained model — how much "
                "less depends on how long its training traces were", t, 12, t["muted"])]
    for v in range(0, 71, 10):
        out.append(f'<line x1="{sx(v):.1f}" y1="{TOP}" x2="{sx(v):.1f}" y2="{TOP + ph}" '
                   f'stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(text(sx(v), TOP + ph + 16, f"{v}%", t, 10, t["muted"], anchor="middle",
                        tabular=True))
    for v in range(15, 51, 5):
        out.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{L + pw}" y2="{sy(v):.1f}" '
                   f'stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(text(L - 8, sy(v) + 4, str(v), t, 10, t["muted"], anchor="end", tabular=True))
    out.append(text(L + pw / 2, H - 8, "JEEBench problems truncated at the cap", t, 11,
                    t["muted"], anchor="middle"))
    out.append(f'<text x="14" y="{TOP + ph / 2:.1f}" font-family=\'{FONT}\' font-size="11" '
               f'fill="{t["muted"]}" text-anchor="middle" '
               f'transform="rotate(-90 14 {TOP + ph / 2:.1f})">JEEBench accuracy %</text>')
    for run, label, role, dx, dy, anchor in POINTS:
        x, y = sx(trunc(run, "JEEBench")), sy(acc(run, "JEEBench"))
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{t[role]}" '
                   f'stroke="{t["page"]}" stroke-width="2"/>')
        out.append(text(x + dx, y + dy, label, t, 11, t["ink"], anchor=anchor))
    # legend
    lx = L
    for role, lab in (("forged", "forged traces"), ("distilled", "surrogate's own traces"),
                      ("oracle", "victim's real traces"), ("ref", "no trace / untrained")):
        out.append(f'<circle cx="{lx + 6}" cy="{TOP - 12}" r="5" fill="{t[role]}"/>')
        out.append(text(lx + 16, TOP - 8, lab, t, 11, t["ink2"]))
        lx += 18 + 7 * len(lab) + 24
    return svg(W, H, "\n".join(out))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("results", chart_results), ("paper-vs-ours", chart_paper_vs_ours),
                     ("termination", chart_termination)):
        for mode in ("light", "dark"):
            p = OUT / f"{name}-{mode}.svg"
            p.write_text(fn(mode))
            print("wrote", p.relative_to(ROOT))
    # self-check: the numbers the README quotes are the ones in the record
    assert acc("oracle", "JEEBench") == 45.6 and acc("surr-7b", "JEEBench") == 45.4
    assert abs(trunc("baseline-think", "JEEBench") - 65.6) < 0.1
    print("self-check passed")
