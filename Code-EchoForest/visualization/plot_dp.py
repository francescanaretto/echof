#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import re
import pandas as pd
import matplotlib.pyplot as plt

BASE_FONT_SIZE = 15
plt.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": BASE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE,
        "xtick.labelsize": BASE_FONT_SIZE - 1,
        "ytick.labelsize": BASE_FONT_SIZE - 1,
        "legend.fontsize": BASE_FONT_SIZE - 1,
    }
)

# CONFIG
REPORTS_DIR = Path("../Reports-eval")

OUT_IMG = Path("f1_macro_dpquery_plateau.png")
OUT_CSV = Path("f1_macro_dpquery_plateau.csv")

DP_ORDER = ["0.1", "0.5", "1.0", "5.0"]

TARGET_KIND = "logit"
TARGET_GUIDING_BB = "nn"
TARGET_PERCENTILE = "25"
TARGET_MECH = "laplace"

# PARSER
REPORT_RE = re.compile(
    r"^report_original_on_synth_test_(.+?)_(.+?)_(\d+)_dpquery_([0-9.]+)_(.+?)\.json$"
)

def load_report_dict(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()

    try:
        return json.loads(txt)
    except Exception:
        pass

    pos = txt.find("{")
    if pos == -1:
        return None

    try:
        return json.loads(txt[pos:])
    except Exception:
        return None

def parse_one_report(path: Path):
    m = REPORT_RE.match(path.name)
    if not m:
        return None

    kind, guiding_bb, percentile, dp_eps, mech = m.groups()

    data = load_report_dict(path)
    if data is None:
        return None

    try:
        f1_macro = float(data["macro avg"]["f1-score"])
    except Exception:
        return None

    return {
        "dataset": path.parent.name,
        "kind": kind,
        "guiding_bb": guiding_bb,
        "percentile": str(percentile),
        "dp_epsilon": str(dp_eps),
        "mechanism": mech,
        "f1_macro": f1_macro,
    }

# LOAD
rows = []

for dataset_dir in REPORTS_DIR.iterdir():
    if not dataset_dir.is_dir():
        continue

    for path in dataset_dir.glob("report_original_on_synth_test_*.json"):
        rec = parse_one_report(path)
        if rec is not None:
            rows.append(rec)

if not rows:
    raise ValueError("No report found.")

df = pd.DataFrame(rows)

# FILTER
df = df[
    (df["kind"] == TARGET_KIND) &
    (df["guiding_bb"] == TARGET_GUIDING_BB) &
    (df["percentile"] == TARGET_PERCENTILE) &
    (df["mechanism"] == TARGET_MECH) &
    (df["dp_epsilon"].isin(DP_ORDER))
].copy()

if df.empty:
    raise ValueError("The filter is too restrictive: no data found.")

# MEDIA PER EPSILON
summary = (
    df.groupby("dp_epsilon")["f1_macro"]
      .mean()
      .reindex(DP_ORDER)
      .reset_index()
)

summary.to_csv(OUT_CSV, index=False)

print("\n=== Mean macro F1 ===")
print(summary)

# LINE PLOT (plateau)
x = summary["dp_epsilon"].astype(float)
y = summary["f1_macro"]

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(x, y, marker="o")

ax.set_title("Macro F1 vs DP-query epsilon")
ax.set_xlabel("DP-query epsilon")
ax.set_ylabel("Macro F1")

ax.set_ylim(0, min(1.0, float(y.max() + 0.05)))
ax.grid(True, linestyle="--", alpha=0.4)

# etichette sui punti
for xi, yi in zip(x, y):
    ax.text(xi, yi + 0.01, f"{yi:.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_IMG, dpi=200)
plt.close()

print(f"[OK] Plot saved to: {OUT_IMG.resolve()}")
