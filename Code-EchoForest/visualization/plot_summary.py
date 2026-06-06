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

OUT_IMG = Path("f1_macro_summary_all_datasets.png")
OUT_CSV = Path("f1_macro_summary_all_datasets.csv")
OUT_LONG_CSV = Path("f1_macro_extracted_reports.csv")

METHODS = [
    ("entropy", "25", "entropy 25"),
    ("entropy", "50", "entropy 50"),
    ("margin",  "25", "margin"),
    ("logit",   "25", "logit"),
    ("kappa",   "25", "kappa"),
]

# PARSER
REPORT_RE = re.compile(
    r"^report_synth_on_real_(train|test)_(.+?)_(.+?)_(\d+)\.json$"
)

def load_report_dict(path: Path):
    """
    Support both:
      1) plain JSON
      2) a text header followed by a JSON block
    """
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()

    # tentativo 1: JSON puro
    try:
        return json.loads(txt)
    except Exception:
        pass

    # tentativo 2: trova il primo blocco JSON
    pos = txt.find("{")
    if pos == -1:
        return None

    txt_json = txt[pos:]
    try:
        return json.loads(txt_json)
    except Exception:
        return None

def parse_one_report(path: Path):
    m = REPORT_RE.match(path.name)
    if not m:
        return None

    split, kind, guiding_bb, percentile = m.groups()

    data = load_report_dict(path)
    if data is None:
        return None

    try:
        f1_macro = float(data["macro avg"]["f1-score"])
    except Exception:
        return None

    return {
        "dataset": path.parent.name,
        "split": split,
        "kind": kind,
        "guiding_bb": guiding_bb,
        "percentile": str(percentile),
        "f1_macro": f1_macro,
        "source_file": path.name,
    }

# LOAD ALL REPORTS
rows = []

if not REPORTS_DIR.exists():
    raise FileNotFoundError(f"Directory not found: {REPORTS_DIR.resolve()}")

for dataset_dir in REPORTS_DIR.iterdir():
    if not dataset_dir.is_dir():
        continue

    for path in dataset_dir.glob("report_synth_on_real_*.json"):
        rec = parse_one_report(path)
        if rec is not None:
            rows.append(rec)

if not rows:
    raise ValueError(f"No valid JSON report found in {REPORTS_DIR.resolve()}")

df = pd.DataFrame(rows)

# FILTER METHODS OF INTEREST
wanted = {(k, p) for k, p, _ in METHODS}
df = df[df.apply(lambda r: (r["kind"], r["percentile"]) in wanted, axis=1)].copy()

if df.empty:
    raise ValueError("No file matches the requested methods.")

label_map = {(k, p): label for k, p, label in METHODS}
order_map = {(k, p): i for i, (k, p, _) in enumerate(METHODS)}

df["method"] = df.apply(lambda r: label_map[(r["kind"], r["percentile"])], axis=1)
df["method_order"] = df.apply(lambda r: order_map[(r["kind"], r["percentile"])], axis=1)

# Save the long-format table.
df = df.sort_values(["dataset", "split", "method_order"])
df.to_csv(OUT_LONG_CSV, index=False)





# MEAN + STD (train + test insieme)
summary = (
    df.groupby(["method", "method_order"])["f1_macro"]
      .agg(["mean", "std"])
      .reset_index()
      .sort_values("method_order")
)

# Save the aggregate CSV.
summary.to_csv(OUT_CSV, index=False)

print("\n=== Mean ± Std macro F1 (train + test) ===")
print(summary.round(4))

# PLOT (una barra + errore)
fig, ax = plt.subplots(figsize=(9, 5.5))

x = summary["method"]
y = summary["mean"]
yerr = summary["std"]

ax.bar(x, y, yerr=yerr, capsize=5)

ax.set_title("Average macro F1 across datasets (train + test)")
ax.set_xlabel("Method")
ax.set_ylabel("Macro F1")

ax.set_ylim(0, min(1.0, float(y.max() + yerr.max() + 0.05)))

ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.xticks(rotation=0)

# Add labels above the bars.
for i, (mean, std) in enumerate(zip(y, yerr)):
    ax.text(i, mean + std + 0.01, f"{mean:.3f}±{std:.3f}",
            ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_IMG, dpi=200, bbox_inches="tight")
plt.close()

print(f"[OK] Plot saved to: {OUT_IMG.resolve()}")
