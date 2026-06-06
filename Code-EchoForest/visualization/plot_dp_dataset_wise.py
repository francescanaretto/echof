#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
from pathlib import Path

import pandas as pd
import matplotlib
import numpy as np

# Use a non-interactive backend and a writable Matplotlib cache so the script
# runs reliably on remote/sandboxed machines.
if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = str((Path(__file__).resolve().parents[2] / ".mplconfig").resolve())
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

# Fixed plot title requested for the DP sweep figure.
PLOT_TITLE = "Differantil Privacy during synthetic data generation"
Y_AXIS_MIN = 0.00
Y_AXIS_MAX = 1.00
MEAN_LINE_COLOR = "#5E548E"  # dark purple
TEXT_COLOR = "#3F3F3F"

# CONFIG
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR_WISE = PROJECT_ROOT / "Model-synthetic-wise"
DATA_DIR = PROJECT_ROOT / "Data-original"

# Reports root. If REPORTS_DIR is relative, resolve it against the project root so the
# script works regardless of the current working directory.
_reports_dir_env = os.environ.get("REPORTS_DIR", "").strip()
if _reports_dir_env:
    _p = Path(_reports_dir_env)
    REPORTS_DIR = (_p if _p.is_absolute() else (PROJECT_ROOT / _p)).resolve()
else:
    REPORTS_DIR = (PROJECT_ROOT / "Reports-eval").resolve()

# Which report family to read. Must match the filename prefix:
#   report_{REPORT_TARGET}_{kind}_{guiding_bb}_{percentile}_dpquery_...
# Examples:
#   original_on_real_test
#   original_on_real_train
#   original_on_synth_test
REPORT_TARGET = os.environ.get("REPORT_TARGET", "original_on_real_test").strip()

DP_ORDER = [
    item.strip()
    for item in os.environ.get("DP_ORDER", "0.1,0.5,1.0,5.0").split(",")
    if item.strip()
]
DP_ORDER_MAP = {eps: idx for idx, eps in enumerate(DP_ORDER)}

TARGET_KIND = "logit"
TARGET_GUIDING_BB = "nn"
TARGET_PERCENTILE = "25"
TARGET_MECHANISMS = ["laplace", "gaussian"]
# If None, use all discovered datasets.
# Otherwise provide an explicit list, for example:
SELECT_DATASETS = ["activity",  "adult", "pol", "spotify", "spotify-r", "landsat-multi"]
#SELECT_DATASETS = None

COMBO_CASES = [
    {
        "name": "true",
        "variants": {"explicit_true"},
        "label": "noise_on_labeling=True",
    },
    {
        "name": "false",
        "variants": {"no_flag"},
        "label": "noise_on_labeling=False (no flag nel nome)",
    },
]

# PARSER
REPORT_RE = re.compile(
    r"^(?P<kind>.+?)_(?P<guiding_bb>.+?)_(?P<percentile>\d+)_"
    r"dpquery_(?P<dp_eps>[0-9.]+)_(?P<mech>.+?)"
    r"(?:_(?P<noise_on_labeling>True|False)(?:_wise)?)?\.json$"
)

MODEL_RE = re.compile(
    r"^rf_dpquery_(?P<dataset>.+?)_(?P<kind>.+?)_(?P<guiding_bb>.+?)_(?P<percentile>\d+)"
    r"_(?P<dp_eps>[0-9.]+)_(?P<mech>laplace|gaussian)(?P<rest>.*)\.sav$"
)

def parse_optional_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None

    txt = str(value).strip().lower()
    if txt in {"true", "1", "yes"}:
        return True
    if txt in {"false", "0", "no"}:
        return False
    return None

def detect_noise_variant(raw_flag):
    if raw_flag == "True":
        return "explicit_true"
    if raw_flag == "False":
        return "explicit_false"
    return "no_flag"

def detect_noise_variant_from_value(raw_flag, inferred_value):
    """
    Historically, some runs encoded noise_on_labeling only in the JSON payload
    (or omitted it entirely), while the filename flag was missing.
    For plotting, we want the "true/false" split to follow the *actual* setting
    whenever we can infer it, falling back to the filename convention otherwise.
    """
    v = parse_optional_bool(inferred_value)
    if v is True:
        return "explicit_true"
    if v is False:
        return "explicit_false"
    return detect_noise_variant(raw_flag)

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
    prefix = f"report_{REPORT_TARGET}_"
    if not path.name.startswith(prefix):
        return None

    suffix = path.name[len(prefix):]
    m = REPORT_RE.match(suffix)
    if not m:
        return None

    kind = m.group("kind")
    guiding_bb = m.group("guiding_bb")
    percentile = m.group("percentile")
    dp_eps = m.group("dp_eps")
    mech = m.group("mech")

    data = load_report_dict(path)
    if data is None:
        return None

    try:
        f1_macro = float(data["macro avg"]["f1-score"])
    except Exception:
        return None

    raw_flag = m.group("noise_on_labeling")
    noise_on_labeling = parse_optional_bool(raw_flag)

    if noise_on_labeling is None:
        for key in ("noise_on_labeling", "query_noise"):
            noise_on_labeling = parse_optional_bool(data.get(key))
            if noise_on_labeling is not None:
                break

    if noise_on_labeling is None and isinstance(data.get("query_summary"), dict):
        for key in ("noise_on_labeling", "query_noise"):
            noise_on_labeling = parse_optional_bool(data["query_summary"].get(key))
            if noise_on_labeling is not None:
                break

    return {
        "dataset": path.parent.name,
        "report_target": REPORT_TARGET,
        "kind": kind,
        "guiding_bb": guiding_bb,
        "percentile": str(percentile),
        "dp_epsilon": str(dp_eps),
        "mechanism": mech,
        "noise_variant": detect_noise_variant_from_value(raw_flag, noise_on_labeling),
        "noise_on_labeling": noise_on_labeling,
        "f1_macro": f1_macro,
        "source_file": path.name,
    }


def _read_labels_1d(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    if df.shape[1] == 0:
        raise ValueError(f"Empty labels file: {csv_path}")
    s = df.iloc[:, -1].astype(str).str.strip()
    s = s[s.str.lower() != "labels"]
    return s.reset_index(drop=True)


def load_real_splits_scaled(dataset: str):
    """
    Load real train/test and apply StandardScaler fitted on train.
    This matches the preprocessing used across the repo for RF surrogates.
    """
    dd = DATA_DIR / dataset
    xtr_df = pd.read_csv(dd / f"train_set_{dataset}.csv")
    xte_df = pd.read_csv(dd / f"test_set_{dataset}.csv")
    xtr_df = xtr_df.loc[:, ~xtr_df.columns.astype(str).str.startswith("Unnamed")]
    xte_df = xte_df.loc[:, ~xte_df.columns.astype(str).str.startswith("Unnamed")]

    ytr_s = _read_labels_1d(dd / f"train_labels_{dataset}.csv")
    yte_s = _read_labels_1d(dd / f"test_labels_{dataset}.csv")

    le = LabelEncoder()
    ytr = le.fit_transform(ytr_s.to_numpy())
    yte = le.transform(yte_s.to_numpy())

    scaler = StandardScaler()
    xtr = scaler.fit_transform(xtr_df.values).astype(np.float32)
    xte = scaler.transform(xte_df.values).astype(np.float32)
    return xtr, ytr, xte, yte


def parse_one_model(path: Path):
    """
    Support legacy dpquery models saved without the noise_on_labeling flag in the report name.
    Example:
      rf_dpquery_activity_logit_nn_25_5.0_laplace_wise.sav
    These correspond to noise_on_labeling=False (no flag in earlier naming).
    """
    m = MODEL_RE.match(path.name)
    if not m:
        return None

    meta = m.groupdict()
    dataset = meta["dataset"]
    kind = meta["kind"]
    guiding_bb = meta["guiding_bb"]
    percentile = meta["percentile"]
    dp_eps = meta["dp_eps"]
    mech = meta["mech"]
    rest = meta.get("rest") or ""

    if kind != TARGET_KIND or guiding_bb != TARGET_GUIDING_BB or percentile != TARGET_PERCENTILE:
        return None
    if mech not in TARGET_MECHANISMS:
        return None
    if dp_eps not in DP_ORDER:
        return None

    # Evaluate macro-F1 on the requested target split.
    try:
        _, _, xte, yte = load_real_splits_scaled(dataset)
        with path.open("rb") as f:
            model = pickle.load(f)
        y_pred = np.asarray(model.predict(xte))
        f1_macro = float(f1_score(yte, y_pred, average="macro"))
    except Exception:
        return None

    return {
        "dataset": dataset,
        "report_target": "model_eval_on_real_test",
        "kind": kind,
        "guiding_bb": guiding_bb,
        "percentile": str(percentile),
        "dp_epsilon": str(dp_eps),
        "mechanism": mech,
        # Legacy naming: treat as the "false" case (no flag).
        "noise_variant": "no_flag",
        "noise_on_labeling": False,
        "f1_macro": f1_macro,
        "source_file": path.name,
    }


def build_combo_summary(df: pd.DataFrame, mechanism: str, variants: set[str]) -> pd.DataFrame:
    base_index = pd.Index(DP_ORDER, name="dp_epsilon")
    combo_df = df[
        (df["mechanism"] == mechanism) &
        (df["noise_variant"].isin(variants))
    ].copy()
    summary = (
        combo_df.groupby("dp_epsilon")["f1_macro"]
                .agg(["mean", "std", "count"])
                .reindex(base_index)
                .reset_index()
    )
    summary["dp_order"] = summary["dp_epsilon"].map(DP_ORDER_MAP)
    summary["mechanism"] = mechanism
    summary["count"] = summary["count"].fillna(0).astype(int)
    summary = summary[
        ["mechanism", "dp_epsilon", "dp_order", "mean", "std", "count"]
    ]
    return summary

def save_combo_plot(summary: pd.DataFrame, title: str, out_img: Path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    valid = summary[summary["mean"].notna()].copy()

    if valid.empty:
        ax.set_title(PLOT_TITLE)
        ax.set_xlabel("DP-query epsilon")
        ax.set_ylabel("Macro F1")
        ax.set_xlim(-0.5, len(DP_ORDER) - 0.5)
        ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
        ax.set_xticks(range(len(DP_ORDER)))
        ax.set_xticklabels(DP_ORDER)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.text(0.5, 0.5, "No report available", ha="center", va="center", transform=ax.transAxes)
        plt.tight_layout()
        plt.savefig(out_img, dpi=200)
        plt.close()
        return

    x = valid["dp_epsilon"].astype(float)
    y = valid["mean"]
    y_std = valid["std"].fillna(0.0).to_numpy(dtype=float)
    ax.plot(x, y, marker="o", linewidth=2.4, color=MEAN_LINE_COLOR)

    # Mean ± std band (clipped to [0, 1]) to visualize variability.
    lo = np.clip(y.to_numpy(dtype=float) - y_std, 0.0, 1.0)
    hi = np.clip(y.to_numpy(dtype=float) + y_std, 0.0, 1.0)
    ax.fill_between(x, lo, hi, color="#CC79A7", alpha=0.18, linewidth=0)

    for xi, yi in zip(x, y):
        # Move labels clearly above the line (no background box).
        y_text = min(Y_AXIS_MAX - 0.01, yi + 0.06)
        ax.text(
            xi,
            y_text,
            f"{yi:.2f}",
            ha="center",
            va="bottom",
            fontsize=max(12, BASE_FONT_SIZE - 1),
            color=TEXT_COLOR,
            zorder=8,
        )

    ax.set_title(PLOT_TITLE)
    ax.set_xlabel("DP-query epsilon")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_img, dpi=200)
    plt.close()


def save_datasetwise_plot(combo_df: pd.DataFrame, title: str, out_img: Path) -> None:
    """
    Plot one line per dataset (plus the global mean) so it is easy to see
    which datasets drive counterintuitive trends.
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.8))

    # Ensure numeric eps and a stable ordering.
    eps_order = [float(x) for x in DP_ORDER]
    combo_df = combo_df.copy()
    combo_df["dp_epsilon_f"] = combo_df["dp_epsilon"].astype(float)
    combo_df = combo_df[combo_df["dp_epsilon_f"].isin(eps_order)]

    if combo_df.empty:
        ax.set_title(PLOT_TITLE)
        ax.set_xlabel("DP-query epsilon")
        ax.set_ylabel("Macro F1")
        ax.set_xlim(min(eps_order), max(eps_order))
        ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.text(0.5, 0.5, "No report available", ha="center", va="center", transform=ax.transAxes)
        plt.tight_layout()
        plt.savefig(out_img, dpi=200)
        plt.close()
        return

    # Per-dataset lines
    for ds, g in combo_df.groupby("dataset"):
        g = g.sort_values("dp_epsilon_f")
        ax.plot(
            g["dp_epsilon_f"].to_numpy(),
            g["f1_macro"].to_numpy(),
            marker="o",
            linewidth=1.6,
            alpha=0.55,
            label=ds,
        )

    # Global mean line (thicker)
    grouped = combo_df.groupby("dp_epsilon_f")["f1_macro"]
    mean_df = grouped.mean().reindex(eps_order).reset_index().rename(columns={"dp_epsilon_f": "eps", "f1_macro": "mean"})
    std_df = grouped.std(ddof=1).reindex(eps_order).reset_index().rename(columns={"dp_epsilon_f": "eps", "f1_macro": "std"})
    ax.plot(
        mean_df["eps"].to_numpy(),
        mean_df["mean"].to_numpy(),
        color=MEAN_LINE_COLOR,
        linewidth=3.0,
        marker="o",
        label="mean",
        zorder=5,
    )
    # Mean ± std band (clipped).
    std_vals = std_df["std"].fillna(0.0).to_numpy(dtype=float)
    lo = np.clip(mean_df["mean"].to_numpy(dtype=float) - std_vals, 0.0, 1.0)
    hi = np.clip(mean_df["mean"].to_numpy(dtype=float) + std_vals, 0.0, 1.0)
    ax.fill_between(mean_df["eps"].to_numpy(), lo, hi, color="#CC79A7", alpha=0.18, linewidth=0, zorder=4)

    ax.set_title(PLOT_TITLE)
    ax.set_xlabel("DP-query epsilon")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
    ax.set_xticks(eps_order)
    ax.set_xticklabels([str(x) for x in DP_ORDER])
    ax.grid(True, linestyle="--", alpha=0.35)

    # Legend outside to avoid clutter.
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        ncol=1,
    )

    plt.tight_layout()
    plt.savefig(out_img, dpi=200, bbox_inches="tight")
    plt.close()

# LOAD
rows = []

if not REPORTS_DIR.exists():
    raise FileNotFoundError(f"Cartella non trovata: {REPORTS_DIR.resolve()}")

print(f"[info] using script: {Path(__file__).resolve()}")
print(f"[info] REPORTS_DIR={REPORTS_DIR}")
print(f"[info] REPORT_TARGET={REPORT_TARGET} | kind={TARGET_KIND} | guiding_bb={TARGET_GUIDING_BB} | percentile={TARGET_PERCENTILE}")
print(f"[info] SELECT_DATASETS={SELECT_DATASETS}")

for dataset_dir in REPORTS_DIR.iterdir():
    if not dataset_dir.is_dir():
        continue

    dataset_name = dataset_dir.name

    if SELECT_DATASETS is not None and dataset_name not in SELECT_DATASETS:
        continue

    for path in dataset_dir.glob("report_*.json"):
        rec = parse_one_report(path)
        if rec is not None:
            rows.append(rec)

    # Optionally ingest legacy dpquery RF models and evaluate them directly.
    # This is useful when the JSON reports are missing, but it should not mix with
    # report-based targets like `synth_on_real_test` / `original_on_real_test`.
    if REPORT_TARGET == "model_eval_on_real_test":
        model_dir = MODEL_DIR_WISE / dataset_name
        if model_dir.exists():
            for mp in model_dir.glob("rf_dpquery_*.sav"):
                rec = parse_one_model(mp)
                if rec is not None:
                    rows.append(rec)

if not rows:
    raise ValueError("No report found for the selected datasets.")

df = pd.DataFrame(rows)

print("\n[info] datasets found in the loaded reports:")
print(sorted(df["dataset"].unique()))

if "report_target" in df.columns:
    print("[info] report_target presenti:", sorted(df["report_target"].dropna().unique().tolist()))

# FILTER BASE
df = df[
    (df["kind"] == TARGET_KIND) &
    (df["guiding_bb"] == TARGET_GUIDING_BB) &
    (df["percentile"] == TARGET_PERCENTILE) &
    (df["mechanism"].isin(TARGET_MECHANISMS)) &
    (df["dp_epsilon"].isin(DP_ORDER))
].copy()

if df.empty:
    raise ValueError(
        "The filter is too restrictive: no data found.\n"
        f"dataset={SELECT_DATASETS}, kind={TARGET_KIND}, "
        f"guiding_bb={TARGET_GUIDING_BB}, percentile={TARGET_PERCENTILE}, "
        f"mechanisms={TARGET_MECHANISMS}"
    )

print("\n[info] dataset usati dopo il filtro base:")
print(sorted(df["dataset"].unique()))

for case in COMBO_CASES:
    for mechanism in TARGET_MECHANISMS:
        print(f"\n=== Plot: {case['name']} / {mechanism} ===")
        combo_df = df[
            (df["noise_variant"].isin(case["variants"])) &
            (df["mechanism"] == mechanism)
        ].copy()

        if combo_df.empty:
            print("[warn] no report found for this combination.")
        else:
            print("[info] dataset disponibili:")
            print(sorted(combo_df["dataset"].unique()))

        summary = build_combo_summary(df, mechanism, case["variants"])
        out_dir = (PROJECT_ROOT / "Reports-eval" / "dpquery_sweep" / "dataset_wise").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"f1_macro_dpquery_{REPORT_TARGET}_{case['name']}_{mechanism}.csv"
        out_img = out_dir / f"f1_macro_dpquery_{REPORT_TARGET}_{case['name']}_{mechanism}.png"
        out_img_ds = out_dir / f"f1_macro_dpquery_{REPORT_TARGET}_{case['name']}_{mechanism}_per_dataset.png"
        out_pivot = out_dir / f"f1_macro_dpquery_{REPORT_TARGET}_{case['name']}_{mechanism}_per_dataset.csv"
        summary.to_csv(out_csv, index=False)

        print("[info] summary:")
        print(summary)

        title = (
            f"Macro F1 vs DP-query epsilon ({REPORT_TARGET}, "
            f"{case['label']}, mechanism={mechanism})"
        )
        save_combo_plot(summary, title, out_img)
        # Dataset-wise plot and pivot table help interpret non-monotonic behavior.
        save_datasetwise_plot(combo_df, title, out_img_ds)
        pivot = (
            combo_df.pivot_table(index="dataset", columns="dp_epsilon", values="f1_macro", aggfunc="mean")
            .reindex(columns=DP_ORDER)
        )
        pivot.to_csv(out_pivot)

        print(f"[OK] CSV saved to:  {out_csv}")
        print(f"[OK] Plot saved to: {out_img}")
        print(f"[OK] Plot per-dataset saved to: {out_img_ds}")
        print(f"[OK] CSV per-dataset saved to:  {out_pivot}")
