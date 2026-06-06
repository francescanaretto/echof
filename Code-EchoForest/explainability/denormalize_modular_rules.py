#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
De-standardize (and partially decode) modular rule exports into the original feature space.

"""

from __future__ import annotations

import csv
import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULESEL_DIR = PROJECT_ROOT / "Reports-eval" / "supporting-rule-selection"
RULESEL_DIR = Path(os.environ.get("RULESEL_DIR", str(DEFAULT_RULESEL_DIR))).resolve()

DATASETS = [
    d.strip()
    for d in os.environ.get(
        "DATASETS",
        "activity,adult,electricity,landsat,landsat2,landsat-multi,pol,spotify,spotify-r,wave-binary,wave-multi",
    ).split(",")
    if d.strip()
]

# Modular directory:
# - prefer the newer layout Reports-eval/modular if present
# - otherwise fall back to Reports-eval/supporting-rule-selection/modular
DEFAULT_MODULAR_DIR = PROJECT_ROOT / "Reports-eval" / "modular"
FALLBACK_MODULAR_DIR = RULESEL_DIR / "modular"
_mod_env = os.environ.get("MODULAR_DIR", "").strip()
if _mod_env:
    MODULAR_DIR = Path(_mod_env)
    MODULAR_DIR = (MODULAR_DIR if MODULAR_DIR.is_absolute() else (PROJECT_ROOT / MODULAR_DIR)).resolve()
else:
    MODULAR_DIR = (DEFAULT_MODULAR_DIR if DEFAULT_MODULAR_DIR.exists() else FALLBACK_MODULAR_DIR).resolve()

DOMAIN_DIR = PROJECT_ROOT / "Data-synthetic"
ORIG_DIR = PROJECT_ROOT / "Data-original"
BB_DIR = PROJECT_ROOT / "Model-original"


RULE_ATOM_RE = re.compile(
    r"(?P<feat>.+?)\s*(?P<op><=|>)\s*(?P<thr>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _load_scaler(dataset: str):
    """
    Load StandardScaler saved by generation scripts.
    Prefer joblib if available, otherwise try pickle.
    """
    p = BB_DIR / dataset / f"scaler_{dataset}.joblib"
    if not p.exists():
        raise FileNotFoundError(f"Scaler not found: {p}")

    try:
        import joblib  # type: ignore

        return joblib.load(p)
    except Exception:
        with p.open("rb") as f:
            return pickle.load(f)


def _feature_order(dataset: str) -> List[str]:
    """
    Reconstruct feature order from the original train CSV (the scaler is fit on it).
    """
    p = ORIG_DIR / dataset / f"train_set_{dataset}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Original train set not found: {p}")

    # Avoid pandas dependency: read header only.
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip()
    cols = [c.strip() for c in header.split(",") if c.strip()]
    cols = [c for c in cols if not c.startswith("Unnamed")]
    return cols


@dataclass
class DomainCol:
    name: str
    col_type: str
    values: Optional[List[str]] = None
    min: Optional[float] = None
    max: Optional[float] = None


def _load_domain(dataset: str) -> Dict[str, DomainCol]:
    p = DOMAIN_DIR / f"domain_{dataset}.json"
    if not p.exists():
        return {}
    d = _load_json(p)
    cols = d.get("columns", {}) or {}
    out: Dict[str, DomainCol] = {}
    for name, meta in cols.items():
        if not isinstance(meta, dict):
            continue
        out[name] = DomainCol(
            name=name,
            col_type=str(meta.get("type", "continuous")),
            values=meta.get("values") if isinstance(meta.get("values"), list) else None,
            min=float(meta["min"]) if "min" in meta else None,
            max=float(meta["max"]) if "max" in meta else None,
        )
    return out


def _is_categorical(dom: Dict[str, DomainCol], feat: str) -> bool:
    dc = dom.get(feat)
    if dc is None:
        return False
    t = (dc.col_type or "").lower()
    return t in {"categorical", "binary", "discrete", "ordinal"}


def _decode_category(domcol: DomainCol, thr_orig: float) -> Optional[str]:
    """
    Heuristic: pick the closest known value (when values are numeric-like).
    """
    if not domcol.values:
        return None
    candidates: List[Tuple[float, str]] = []
    for v in domcol.values:
        try:
            vf = float(str(v))
        except Exception:
            continue
        candidates.append((vf, str(v)))
    if not candidates:
        return None
    best = min(candidates, key=lambda x: abs(x[0] - thr_orig))
    return best[1]


def _invert_threshold(scaler, feat_to_idx: Dict[str, int], feat: str, thr_z: float) -> float:
    j = feat_to_idx.get(feat)
    if j is None:
        # unknown feature name: return unchanged
        return float(thr_z)
    mean = float(scaler.mean_[j])
    scale = float(scaler.scale_[j])
    return float(thr_z) * scale + mean


def _rewrite_rendered_rule(
    rendered_rule: str,
    scaler,
    feat_to_idx: Dict[str, int],
    domain: Dict[str, DomainCol],
) -> str:
    """
    Parse a rendered rule like:
      "f1 > -0.12 AND f2 <= 0.83"
    and rewrite thresholds into original space.
    """
    parts = [p.strip() for p in (rendered_rule or "").split("AND")]
    out_parts: List[str] = []
    for atom in parts:
        atom = atom.strip()
        if not atom:
            continue
        m = RULE_ATOM_RE.match(atom)
        if not m:
            out_parts.append(atom)
            continue
        feat = m.group("feat").strip()
        op = m.group("op").strip()
        thr_z = float(m.group("thr"))
        thr_orig = _invert_threshold(scaler, feat_to_idx, feat, thr_z)

        if _is_categorical(domain, feat):
            dc = domain.get(feat)
            label = _decode_category(dc, thr_orig) if dc else None
            if label is not None:
                out_parts.append(f"{feat} {op} {thr_orig:.6f} (≈ {label})")
            else:
                out_parts.append(f"{feat} {op} {thr_orig:.6f}")
        else:
            out_parts.append(f"{feat} {op} {thr_orig:.6f}")
    return " AND ".join(out_parts) if out_parts else rendered_rule


def _denorm_rules_file(dataset: str, in_path: Path, out_path: Path) -> None:
    scaler = _load_scaler(dataset)
    features = _feature_order(dataset)
    feat_to_idx = {f: i for i, f in enumerate(features)}
    domain = _load_domain(dataset)

    with in_path.open("r", encoding="utf-8", errors="ignore", newline="") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "rendered_rule" not in fieldnames:
        # nothing to do
        out_path.write_text(in_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        return

    if "rendered_rule_orig" not in fieldnames:
        fieldnames.append("rendered_rule_orig")

    for r in rows:
        rr = r.get("rendered_rule", "") or ""
        r["rendered_rule_orig"] = _rewrite_rendered_rule(rr, scaler, feat_to_idx, domain)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _denorm_feature_view(dataset: str, in_path: Path, out_path: Path) -> None:
    """
    feature_view.csv already carries lb/ub; we invert those bounds.
    """
    scaler = _load_scaler(dataset)
    features = _feature_order(dataset)
    feat_to_idx = {f: i for i, f in enumerate(features)}
    domain = _load_domain(dataset)

    with in_path.open("r", encoding="utf-8", errors="ignore", newline="") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    for extra in ["lb_orig", "ub_orig", "interval_orig", "decoded_value_hint"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    for r in rows:
        feat = r.get("feature_name", "") or ""
        lb = r.get("lb")
        ub = r.get("ub")
        lb_o = None
        ub_o = None
        if lb not in (None, ""):
            try:
                lb_o = _invert_threshold(scaler, feat_to_idx, feat, float(lb))
            except Exception:
                lb_o = None
        if ub not in (None, ""):
            try:
                ub_o = _invert_threshold(scaler, feat_to_idx, feat, float(ub))
            except Exception:
                ub_o = None

        r["lb_orig"] = "" if lb_o is None else f"{lb_o:.6f}"
        r["ub_orig"] = "" if ub_o is None else f"{ub_o:.6f}"
        if lb_o is not None and ub_o is not None:
            r["interval_orig"] = f"({lb_o:.6f}, {ub_o:.6f}]"
        elif lb_o is not None:
            r["interval_orig"] = f"({lb_o:.6f}, +inf)"
        elif ub_o is not None:
            r["interval_orig"] = f"(-inf, {ub_o:.6f}]"
        else:
            r["interval_orig"] = r.get("interval", "")

        hint = ""
        if _is_categorical(domain, feat):
            dc = domain.get(feat)
            if dc is not None:
                # hint by closest value (based on midpoint if interval)
                probe = None
                if lb_o is not None and ub_o is not None:
                    probe = 0.5 * (lb_o + ub_o)
                elif lb_o is not None:
                    probe = lb_o
                elif ub_o is not None:
                    probe = ub_o
                if probe is not None:
                    dec = _decode_category(dc, probe)
                    if dec is not None:
                        hint = dec
        r["decoded_value_hint"] = hint

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print(f"[info] RULESEL_DIR={RULESEL_DIR}")
    print(f"[info] MODULAR_DIR={MODULAR_DIR}")
    print(f"[info] DATASETS={DATASETS}")

    for ds in DATASETS:
        ds_dir = MODULAR_DIR / ds
        if not ds_dir.exists():
            print(f"[warn] missing dataset dir: {ds_dir}")
            continue

        top1_in = ds_dir / "top1_rules.csv"
        cov_in = ds_dir / "coverage_rules.csv"
        feat_in = ds_dir / "feature_view.csv"

        if top1_in.exists():
            _denorm_rules_file(ds, top1_in, ds_dir / "top1_rules_orig.csv")
            print(f"[OK] wrote {ds_dir / 'top1_rules_orig.csv'}")
        if cov_in.exists():
            _denorm_rules_file(ds, cov_in, ds_dir / "coverage_rules_orig.csv")
            print(f"[OK] wrote {ds_dir / 'coverage_rules_orig.csv'}")
        if feat_in.exists():
            _denorm_feature_view(ds, feat_in, ds_dir / "feature_view_orig.csv")
            print(f"[OK] wrote {ds_dir / 'feature_view_orig.csv'}")


if __name__ == "__main__":
    main()
