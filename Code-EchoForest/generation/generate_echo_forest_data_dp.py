#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wise generator with:
- dataset-level timing
- class-balance control
- adaptive distribution mixing
- a privacy-aware query interface with output perturbation on black-box probabilities

Methodological note:
- this script does NOT provide formal differential privacy for the black-box training set
- it introduces a noisy query interface that limits the precision of black-box outputs
  and tracks the query cost
"""

import typing
try:
    from typing_extensions import Self as _Self
    typing.Self = _Self  # type: ignore[attr-defined]
except Exception:
    pass

import warnings
warnings.filterwarnings("ignore")

import argparse
import sys
import os
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make the Code root importable regardless of the current working directory.
CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

# Configuration
def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}

def _parse_list_env(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    items = [x.strip() for x in raw.split(",")]
    return [x for x in items if x]

# DATASETS = ["adult", "activity", "pol", "spotify", "spotify-r", "landsat", "landsat2", "electricity"]
DATASETS = _parse_list_env("DATASETS", ["wave-multi", "landsat-multi"])

DIR_DATA   = (PROJECT_ROOT / "Data-original").resolve()
DIR_MODELS = (PROJECT_ROOT / "Model-original").resolve()
DIR_SYNTH  = (PROJECT_ROOT / "Data-synthetic" / "wise").resolve()

# Wise generator
N_SYNTH          = 800000
MAX_TRIALS       = 100000
BATCH_SIZE_CAND  = 1024
KIND             = os.environ.get("KIND", "logit")      # "entropy" | "margin" | "kappa" | "logit"
PERCENTILE       = int(os.environ.get("PERCENTILE", "25"))
GUIDING_BB       = os.environ.get("GUIDING_BB", "nn")   # "nn" | "rf"
RANDOM_SEED      = int(os.environ.get("RANDOM_SEED", "42"))
BATCH_SIZE_LABEL = 4096

# Balance controller
BALANCE_MODE         = "ratios"   # "uniform" or "ratios"
TARGET_CLASS_RATIOS  = None
BALANCE_TOLERANCE    = 4000
EARLY_STALL_PATIENCE = 50

# Adaptive distribution mix
ADAPTIVE_MIX      = True
ADAPT_CLASS_AWARE = True
DIRICHLET_ALPHA   = 1.0
MIX_UPDATE_EVERY  = 10

# --- exploration / exploitation ---
CHOICE_POLICY        = "eps_greedy"   # "eps_greedy" | "adaptive_mix" | "uniform"
EPS_START            = 0.25
EPS_MIN              = 0.02
EPS_DECAY            = 0.999
WARMUP_ITERS         = 50

EXP_WINDOW           = 300
EXP_TOP_K            = 4
CLASS_AWARE_EXPLOIT  = True

# QUERY INTERFACE Configuration
USE_NOISY_QUERY_INTERFACE = _parse_bool_env("USE_NOISY_QUERY_INTERFACE", True)

QUERY_NOISE_MECH   = os.environ.get("QUERY_NOISE_MECH", "laplace")  # "laplace" | "gaussian"
QUERY_EPSILON      = float(os.environ.get("QUERY_EPSILON", "5.0"))
QUERY_DELTA        = float(os.environ.get("QUERY_DELTA", "1e-5"))
QUERY_SENSITIVITY  = float(os.environ.get("QUERY_SENSITIVITY", "1.0"))

MAX_QUERY_CALLS    = None  # keep None by default; can be extended to env later
MAX_QUERY_POINTS   = None  # keep None by default; can be extended to env later

NOISE_ON_LABELING  = _parse_bool_env("NOISE_ON_LABELING", True)

# Robust fallback
MIN_SELECTED_PER_BATCH_FRACTION = 0.02   # minimum fallback when idx is empty
STALL_PERCENTILE_STEP           = 5
MAX_PERCENTILE_RELAX            = 95


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate DP-query synthetic data for EchoForest."
    )
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to process.")
    parser.add_argument("--n-synth", type=int, default=None, help="Number of synthetic records to generate.")
    parser.add_argument(
        "--mode",
        choices=["entropy25", "entropy50", "margin", "kappa", "logit"],
        default=None,
        help="Selection mode. entropy25 and entropy50 set both kind and percentile.",
    )
    parser.add_argument("--guiding-bb", choices=["nn", "rf"], default=None, help="Black-box type.")
    parser.add_argument("--epsilon", type=float, default=None, help="DP-query epsilon.")
    parser.add_argument("--noise", choices=["laplace", "gaussian"], default=None, help="Noise mechanism.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def apply_cli_overrides(args):
    global DATASETS, N_SYNTH, KIND, PERCENTILE, GUIDING_BB, QUERY_EPSILON, QUERY_NOISE_MECH, RANDOM_SEED
    if args.datasets:
        DATASETS = args.datasets
    if args.n_synth is not None:
        N_SYNTH = args.n_synth
    if args.guiding_bb is not None:
        GUIDING_BB = args.guiding_bb
    if args.epsilon is not None:
        QUERY_EPSILON = args.epsilon
    if args.noise is not None:
        QUERY_NOISE_MECH = args.noise
    if args.seed is not None:
        RANDOM_SEED = args.seed
    if args.mode is not None:
        if args.mode == "entropy25":
            KIND = "entropy"
            PERCENTILE = 25
        elif args.mode == "entropy50":
            KIND = "entropy"
            PERCENTILE = 50
        else:
            KIND = args.mode


# Helper functions
def load_train_df(dataset: str) -> pd.DataFrame:
    p = DIR_DATA / dataset / f"train_set_{dataset}.csv"
    df = pd.read_csv(p)
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]
    return df


def load_rf_model(dataset: str):
    with open(DIR_MODELS / dataset / f"rf_{dataset}.sav", "rb") as f:
        return pickle.load(f)


from loaders import load_trained_nn


def get_expected_in_features(model: nn.Module) -> int:
    first_linear = None
    if hasattr(model, "net") and isinstance(model.net, nn.Sequential):
        for m in model.net:
            if isinstance(m, nn.Linear):
                first_linear = m
                break
    if first_linear is None and hasattr(model, "lin1"):
        first_linear = model.lin1
    if first_linear is None:
        raise RuntimeError("Could not find the first Linear layer in the network.")
    return int(first_linear.in_features)


class QueryInterface:
    """
    Wrapper for querying the black-box through output perturbation.
    Tracks the number of queries and queried points.
    """

    def __init__(
        self,
        bb,
        rng: np.random.Generator,
        use_noisy_interface: bool = True,
        noise_mech: str = "laplace",
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
        max_query_calls: int | None = None,
        max_query_points: int | None = None,
    ):
        self.bb = bb
        self.rng = rng
        self.use_noisy_interface = use_noisy_interface
        self.noise_mech = noise_mech
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.sensitivity = float(sensitivity)
        self.max_query_calls = max_query_calls
        self.max_query_points = max_query_points

        self.query_calls = 0
        self.query_points = 0

        self.query_calls_per_phase = {
            "selection": 0,
            "labeling": 0,
            "other": 0,
        }
        self.query_points_per_phase = {
            "selection": 0,
            "labeling": 0,
            "other": 0,
        }

    def _check_budget(self, n_points: int):
        if self.max_query_calls is not None and self.query_calls + 1 > self.max_query_calls:
            raise RuntimeError(f"Exceeded MAX_QUERY_CALLS={self.max_query_calls}")
        if self.max_query_points is not None and self.query_points + n_points > self.max_query_points:
            raise RuntimeError(f"Exceeded MAX_QUERY_POINTS={self.max_query_points}")

    def _raw_predict_proba(self, X: np.ndarray) -> np.ndarray:
        bb = self.bb

        if hasattr(bb, "predict_proba"):
            proba = bb.predict_proba(X)
            return np.asarray(proba, dtype=np.float32)

        if isinstance(bb, nn.Module):
            bb.eval()
            with torch.no_grad():
                xt = torch.tensor(X, dtype=torch.float32)
                logits = bb(xt)
                if logits.ndim == 2 and logits.shape[1] > 1:
                    proba = torch.softmax(logits, dim=1).cpu().numpy()
                else:
                    p1 = torch.sigmoid(logits.flatten()).cpu().numpy()
                    proba = np.stack([1 - p1, p1], axis=1)
            return np.asarray(proba, dtype=np.float32)

        raise RuntimeError("Black-box type not recognized (expected sklearn model or nn.Module).")

    def _add_noise(self, proba: np.ndarray) -> np.ndarray:
        if not self.use_noisy_interface:
            return proba

        if self.epsilon <= 0:
            raise ValueError(f"QUERY_EPSILON must be > 0, got: {self.epsilon}")

        if self.noise_mech == "laplace":
            scale = self.sensitivity / self.epsilon
            noise = self.rng.laplace(loc=0.0, scale=scale, size=proba.shape)
        elif self.noise_mech == "gaussian":
            sigma = np.sqrt(2.0 * np.log(1.25 / self.delta)) * self.sensitivity / self.epsilon
            noise = self.rng.normal(loc=0.0, scale=sigma, size=proba.shape)
        else:
            raise ValueError(f"Unknown noise_mech: {self.noise_mech}")

        noisy = proba + noise
        noisy = np.clip(noisy, 0.0, None)

        row_sums = noisy.sum(axis=1, keepdims=True)
        zero_mask = (row_sums.squeeze(axis=1) <= 1e-12)

        if np.any(zero_mask):
            noisy[zero_mask] = 1.0 / noisy.shape[1]
            row_sums = noisy.sum(axis=1, keepdims=True)

        noisy = noisy / row_sums
        return noisy.astype(np.float32)

    def predict_proba(self, X: np.ndarray, phase: str = "other") -> np.ndarray:
        n_points = int(X.shape[0])
        self._check_budget(n_points)

        self.query_calls += 1
        self.query_points += n_points

        if phase not in self.query_calls_per_phase:
            phase = "other"

        self.query_calls_per_phase[phase] += 1
        self.query_points_per_phase[phase] += n_points

        proba = self._raw_predict_proba(X)
        proba = self._add_noise(proba)
        return proba

    def predict_label(self, X: np.ndarray, phase: str = "other") -> np.ndarray:
        proba = self.predict_proba(X, phase=phase)
        return np.argmax(proba, axis=1)

    def predict_label_clean(self, X: np.ndarray) -> np.ndarray:
        proba = self._raw_predict_proba(X)
        return np.argmax(proba, axis=1)

    def summary(self) -> dict:
        return {
            "use_noisy_interface": self.use_noisy_interface,
            "noise_mech": self.noise_mech,
            "query_epsilon": self.epsilon,
            "query_delta": self.delta,
            "query_sensitivity": self.sensitivity,
            "max_query_calls": self.max_query_calls,
            "max_query_points": self.max_query_points,
            "query_calls_total": self.query_calls,
            "query_points_total": self.query_points,
            "query_calls_selection": self.query_calls_per_phase["selection"],
            "query_points_selection": self.query_points_per_phase["selection"],
            "query_calls_labeling": self.query_calls_per_phase["labeling"],
            "query_points_labeling": self.query_points_per_phase["labeling"],
            "query_calls_other": self.query_calls_per_phase["other"],
            "query_points_other": self.query_points_per_phase["other"],
        }


def label_with_query_interface(
    query_interface: QueryInterface,
    X: np.ndarray,
    batch_size: int = 4096,
    noisy_labeling: bool = False,
) -> np.ndarray:
    preds = []
    for start in range(0, X.shape[0], batch_size):
        chunk = X[start:start + batch_size]
        if noisy_labeling:
            pred = query_interface.predict_label(chunk, phase="labeling")
        else:
            pred = query_interface.predict_label_clean(chunk)
        preds.append(pred)
    return np.concatenate(preds, axis=0)


def _compute_target_counts(
    n_samples: int,
    n_classs: int,
    balance_mode: str,
    target_ratios: list[float] | None,
) -> np.ndarray:
    if balance_mode != "ratios" or target_ratios is None:
        base = n_samples // n_classs
        target = np.array([base] * n_classs, dtype=int)
        target[: (n_samples - base * n_classs)] += 1
        return target

    r = np.array(target_ratios, dtype=float)
    if r.ndim != 1 or r.size != n_classs or not np.isfinite(r).all() or (r <= 0).all():
        print(
            f"[warn] target_ratios are incompatible with n_classs={n_classs} "
            f"(len={0 if target_ratios is None else len(target_ratios)}). Falling back to 'uniform'."
        )
        base = n_samples // n_classs
        target = np.array([base] * n_classs, dtype=int)
        target[: (n_samples - base * n_classs)] += 1
        return target

    r = r / r.sum()
    target = np.floor(r * n_samples).astype(int)
    diff = n_samples - target.sum()
    for k in range(diff):
        target[k % n_classs] += 1
    return target


# Wise generator
class WiseGenerator:
    """
    - Multi-distribution sampler (19 families) with adaptive mixing
    - Uncertainty-based selection (entropy/margin/logit/kappa)
    - On-the-fly class balancing toward a target distribution
    - Black-box querying through a noisy QueryInterface
    """

    NUM_DISTS = 19

    def __init__(
        self,
        bb,
        query_interface: QueryInterface,
        n_features: int,
        n_classs: int,
        dataset_name: str,
        kind: str = "entropy",
        percentile: int = 25,
        random_state: int = 42,
        adaptive_mix: bool = True,
        adapt_class_aware: bool = True,
        dirichlet_alpha: float = 1.0,
        mix_update_every: int = 10,
    ):
        self.bb = bb
        self.query_interface = query_interface
        self.n_features = n_features
        self.n_classs = n_classs
        self.dataset_name = dataset_name
        self.kind = kind
        self.percentile = int(percentile)
        self.rng = np.random.default_rng(random_state)

        self.adaptive_mix = adaptive_mix
        self.adapt_class_aware = adapt_class_aware
        self.dirichlet_alpha = dirichlet_alpha
        self.mix_update_every = mix_update_every

        self.dist_success = np.zeros(self.NUM_DISTS, dtype=float)
        self.dist_trials = np.zeros(self.NUM_DISTS, dtype=float)
        self.dist_success_by_class = np.zeros((self.NUM_DISTS, n_classs), dtype=float)

        self.dist_weights = np.ones(self.NUM_DISTS, dtype=float) / self.NUM_DISTS

        self.hist_iters = []
        self.hist_choice = []
        self.hist_accept = []
        self.hist_counts = []
        self.hist_percentile = []
        self.hist_weights = []
        self.hist_update_step = []
        self.hist_acc_by_class = []

        self.choice_policy = CHOICE_POLICY
        self.eps_start = EPS_START
        self.eps_min = EPS_MIN
        self.eps_decay = EPS_DECAY
        self.warmup_iters = WARMUP_ITERS
        self.exp_window = EXP_WINDOW
        self.exp_top_k = EXP_TOP_K
        self.class_aware_exploit = CLASS_AWARE_EXPLOIT

    # ---------- selection criteria ----------

    def _recent_scores(self, deficits: np.ndarray | None) -> np.ndarray:
        n = len(self.hist_choice)
        if n == 0:
            return np.ones(self.NUM_DISTS, dtype=float)

        start = max(0, n - self.exp_window)
        scores = np.zeros(self.NUM_DISTS, dtype=float)

        if self.class_aware_exploit and deficits is not None:
            if deficits.shape[0] != self.n_classs:
                d = np.zeros(self.n_classs, dtype=float)
            else:
                d = np.maximum(deficits.astype(float), 0.0)

            for t in range(start, n):
                cid = self.hist_choice[t]
                acc_vec = self.hist_acc_by_class[t] if t < len(self.hist_acc_by_class) else None
                if acc_vec is None:
                    scores[cid] += float(self.hist_accept[t])
                else:
                    scores[cid] += float(np.dot(acc_vec, d))
        else:
            for t in range(start, n):
                cid = self.hist_choice[t]
                scores[cid] += float(self.hist_accept[t])

        return scores + 1e-9

    def _choose_distribution(self, iter_idx: int, deficits: np.ndarray | None) -> int:
        if iter_idx <= self.warmup_iters:
            return int(self.rng.integers(0, self.NUM_DISTS))

        if self.choice_policy == "eps_greedy":
            eps = max(self.eps_min, self.eps_start * (self.eps_decay ** (iter_idx - self.warmup_iters)))
            if self.rng.random() < eps:
                return int(self.rng.integers(0, self.NUM_DISTS))

            scores = self._recent_scores(deficits)
            if np.all(scores <= 1e-12):
                if self.adaptive_mix:
                    p = self.dist_weights / self.dist_weights.sum()
                    return int(self.rng.choice(self.NUM_DISTS, p=p))
                return int(self.rng.integers(0, self.NUM_DISTS))

            top_k = min(self.exp_top_k, self.NUM_DISTS)
            top_idx = np.argsort(-scores)[:top_k]
            top_scores = scores[top_idx]
            p = top_scores / top_scores.sum()
            return int(self.rng.choice(top_idx, p=p))

        if self.choice_policy == "adaptive_mix":
            p = self.dist_weights / self.dist_weights.sum()
            return int(self.rng.choice(self.NUM_DISTS, p=p))

        return int(self.rng.integers(0, self.NUM_DISTS))

    def _entropy(self, p: np.ndarray) -> float:
        p = np.clip(p, 1e-12, 1.0)
        if self.n_classs <= 1:
            return 0.0
        return float(-(p * np.log(p)).sum() / np.log(self.n_classs))

    def _topk_fallback_low(self, values: np.ndarray) -> np.ndarray:
        k = max(1, int(len(values) * max(1, self.percentile) / 100.0))
        return np.argsort(values)[:k]

    def _topk_fallback_high(self, values: np.ndarray) -> np.ndarray:
        k = max(1, int(len(values) * max(1, self.percentile) / 100.0))
        return np.argsort(-values)[:k]

    def check_entropy(self, dataset_proba: np.ndarray) -> np.ndarray:
        entropies = np.apply_along_axis(self._entropy, 1, dataset_proba)
        thr = np.percentile(entropies, 100 - self.percentile)
        idx = np.where(entropies >= thr)[0]
        if len(idx) == 0:
            idx = self._topk_fallback_high(entropies)
        return idx

    def check_margin(self, dataset_proba: np.ndarray) -> np.ndarray:
        sorted_proba = -np.sort(-dataset_proba, axis=1)
        if sorted_proba.shape[1] < 2:
            return np.arange(len(dataset_proba))
        margins = sorted_proba[:, 0] - sorted_proba[:, 1]
        thr = np.percentile(margins, self.percentile)
        idx = np.where(margins <= thr)[0]
        if len(idx) == 0:
            idx = self._topk_fallback_low(margins)
        return idx

    def check_logit_margin(self, proba: np.ndarray) -> np.ndarray:
        proba = np.clip(proba, 1e-9, 1.0)
        logits = np.log(proba)
        sorted_logits = -np.sort(-logits, axis=1)
        if sorted_logits.shape[1] < 2:
            return np.arange(len(proba))
        margins = sorted_logits[:, 0] - sorted_logits[:, 1]
        thr = np.percentile(margins, self.percentile)
        idx = np.where(margins <= thr)[0]
        if len(idx) == 0:
            idx = self._topk_fallback_low(margins)
        return idx

    def check_kappa(self, proba: np.ndarray) -> np.ndarray:
        sorted_p = -np.sort(-proba, axis=1)
        if sorted_p.shape[1] < 2:
            return np.arange(len(proba))
        pmax, p2 = sorted_p[:, 0], sorted_p[:, 1]
        kappa = (pmax - p2) / (1 - pmax + 1e-9)
        thr = np.percentile(kappa, self.percentile)
        idx = np.where(kappa <= thr)[0]
        if len(idx) == 0:
            idx = self._topk_fallback_low(kappa)
        return idx

    # ---------- distribution sampling ----------

    def _sample_from_choice(self, choice: int, size: int) -> np.ndarray:
        rng = self.rng
        d = self.n_features

        if choice == 0:
            data = rng.normal(0.0, 1.0, size=(size, d))
        elif choice == 1:
            mu = rng.uniform(-0.5, 0.5)
            sigma = rng.uniform(0.1, 1.0)
            data = rng.lognormal(mu, sigma, size=(size, d))
        elif choice == 2:
            shape = rng.uniform(0.5, 5.0)
            scale = rng.uniform(0.5, 3.0)
            data = rng.gamma(shape, scale, size=(size, d))
        elif choice == 3:
            a = rng.uniform(0.5, 5.0)
            b = rng.uniform(0.5, 5.0)
            base = rng.beta(a, b, size=(size, d))
            data = -3.0 + 6.0 * base
        elif choice == 4:
            df = rng.uniform(1.0, 5.0)
            data = rng.standard_t(df=df, size=(size, d))
        elif choice == 5:
            lam = rng.uniform(0.1, 2.0)
            base = rng.exponential(1.0 / lam, size=(size, d))
            sign = rng.choice([-1.0, 1.0], size=(size, d))
            data = base * sign
        elif choice == 6:
            lam = rng.uniform(1.0, 20.0)
            base = rng.poisson(lam, size=(size, d)).astype(float)
            data = base - lam
        elif choice == 7:
            pi = rng.uniform(0.2, 0.8)
            comp = rng.binomial(1, pi, size=size)
            mu1 = rng.uniform(-2.0, 2.0)
            mu2 = rng.uniform(-5.0, 5.0)
            s1 = rng.uniform(0.4, 2.5)
            s2 = rng.uniform(0.4, 3.0)
            g1 = rng.normal(mu1, s1, size=(size, d))
            g2 = rng.normal(mu2, s2, size=(size, d))
            data = np.where(comp[:, None] == 1, g1, g2)
        elif choice == 8:
            p = rng.uniform(0.1, 0.9)
            base = rng.binomial(1, p, size=(size, d))
            data = 2 * base - 1
        elif choice == 9:
            K = rng.integers(3, 6)
            pis = rng.dirichlet(np.ones(K))
            comps = rng.choice(K, size=size, p=pis)
            data = np.zeros((size, d), dtype=np.float32)
            for k in range(K):
                idx = np.where(comps == k)[0]
                if len(idx) == 0:
                    continue
                mu = rng.uniform(-4, 4)
                sigma = rng.uniform(0.3, 3.0)
                data[idx] = rng.normal(mu, sigma, size=(len(idx), d))
        elif choice == 10:
            shape = rng.uniform(1.0, 5.0)
            data = rng.pareto(shape, size=(size, d))
            sign = rng.choice([-1.0, 1.0], size=(size, d))
            data = data * sign
        elif choice == 11:
            a = rng.uniform(0.5, 3.0)
            data = rng.weibull(a, size=(size, d))
        elif choice == 12:
            loc = rng.uniform(-2.0, 2.0)
            scale = rng.uniform(0.2, 2.0)
            data = rng.laplace(loc=loc, scale=scale, size=(size, d))
        elif choice == 13:
            data = rng.standard_cauchy(size=(size, d))
            data = np.clip(data, -20, 20)
        elif choice == 14:
            left = rng.uniform(-4, -1)
            mode = rng.uniform(left, left + 5)
            right = rng.uniform(mode, mode + 5)
            data = rng.triangular(left, mode, right, size=(size, d))
        elif choice == 15:
            loc = rng.uniform(-2, 2)
            scale = rng.uniform(0.3, 2.0)
            u = rng.uniform(0, 1, size=(size, d))
            data = loc + scale * np.log(u / (1 - u))
        elif choice == 16:
            A = rng.normal(size=(d, d))
            cov = A @ A.T + np.eye(d) * 1e-3
            mean = rng.normal(0, 1, size=d)
            data = rng.multivariate_normal(mean, cov, size=size)
        elif choice == 17:
            K = rng.integers(2, 5)
            pis = rng.dirichlet(np.ones(K))
            comps = rng.choice(K, size=size, p=pis)
            data = np.zeros((size, d), dtype=np.float32)
            for k in range(K):
                idx = np.where(comps == k)[0]
                if len(idx) == 0:
                    continue
                A = rng.normal(size=(d, d))
                cov = A @ A.T + np.eye(d) * 1e-3
                mean = rng.normal(0, 2, size=d)
                data[idx] = rng.multivariate_normal(mean, cov, size=len(idx))
        else:
            base = rng.binomial(1, 0.5, size=(size, d)).astype(float)
            n_pairs = max(1, d // 3)
            for _ in range(n_pairs):
                i = rng.integers(0, d)
                j = rng.integers(0, d)
                if i == j:
                    continue
                mask = rng.binomial(1, 0.7, size=size).astype(bool)
                base[mask, j] = base[mask, i]
            data = 2 * base - 1

        return data.astype(np.float32)

    def _sample_distribution_block(self, size: int, iter_idx: int, deficits: np.ndarray | None) -> tuple[np.ndarray, int]:
        choice = self._choose_distribution(iter_idx=iter_idx, deficits=deficits)
        data = self._sample_from_choice(choice, size)
        return data, choice

    def _update_mix_weights(self, deficits: np.ndarray):
        if not self.adaptive_mix:
            return

        alpha = self.dirichlet_alpha

        if self.adapt_class_aware and deficits is not None:
            if deficits.ndim != 1 or deficits.shape[0] != self.n_classs:
                print(
                    f"[warn] _update_mix_weights: deficits shape {getattr(deficits, 'shape', None)} "
                    f"!= n_classs ({self.n_classs}). Ignoring deficits and using a zero vector."
                )
                deficit_w = np.zeros(self.n_classs, dtype=float)
            else:
                deficit_w = np.maximum(deficits.astype(float), 0.0)

            if deficit_w.sum() > 0:
                weighted_succ = (self.dist_success_by_class * deficit_w[None, :]).sum(axis=1)
            else:
                weighted_succ = self.dist_success.copy()

            post = alpha + weighted_succ
        else:
            post = alpha + self.dist_success

        s = post.sum()
        if s <= 0:
            self.dist_weights = np.ones_like(post) / len(post)
        else:
            self.dist_weights = post / s

    def _relax_selection(self):
        old = self.percentile
        self.percentile = min(MAX_PERCENTILE_RELAX, self.percentile + STALL_PERCENTILE_STEP)
        if self.percentile != old:
            print(f"[warn] stall detected: relaxing percentile from {old} to {self.percentile} (kind={self.kind})")

    # ---------- main generation loop ----------

    def generate(
        self,
        n_samples: int,
        max_trials: int = 300000,
        batch_size: int = 1024,
        balance_mode: str = "uniform",
        target_ratios: list[float] | None = None,
        balance_tolerance: int = 0,
        early_stall_patience: int = 200,
    ) -> np.ndarray:
        kept_per_class = [[] for _ in range(self.n_classs)]
        counts = np.zeros(self.n_classs, dtype=int)

        target_counts = _compute_target_counts(
            n_samples=n_samples,
            n_classs=self.n_classs,
            balance_mode=balance_mode,
            target_ratios=target_ratios,
        )

        kept_total = 0
        trials = 0
        iters_since_accept = 0

        print(f"[info] wise generation target={n_samples}, kind={self.kind}")
        print(f"[info] target counts: {dict(enumerate(target_counts.tolist()))}")

        while kept_total < n_samples and trials < max_trials:
            trials += 1
            iters_since_accept += 1

            deficits = target_counts - counts

            if deficits.ndim != 1 or deficits.shape[0] != self.n_classs:
                print(
                    f"[warn] generate: deficits shape {getattr(deficits, 'shape', None)} "
                    f"!= n_classs ({self.n_classs}). Forcing a zero vector."
                )
                deficits = np.zeros(self.n_classs, dtype=int)

            cand, choice_id = self._sample_distribution_block(
                batch_size,
                iter_idx=trials,
                deficits=deficits,
            )
            self.dist_trials[choice_id] += 1

            proba = self.query_interface.predict_proba(cand, phase="selection")

            if self.kind == "entropy":
                idx = self.check_entropy(proba)
            elif self.kind == "margin":
                idx = self.check_margin(proba)
            elif self.kind == "kappa":
                idx = self.check_kappa(proba)
            elif self.kind == "logit":
                idx = self.check_logit_margin(proba)
            else:
                idx = np.arange(len(cand))

            if len(idx) == 0:
                k = max(1, int(len(cand) * MIN_SELECTED_PER_BATCH_FRACTION))
                idx = np.arange(min(k, len(cand)))

            yhat = np.argmax(proba, axis=1)

            accepted_idx = []
            acc_by_class = np.zeros(self.n_classs, dtype=int)

            for i in idx:
                y = int(yhat[i])
                if 0 <= y < self.n_classs and counts[y] < (target_counts[y] + balance_tolerance):
                    accepted_idx.append(i)
                    acc_by_class[y] += 1

            if len(accepted_idx) == 0:
                deficit_order = np.argsort(-(target_counts - counts))
                for cls in deficit_order:
                    cls = int(cls)
                    cand_cls = [i for i in idx if int(yhat[i]) == cls]
                    if len(cand_cls) > 0 and counts[cls] < (target_counts[cls] + balance_tolerance):
                        take = min(len(cand_cls), max(1, batch_size // 20))
                        accepted_idx.extend(cand_cls[:take])
                        acc_by_class[cls] += take
                        break

            if accepted_idx:
                for i in accepted_idx:
                    y = int(yhat[i])
                    kept_per_class[y].append(cand[i:i + 1])

                counts += acc_by_class
                kept_total = int(counts.sum())
                iters_since_accept = 0

                self.dist_success[choice_id] += len(accepted_idx)
                for c in range(self.n_classs):
                    self.dist_success_by_class[choice_id, c] += acc_by_class[c]
            else:
                if early_stall_patience and iters_since_accept >= early_stall_patience:
                    self._relax_selection()
                    iters_since_accept = 0

            self.hist_iters.append(trials)
            self.hist_choice.append(choice_id)
            self.hist_accept.append(int(np.sum(acc_by_class)))
            self.hist_counts.append(counts.copy())
            self.hist_percentile.append(self.percentile)
            self.hist_acc_by_class.append(acc_by_class.copy())

            if (trials % self.mix_update_every) == 0:
                self.hist_weights.append(self.dist_weights.copy())
                self.hist_update_step.append(trials)
                self._update_mix_weights(deficits)

            if np.all(counts >= target_counts):
                break

            if trials % 1000 == 0:
                print(f"[info] iter={trials} kept={kept_total}/{n_samples} percentile={self.percentile}")

        out_blocks = []
        for c in range(self.n_classs):
            if len(kept_per_class[c]) == 0:
                continue
            Xc = np.concatenate(kept_per_class[c], axis=0)
            if Xc.shape[0] > target_counts[c]:
                Xc = Xc[:target_counts[c]]
            out_blocks.append(Xc)

        if len(out_blocks) == 0:
            raise RuntimeError("No point selected: criteria remained too restrictive even after relaxation.")

        X_all = np.concatenate(out_blocks, axis=0)

        if X_all.shape[0] == 0:
            raise RuntimeError("Empty generation: no points were accepted.")

        idx_perm = self.rng.permutation(X_all.shape[0])
        X_all = X_all[idx_perm]

        print(f"[info] generated {X_all.shape[0]} points (trials={trials})")
        print(f"[info] final counts: {dict(enumerate(counts.tolist()))}")
        if self.adaptive_mix:
            print(f"[info] final distribution weights: {np.round(self.dist_weights, 3)}")
        return X_all


def save_wise_plots(
    dataset: str,
    gen: WiseGenerator,
    final_counts: np.ndarray,
    target_counts: np.ndarray,
    out_dir: Path,
    name: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(gen.hist_weights):
        W = np.vstack(gen.hist_weights)
        it = np.array(gen.hist_update_step)
        final_w = W[-1]
        top_idx = np.argsort(-final_w)[:6]

        plt.figure(figsize=(9, 5))
        for j in range(W.shape[1]):
            if j in top_idx:
                plt.plot(it, W[:, j], label=f"d{j}", linewidth=2.0)
            else:
                plt.plot(it, W[:, j], alpha=0.25, linewidth=1.0)
        plt.title(f"{dataset}, Weight evolution")
        plt.xlabel("iter")
        plt.ylabel("weight")
        plt.legend(ncol=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"plot_weights_{name}.png", dpi=160)
        plt.close()

    x = np.arange(len(target_counts))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, target_counts, width)
    plt.bar(x + width / 2, final_counts, width)
    plt.title(f"{dataset}, Target vs actual per class")
    plt.xlabel("class")
    plt.ylabel("Num. samples")
    plt.xticks(x)
    plt.legend(["target", "actual"])
    plt.tight_layout()
    plt.savefig(out_dir / f"plot_classs_{name}.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(gen.NUM_DISTS), gen.dist_success)
    plt.title(f"{dataset}, Elements per distribution")
    plt.xlabel("id stat. distribution")
    plt.ylabel("Num. samples")
    plt.tight_layout()
    plt.savefig(out_dir / f"plot_dist_success_{name}.png", dpi=160)
    plt.close()

    if len(gen.hist_accept) > 0:
        a = np.array(gen.hist_accept, dtype=float)
        k = max(1, len(a) // 50)
        kern = np.ones(k) / k
        rolling = np.convolve(a, kern, mode="same")
        plt.figure(figsize=(9, 4))
        plt.plot(gen.hist_iters, rolling)
        plt.title(f"{dataset}, Acceptance rate")
        plt.xlabel("iter")
        plt.ylabel("Num. of accepted per iter")
        plt.tight_layout()
        plt.savefig(out_dir / f"plot_accept_rate_{name}.png", dpi=160)
        plt.close()


def save_history_csv(dataset: str, gen: WiseGenerator, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    counts_mat = np.vstack(gen.hist_counts) if len(gen.hist_counts) else np.zeros((0, gen.n_classs), dtype=int)
    hist = pd.DataFrame({
        "iter": gen.hist_iters,
        "choice_id": gen.hist_choice,
        "accepted": gen.hist_accept,
        "percentile": gen.hist_percentile,
    })
    for c in range(gen.n_classs):
        hist[f"counts_{c}"] = counts_mat[:, c] if counts_mat.shape[0] else []
    hist.to_csv(out_dir / f"wise_history_{name}.csv", index=False)

    if len(gen.hist_weights):
        W = np.vstack(gen.hist_weights)
        dfw = pd.DataFrame(W, columns=[f"d{j}" for j in range(gen.NUM_DISTS)])
        dfw.insert(0, "iter", gen.hist_update_step)
        dfw.to_csv(out_dir / f"wise_weights_{name}.csv", index=False)


def save_distribution_summary(dataset: str, gen: WiseGenerator, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = np.maximum(gen.dist_trials, 1.0)
    df = pd.DataFrame({
        "dist_id": np.arange(gen.NUM_DISTS),
        "success": gen.dist_success,
        "trials": gen.dist_trials,
        "success_per_trial": gen.dist_success / trials,
        "weight_final": gen.dist_weights / np.sum(gen.dist_weights),
    })
    df.to_csv(out_dir / f"wise_dist_summary_{name}.csv", index=False)
    return df


# Main with timing and artifact saving
def main():
    args = parse_args()
    apply_cli_overrides(args)
    DIR_SYNTH.mkdir(parents=True, exist_ok=True)
    timings = []

    print(
        "[info] generate_echo_forest_data_dp.py config:"
        f" datasets={DATASETS}"
        f" | kind={KIND} guiding_bb={GUIDING_BB} percentile={PERCENTILE}"
        f" | use_noisy_interface={USE_NOISY_QUERY_INTERFACE}"
        f" | n_synth={N_SYNTH} eps={QUERY_EPSILON} mech={QUERY_NOISE_MECH} delta={QUERY_DELTA} sens={QUERY_SENSITIVITY}"
        f" | noise_on_labeling={NOISE_ON_LABELING}"
        f" | seed={RANDOM_SEED}"
    )

    for dataset in DATASETS:
        t0_ds = time.perf_counter()
        try:
            print(f"\n=== Dataset: {dataset} | Wise generator ({KIND}, BB={GUIDING_BB}) ===")

            if GUIDING_BB == "rf":
                bb = load_rf_model(dataset)
                n_features = int(getattr(bb, "n_features_in_", len(getattr(bb, "feature_names_in_", []))))
                if n_features == 0:
                    raise RuntimeError("RF model has no n_features_in_ or feature_names_in_.")
                n_classs = len(bb.classs_) if hasattr(bb, "classs_") else 2
            else:
                bb = load_trained_nn(dataset=dataset, base_dir=str(DIR_MODELS))
                n_features = get_expected_in_features(bb)
                bb.eval()
                with torch.no_grad():
                    dummy = torch.zeros(1, n_features)
                    out = bb(dummy)
                n_classs = int(out.shape[1]) if (out.ndim == 2 and out.shape[1] > 1) else 2

            query_interface = QueryInterface(
                bb=bb,
                rng=np.random.default_rng(RANDOM_SEED),
                use_noisy_interface=USE_NOISY_QUERY_INTERFACE,
                noise_mech=QUERY_NOISE_MECH,
                epsilon=QUERY_EPSILON,
                delta=QUERY_DELTA,
                sensitivity=QUERY_SENSITIVITY,
                max_query_calls=MAX_QUERY_CALLS,
                max_query_points=MAX_QUERY_POINTS,
            )

            gen = WiseGenerator(
                bb=bb,
                query_interface=query_interface,
                n_features=n_features,
                n_classs=n_classs,
                dataset_name=dataset,
                kind=KIND,
                percentile=PERCENTILE,
                random_state=RANDOM_SEED,
                adaptive_mix=ADAPTIVE_MIX,
                adapt_class_aware=ADAPT_CLASS_AWARE,
                dirichlet_alpha=DIRICHLET_ALPHA,
                mix_update_every=MIX_UPDATE_EVERY,
            )

            t0_gen = time.perf_counter()
            X_synth = gen.generate(
                n_samples=N_SYNTH,
                max_trials=MAX_TRIALS,
                batch_size=BATCH_SIZE_CAND,
                balance_mode=BALANCE_MODE,
                target_ratios=TARGET_CLASS_RATIOS,
                balance_tolerance=BALANCE_TOLERANCE,
                early_stall_patience=EARLY_STALL_PATIENCE,
            )
            t1_gen = time.perf_counter()

            t0_lab = time.perf_counter()
            y_synth = label_with_query_interface(
                query_interface=query_interface,
                X=X_synth,
                batch_size=BATCH_SIZE_LABEL,
                noisy_labeling=NOISE_ON_LABELING,
            )
            t1_lab = time.perf_counter()

            vals, counts = np.unique(y_synth, return_counts=True)
            print("[info] synthetic class distribution:")
            for v, c in zip(vals, counts):
                print(f"  class {v}: {c} instances")

            t0_save = time.perf_counter()
            out_dir_ds = DIR_SYNTH / dataset
            out_dir_ds.mkdir(parents=True, exist_ok=True)

            name = f"synthetic_dp_{QUERY_EPSILON}_{QUERY_NOISE_MECH}_{NOISE_ON_LABELING}_{dataset}_{KIND}_{GUIDING_BB}_{PERCENTILE}"
            out_path = out_dir_ds / f"{name}.csv"

            df_out = pd.DataFrame(X_synth)
            df_out["label"] = y_synth
            df_out.to_csv(out_path, index=False)

            query_summary = query_interface.summary()
            query_summary["dataset"] = dataset
            query_summary["kind"] = KIND
            query_summary["guiding_bb"] = GUIDING_BB
            query_summary["percentile"] = PERCENTILE
            query_summary["noise_on_labeling"] = NOISE_ON_LABELING

            with open(out_dir_ds / f"query_summary_{name}.json", "w") as f:
                json.dump(query_summary, f, indent=2)

            t1_save = time.perf_counter()
            t1_ds = time.perf_counter()

            timings.append({
                "dataset": dataset,
                "n_classs": n_classs,
                "n_features": n_features,
                "n_synth": int(X_synth.shape[0]),
                "time_generate_s": round(t1_gen - t0_gen, 3),
                "time_label_s": round(t1_lab - t0_lab, 3),
                "time_save_s": round(t1_save - t0_save, 3),
                "time_total_s": round(t1_ds - t0_ds, 3),
                "query_calls_total": int(query_summary["query_calls_total"]),
                "query_points_total": int(query_summary["query_points_total"]),
                "query_calls_selection": int(query_summary["query_calls_selection"]),
                "query_points_selection": int(query_summary["query_points_selection"]),
                "query_calls_labeling": int(query_summary["query_calls_labeling"]),
                "query_points_labeling": int(query_summary["query_points_labeling"]),
                "query_epsilon": float(query_summary["query_epsilon"]),
                "noise_mech": query_summary["noise_mech"],
                "noise_on_labeling": bool(query_summary["noise_on_labeling"]),
            })

            print(f"[OK] saved: {out_path}")

            vals, cnts = np.unique(y_synth, return_counts=True)
            final_counts = np.zeros(n_classs, dtype=int)
            final_counts[vals.astype(int)] = cnts

            if BALANCE_MODE == "uniform" or TARGET_CLASS_RATIOS is None:
                base = N_SYNTH // n_classs
                target_counts = np.array([base] * n_classs, dtype=int)
                target_counts[: (N_SYNTH - base * n_classs)] += 1
            else:
                r = np.array(TARGET_CLASS_RATIOS, dtype=float)
                r = r / r.sum()
                target_counts = np.floor(r * N_SYNTH).astype(int)
                diff = N_SYNTH - target_counts.sum()
                for k in range(diff):
                    target_counts[k % n_classs] += 1

            save_distribution_summary(dataset, gen, out_dir_ds, name)
            save_history_csv(dataset, gen, out_dir_ds, name)
            save_wise_plots(dataset, gen, final_counts, target_counts, out_dir_ds, name)

        except Exception as e:
            print(f"[ERR] {dataset}: {e}")
            t1_ds = time.perf_counter()
            timings.append({
                "dataset": dataset,
                "n_classs": None,
                "n_features": None,
                "n_synth": 0,
                "time_generate_s": None,
                "time_label_s": None,
                "time_save_s": None,
                "time_total_s": round(t1_ds - t0_ds, 3),
                "error": str(e),
            })

    timings_df = pd.DataFrame(timings)
    timings_path = DIR_SYNTH / f"wise_timings_querypert_{KIND}_{GUIDING_BB}_{PERCENTILE}.csv"
    timings_df.to_csv(timings_path, index=False)
    print(f"\n[info] dataset-level timing saved to: {timings_path.resolve()}")


if __name__ == "__main__":
    main()
