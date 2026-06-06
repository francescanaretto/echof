
"""
Wise generator
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
from pathlib import Path
import time
import math
import numpy as np
import pandas as pd
import pickle
import random


import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS   = ["adult", "activity", "pol", "spotify", "spotify-r", "california", "htru2", "landast-multi", "landsat", "landsat2", "electricity"]
DIR_DATA   = Path("../Data-original")
DIR_MODELS = Path("../Model-original")
DIR_SYNTH  = Path("../Data-synthetic/wise")


N_SYNTH          = 800000       # n synth data
MAX_TRIALS       = 100000       # n max trials
BATCH_SIZE_CAND  = 1024         # batch size
KIND             = "kappa"    # "entropy" | "margin" | "kappa" | "logit"
PERCENTILE       = 25           # for "entropy"
GUIDING_BB       = "nn"         # "nn" | "rf"
RANDOM_SEED      = 42
BATCH_SIZE_LABEL = 4096


BALANCE_MODE         = "ratios"     # "uniform" or "ratios"
TARGET_CLASS_RATIOS  = None        # es. [0.2, 0.8] se BALANCE_MODE="ratios"
BALANCE_TOLERANCE    = 4000
EARLY_STALL_PATIENCE = 50

# --- exploration / exploitation  ---
ADAPTIVE_MIX      = True
ADAPT_CLASS_AWARE = True
DIRICHLET_ALPHA   = 1.0
MIX_UPDATE_EVERY  = 10

CHOICE_POLICY        = "eps_greedy"  # "eps_greedy" | "adaptive_mix" | "uniform"
EPS_START            = 0.25
EPS_MIN              = 0.02
EPS_DECAY            = 0.999
WARMUP_ITERS         = 50

EXP_WINDOW           = 300
EXP_TOP_K            = 4
CLASS_AWARE_EXPLOIT  = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for EchoForest from an original black-box."
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
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def apply_cli_overrides(args):
    global DATASETS, N_SYNTH, KIND, PERCENTILE, GUIDING_BB, RANDOM_SEED
    if args.datasets:
        DATASETS = args.datasets
    if args.n_synth is not None:
        N_SYNTH = args.n_synth
    if args.guiding_bb is not None:
        GUIDING_BB = args.guiding_bb
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


def load_train_df(dataset: str) -> pd.DataFrame:
    p = DIR_DATA / dataset / f"train_set_{dataset}.csv"
    df = pd.read_csv(p, index_col=0)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
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
        raise RuntimeError("Could not find il primo Linear nella rete.")
    return int(first_linear.in_features)

def predict_proba_bb(bb, X: np.ndarray) -> np.ndarray:
    if hasattr(bb, "predict_proba"):
        return bb.predict_proba(X)
    if isinstance(bb, nn.Module):
        bb.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32)
            logits = bb(xt)
            if logits.ndim == 2 and logits.shape[1] > 1:
                proba = torch.softmax(logits, dim=1).cpu().numpy()
            else:
                p1 = torch.sigmoid(logits.flatten())
                proba = torch.stack([1 - p1, p1], dim=1).cpu().numpy()
        return proba
    raise RuntimeError("BB not recognized (sklearn o nn.Module).")

def label_with_bb(bb, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    if hasattr(bb, "predict"):
        return bb.predict(X)
    if isinstance(bb, nn.Module):
        bb.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, X.shape[0], batch_size):
                chunk = X[start:start+batch_size]
                xt = torch.tensor(chunk, dtype=torch.float32)
                logits = bb(xt)
                if logits.ndim == 2 and logits.shape[1] > 1:
                    pred = torch.argmax(logits, dim=1).cpu().numpy()
                else:
                    p1 = torch.sigmoid(logits.flatten())
                    pred = (p1 >= 0.5).cpu().numpy().astype(int)
                preds.append(pred)
        return np.concatenate(preds, axis=0)
    raise RuntimeError("BB not recognized per labeling.")
def _compute_target_counts(n_samples: int,
                           n_classs: int,
                           balance_mode: str,
                           target_ratios: list[float] | None) -> np.ndarray:
    """
    Crea i target per class in modo robusto:
    - 'uniform': quote uguali
    - 'ratios': usa target_ratios se e solo se len == n_classs
                altrimenti fallback automatico a 'uniform' con warning.
    """
    if balance_mode != "ratios" or target_ratios is None:
        base = n_samples // n_classs
        target = np.array([base] * n_classs, dtype=int)
        target[: (n_samples - base * n_classs)] += 1
        return target

    r = np.array(target_ratios, dtype=float)
    if r.ndim != 1 or r.size != n_classs or not np.isfinite(r).all() or (r <= 0).all():
        print(f"[warn] target_ratios incompatibili con n_classs={n_classs} "
              f"(len={len(target_ratios)}). Fallback a 'uniform'.")
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


class WiseGenerator:
    """
    - Sampler multi-distribuzione (19 famiglie) con mix adattivo
    - Selezione per incertezza (entropy/margin/logit/kappa)
    - Bilanciamento classes on-the-fly secondo target
    """

    NUM_DISTS = 19

    def __init__(
        self,
        bb,
        n_feature:g. int,
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
        self.n_features = n_features
        self.n_classs = n_classs
        self.dataset_name = dataset_name
        self.kind = kind
        self.percentile = percentile
        self.rng = np.random.default_rng(random_state)


        self.adaptive_mix = adaptive_mix
        self.adapt_class_aware = adapt_class_aware
        self.dirichlet_alpha = dirichlet_alpha
        self.mix_update_every = mix_update_every

        self.dist_success = np.zeros(self.NUM_DISTS, dtype=float)
        self.dist_trials  = np.zeros(self.NUM_DISTS, dtype=float)
        self.dist_success_by_class = np.zeros((self.NUM_DISTS, n_classs), dtype=float)

        self.dist_weights = np.ones(self.NUM_DISTS, dtype=float) / self.NUM_DISTS

        # --- report ---
        self.hist_iters = []
        self.hist_choice = []
        self.hist_accept = []
        self.hist_counts = []
        self.hist_percentile = []
        self.hist_weights = []
        self.hist_update_step = []


        self.hist_acc_by_class = []


        self.choice_policy       = CHOICE_POLICY
        self.eps_start           = EPS_START
        self.eps_min             = EPS_MIN
        self.eps_decay           = EPS_DECAY
        self.warmup_iters        = WARMUP_ITERS
        self.exp_window          = EXP_WINDOW
        self.exp_top_k           = EXP_TOP_K
        self.class_aware_exploit = CLASS_AWARE_EXPLOIT



    def _recent_scores(self, deficits: np.ndarray | None) -> np.ndarray:

        n = len(self.hist_choice)
        if n == 0:
            return np.ones(self.NUM_DISTS, dtype=float)

        start = max(0, n - self.exp_window)
        scores = np.zeros(self.NUM_DISTS, dtype=float)

        if self.class_aware_exploit and deficits is not None:
            if deficits is not None and deficits.shape[0] != self.n_classs:
                d = np.zeros(self.n_classs, dtype=float)
            else:
                d = np.maximum(deficits.astype(float), 0.0) if deficits is not None else None

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

        elif self.choice_policy == "adaptive_mix":
            p = self.dist_weights / self.dist_weights.sum()
            return int(self.rng.choice(self.NUM_DISTS, p=p))

        else:
            return int(self.rng.integers(0, self.NUM_DISTS))

    def _entropy(self, p: np.ndarray) -> float:
        p = np.clip(p, 1e-12, 1.0)
        return float(-(p * np.log(p)).sum() / np.log(self.n_classs))

    def check_entropy(self, dataset_proba: np.ndarray) -> np.ndarray:
        entropies = np.apply_along_axis(self._entropy, 1, dataset_proba)
        thr = np.percentile(entropies, max(1, self.percentile))
        return np.where(entropies > thr)[0]

    def check_margin(self, dataset_proba: np.ndarray) -> np.ndarray:
        sorted_proba = -np.sort(-dataset_proba, axis=1)
        margins = sorted_proba[:, 0] - sorted_proba[:, 1]
        mean_m, std_m = margins.mean(), margins.std()
        return np.where(margins < (mean_m - std_m))[0]

    def check_logit_margin(self, proba: np.ndarray) -> np.ndarray:
        logits = np.log(proba + 1e-9)
        sorted_logits = -np.sort(-logits, axis=1)
        margins = sorted_logits[:, 0] - sorted_logits[:, 1]
        mean_m, std_m = margins.mean(), margins.std()
        return np.where(margins < (mean_m - std_m))[0]

    def check_kappa(self, proba: np.ndarray) -> np.ndarray:
        sorted_p = -np.sort(-proba, axis=1)
        pmax, p2 = sorted_p[:, 0], sorted_p[:, 1]
        kappa = (pmax - p2) / (1 - pmax + 1e-9)
        mean_k, std_k = kappa.mean(), kappa.std()
        return np.where(kappa < (mean_k - std_k))[0]


    def _sample_from_choice(self, choice: int, size: int) -> np.ndarray:

        rng = self.rng
        d = self.n_features

        if choice == 0:
            data = rng.normal(0.0, 1.0, size=(size, d))
        elif choice == 1:
            mu = rng.uniform(-0.5, 0.5); sigma = rng.uniform(0.1, 1.0)
            data = rng.lognormal(mu, sigma, size=(size, d))
        elif choice == 2:
            shape = rng.uniform(0.5, 5.0); scale = rng.uniform(0.5, 3.0)
            data = rng.gamma(shape, scale, size=(size, d))
        elif choice == 3:
            a = rng.uniform(0.5, 5.0); b = rng.uniform(0.5, 5.0)
            base = rng.beta(a, b, size=(size, d)); data = -3.0 + 6.0 * base
        elif choice == 4:
            df = rng.uniform(1.0, 5.0); data = rng.standard_t(df=df, size=(size, d))
        elif choice == 5:
            lam = rng.uniform(0.1, 2.0); base = rng.exponential(1.0/lam, size=(size, d))
            sign = rng.choice([-1.0, 1.0], size=(size, d)); data = base * sign
        elif choice == 6:
            lam = rng.uniform(1.0, 20.0); base = rng.poisson(lam, size=(size, d)).astype(float)
            data = base - lam
        elif choice == 7:
            pi = rng.uniform(0.2, 0.8); comp = rng.binomial(1, pi, size=size)
            mu1 = rng.uniform(-2.0, 2.0); mu2 = rng.uniform(-5.0, 5.0)
            s1 = rng.uniform(0.4, 2.5);    s2 = rng.uniform(0.4, 3.0)
            g1 = rng.normal(mu1, s1, size=(size, d)); g2 = rng.normal(mu2, s2, size=(size, d))
            data = np.where(comp[:, None] == 1, g1, g2)
        elif choice == 8:
            p = rng.uniform(0.1, 0.9); base = rng.binomial(1, p, size=(size, d))
            data = 2 * base - 1
        elif choice == 9:
            K = rng.integers(3, 6); pis = rng.dirichlet(np.ones(K))
            comps = rng.choice(K, size=size, p=pis); data = np.zeros((size, d), dtype=np.float32)
            for k in range(K):
                idx = np.where(comps == k)[0];
                if len(idx)==0: continue
                mu = rng.uniform(-4, 4); sigma = rng.uniform(0.3, 3.0)
                data[idx] = rng.normal(mu, sigma, size=(len(idx), d))
        elif choice == 10:
            shape = rng.uniform(1.0, 5.0); data = rng.pareto(shape, size=(size, d))
            sign = rng.choice([-1.0, 1.0], size=(size, d)); data = data * sign
        elif choice == 11:
            a = rng.uniform(0.5, 3.0); data = rng.weibull(a, size=(size, d))
        elif choice == 12:
            loc = rng.uniform(-2.0, 2.0); scale = rng.uniform(0.2, 2.0)
            data = rng.laplace(loc=loc, scale=scale, size=(size, d))
        elif choice == 13:
            data = rng.standard_cauchy(size=(size, d)); data = np.clip(data, -20, 20)
        elif choice == 14:
            left = rng.uniform(-4, -1); mode = rng.uniform(left, left + 5); right = rng.uniform(mode, mode + 5)
            data = rng.triangular(left, mode, right, size=(size, d))
        elif choice == 15:
            loc = rng.uniform(-2, 2); scale = rng.uniform(0.3, 2.0)
            u = rng.uniform(0, 1, size=(size, d)); data = loc + scale * np.log(u / (1 - u))
        elif choice == 16:
            A = rng.normal(size=(d, d)); cov = A @ A.T + np.eye(d)*1e-3
            mean = rng.normal(0, 1, size=d); data = rng.multivariate_normal(mean, cov, size=size)
        elif choice == 17:
            K = rng.integers(2, 5); pis = rng.dirichlet(np.ones(K))
            comps = rng.choice(K, size=size, p=pis); data = np.zeros((size, d), dtype=np.float32)
            for k in range(K):
                idx = np.where(comps == k)[0];
                if len(idx)==0: continue
                A = rng.normal(size=(d, d)); cov = A @ A.T + np.eye(d)*1e-3
                mean = rng.normal(0, 2, size=d); data[idx] = rng.multivariate_normal(mean, cov, size=len(idx))
        else:
            base = rng.binomial(1, 0.5, size=(size, d)).astype(float)
            n_pairs = max(1, d // 3)
            for _ in range(n_pairs):
                i = rng.integers(0, d); j = rng.integers(0, d)
                if i == j: continue
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
                print(f"[warn] _update_mix_weights: deficits shape {getattr(deficits, 'shape', None)} "
                      f"!= n_classs ({self.n_classs}). Ignoro i deficit (uso vettore zero).")
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


    def generate(self,
                 n_samples: int,
                 max_trials: int = 300000,
                 batch_size: int = 1024,
                 balance_mode: str = "uniform",
                 target_ratios: list[float] | None = None,
                 balance_tolerance: int = 0,
                 early_stall_patience: int = 200) -> np.ndarray:
        kept_per_class = [ [] for _ in range(self.n_classs) ]
        counts = np.zeros(self.n_classs, dtype=int)

        target_counts = _compute_target_counts(
            n_samples=n_samples,
            n_classs=self.n_classs,
            balance_mode=balance_mode,
            target_ratios=target_ratios
        )

        kept_total = 0
        trials = 0
        iters_since_accept = 0

        print(f"[info] wise generation target={n_samples}, kind={self.kind}")

        while kept_total < n_samples and trials < max_trials:
            trials += 1
            iters_since_accept += 1


            deficits = (target_counts - counts)


            if deficits.ndim != 1 or deficits.shape[0] != self.n_classs:
                print(f"[warn] generate: deficits shape {getattr(deficits, 'shape', None)} "
                      f"!= n_classs ({self.n_classs}). Forzo vettore zero.")
                deficits = np.zeros(self.n_classs, dtype=int)


            cand, choice_id = self._sample_distribution_block(
                batch_size, iter_idx=trials, deficits=deficits
            )
            self.dist_trials[choice_id] += 1


            proba = predict_proba_bb(self.bb, cand)


            if self.kind == "entropy":
                idx = self.check_entropy(proba)
            elif self.kind == "margin":
                idx = self.check_margin(proba)
            elif self.kind == "kappa":
                idx = self.check_kappa(proba)
            elif self.kind == "logit":
                idx = self.check_logit_margin(proba)
            else:
                idx = np.arange(len(cand))  # fallback


            if len(idx) == 0:
                if (trials % self.mix_update_every) == 0:
                    self._update_mix_weights(deficits)
                if early_stall_patience and iters_since_accept >= early_stall_patience and self.kind == "entropy":
                    self.percentile = max(1, self.percentile - 1)
                    iters_since_accept = 0
                    print(f"[warn] stallo: abbasso percentile a {self.percentile}")

                self.hist_iters.append(trials)
                self.hist_choice.append(choice_id)
                self.hist_accept.append(0)
                self.hist_counts.append(counts.copy())
                self.hist_percentile.append(self.percentile)
                self.hist_acc_by_class.append(np.zeros(self.n_classs, dtype=int))
                if (trials % self.mix_update_every) == 0:
                    self.hist_weights.append(self.dist_weights.copy())
                    self.hist_update_step.append(trials)
                continue


            yhat = np.argmax(proba, axis=1)

            accepted_idx = []
            acc_by_class = np.zeros(self.n_classs, dtype=int)

            for i in idx:
                y = int(yhat[i])
                if counts[y] < (target_counts[y] + balance_tolerance):
                    accepted_idx.append(i)
                    acc_by_class[y] += 1

            if accepted_idx:

                for i in accepted_idx:
                    y = int(yhat[i])
                    kept_per_class[y].append(cand[i:i + 1])  # 1xD

                counts += acc_by_class
                kept_total = int(counts.sum())
                iters_since_accept = 0


                self.dist_success[choice_id] += len(accepted_idx)
                for c in range(self.n_classs):
                    self.dist_success_by_class[choice_id, c] += acc_by_class[c]


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


        out_blocks = []
        for c in range(self.n_classs):
            if len(kept_per_class[c]) == 0:
                continue
            Xc = np.concatenate(kept_per_class[c], axis=0)
            if Xc.shape[0] > target_counts[c]:
                Xc = Xc[:target_counts[c]]
            out_blocks.append(Xc)

        if len(out_blocks) == 0:
            raise RuntimeError("No point selected: criteria too restrictive?")

        X_all = np.concatenate(out_blocks, axis=0)

        idx_perm = self.rng.permutation(X_all.shape[0])
        X_all = X_all[idx_perm]

        print(f"[info] generati {X_all.shape[0]} punti (trials={trials})")
        print(f"[info] final counts: {dict(enumerate(counts.tolist()))}")
        if self.adaptive_mix:
            print(f"[info] dist weights finali (head): {np.round(self.dist_weights, 3)}")
        return X_all

def save_wise_plots(dataset: str,
                    gen: WiseGenerator,
                    final_counts: np.ndarray,
                    target_counts: np.ndarray,
                    out_dir: Path, name:str):
    out_dir.mkdir(parents=True, exist_ok=True)
    print('In gen plot')

    if len(gen.hist_weights):
        W = np.vstack(gen.hist_weights)  # (T, D)
        it = np.array(gen.hist_update_step)
        final_w = W[-1]
        top_idx = np.argsort(-final_w)[:6]

        plt.figure(figsize=(9,5))
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
    plt.figure(figsize=(8,4))
    plt.bar(x - width/2, target_counts, width)
    plt.bar(x + width/2, final_counts, width)
    plt.title(f"{dataset}, Target vs actual per class")
    plt.xlabel("class")
    plt.ylabel("Num. samples")
    plt.xticks(x)
    plt.legend(["target", "actual"])
    plt.tight_layout()
    print('salvo')
    plt.savefig(out_dir / f"plot_classs_{name}.png", dpi=160)
    plt.close()


    plt.figure(figsize=(10,4))
    plt.bar(np.arange(gen.NUM_DISTS), gen.dist_success)
    plt.title(f"{dataset}, Elements per distribution")
    plt.xlabel("id stat. distribution")
    plt.ylabel("Num. samples")
    plt.tight_layout()
    plt.savefig(out_dir / f"plot_dist_success_{name}.png", dpi=160)
    plt.close()


    if len(gen.hist_accept) > 0:
        a = np.array(gen.hist_accept, dtype=float)
        k = max(1, len(a)//50)
        kern = np.ones(k)/k
        rolling = np.convolve(a, kern, mode="same")
        plt.figure(figsize=(9,4))
        plt.plot(gen.hist_iters, rolling)
        plt.title(f"{dataset}, Acceptance rate")
        plt.xlabel("iter")
        plt.ylabel("Num. of accepted per iter")
        plt.tight_layout()
        plt.savefig(out_dir / f"plot_accept_rate_{name}.png", dpi=160)
        plt.close()

def save_history_csv(dataset: str, gen: WiseGenerator, out_dir: Path, name:str):
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
        W = np.vstack(gen.hist_weights)  # shape: (snapshots, NUM_DISTS)
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


def main():
    args = parse_args()
    apply_cli_overrides(args)
    DIR_SYNTH.mkdir(parents=True, exist_ok=True)
    timings = []

    print(
        f"[info] generator config: datasets={DATASETS} n_synth={N_SYNTH} "
        f"mode={KIND} percentile={PERCENTILE} guiding_bb={GUIDING_BB} seed={RANDOM_SEED}"
    )

    for dataset in DATASETS:
        t0_ds = time.perf_counter()
        try:
            print(f"\n=== Dataset: {dataset} | Wise generator ({KIND}, BB={GUIDING_BB}) ===")


            if GUIDING_BB == "rf":
                bb = load_rf_model(dataset)
                n_features = int(getattr(bb, "n_features_in_", len(getattr(bb, "feature_names_in_", []))))
                if n_features == 0:
                    raise RuntimeError("RF senza n_features_in_ / feature_names_in_.")
                n_classs = len(bb.classs_) if hasattr(bb, "classs_") else 2
            else:
                bb = load_trained_nn(dataset=dataset, base_dir=str(DIR_MODELS))
                n_features = get_expected_in_features(bb)
                bb.eval()
                with torch.no_grad():
                    dummy = torch.zeros(1, n_features)
                    out = bb(dummy)
                n_classs = int(out.shape[1]) if (out.ndim == 2 and out.shape[1] > 1) else 2


            gen = WiseGenerator(
                bb=bb,
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
            y_synth = label_with_bb(bb, X_synth, batch_size=BATCH_SIZE_LABEL)
            t1_lab = time.perf_counter()


            vals, counts = np.unique(y_synth, return_counts=True)
            print("[info] distribuzione classes sintetiche:")
            for v, c in zip(vals, counts):
                print(f"  class {v}: {c} istanze")


            t0_save = time.perf_counter()
            out_dir_ds = DIR_SYNTH / dataset
            out_dir_ds.mkdir(parents=True, exist_ok=True)
            out_path = out_dir_ds / f"synthetic_19_checks_{dataset}_{KIND}_{GUIDING_BB}_{PERCENTILE}.csv"
            df_out = pd.DataFrame(X_synth)
            df_out["label"] = y_synth
            df_out.to_csv(out_path, index=False)
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


            save_distribution_summary(dataset, gen, DIR_SYNTH / dataset, f"synthetic_19_checks_{dataset}_{KIND}_{GUIDING_BB}_{PERCENTILE}")
            save_history_csv(dataset, gen, DIR_SYNTH / dataset, f"synthetic_19_checks_{dataset}_{KIND}_{GUIDING_BB}_{PERCENTILE}")


            save_wise_plots(dataset, gen, final_counts, target_counts, DIR_SYNTH / dataset, f"synthetic_19_checks_{dataset}_{KIND}_{GUIDING_BB}_{PERCENTILE}")


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
    timings_path = DIR_SYNTH / f"wise_timings_{dataset}_{KIND}_{GUIDING_BB}_{PERCENTILE}.csv"
    timings_df.to_csv(timings_path, index=False)
    print(f"\n[info] timing per dataset saved to: {timings_path.resolve()}")

if __name__ == "__main__":
    main()
