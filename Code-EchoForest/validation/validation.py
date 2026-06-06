
import warnings
warnings.filterwarnings("ignore")
import scipy.sparse as sp
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, classification_report, f1_score,
    roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from loaders import load_trained_nn

DATASETS = ["adult", "activity", "pol", "spotify", "spotify-r", "landsat", "landsat2", "landsat-multi", "wave", "electricity","letters", "segment", "splice"]
DIR_DATA          = Path("../Data-original")
DIR_SYNTH         = Path("../Data-synthetic/wise")
DIR_MODELS_ORIG   = Path("../Model-original")
DIR_MODELS_SYNTH1 = Path("../Model-synthetic-wise")
DIR_REPORTS       = Path("../Reports-eval")

SYNTH_TEST_SIZE   = 0.30
SYNTH_RANDOM_STATE= 42
KIND       = "entropy"
GUIDING_BB = "nn"
PERCENTILE = 50

def to_np(t):
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return np.asarray(t)

def load_rf_model(dataset: str):
    with open(DIR_MODELS_ORIG / dataset / f"rf_{dataset}.sav", "rb") as f:
        return pickle.load(f)

def load_wise_synthetic(dataset: str,
                        kind: str,
                        guiding_bb: str,
                        percentile: int) -> tuple[np.ndarray, np.ndarray]:
    """Load X,y wise synthetic e returns numpy array."""
    path = DIR_SYNTH / dataset / f"synthetic_19_checks_{dataset}_{kind}_{guiding_bb}_{percentile}.csv"
    if not path.exists():
        raise FileNotFoundError(f"File synthetic not found: {path}")

    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError(f"Missing 'label' column in {path}")

    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.70, random_state=42,
                                                        stratify=y)
    return X_train, Y_train

def load_csv_real_splits(dataset: str):

    dd = DIR_DATA / dataset
    Xtr = pd.read_csv(dd / f"train_set_{dataset}.csv", index_col=0)
    Xte = pd.read_csv(dd / f"test_set_{dataset}.csv", index_col=0)
    ytr = pd.read_csv(dd / f"train_labels_{dataset}.csv", index_col=0).squeeze().to_numpy()
    yte = pd.read_csv(dd / f"test_labels_{dataset}.csv",  index_col=0).squeeze().to_numpy()
    Xtr = Xtr.loc[:, ~Xtr.columns.str.startswith("Unnamed")]
    Xte = Xte.loc[:, ~Xte.columns.str.startswith("Unnamed")]
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr.values)
    Xte = scaler.transform(Xte.values)
    return Xtr, Xte, ytr, yte

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

def load_original(dataset):
    if GUIDING_BB == "rf":
        bb = load_rf_model(dataset)
        n_features = int(getattr(bb, "n_features_in_", len(getattr(bb, "feature_names_in_", []))))
        if n_features == 0:
            raise RuntimeError("RF senza n_features_in_ / feature_names_in_.")
        n_classs = len(bb.classs_) if hasattr(bb, "classs_") else 2
    else:
        bb = load_trained_nn(dataset=dataset, base_dir=str(DIR_MODELS_ORIG))
        n_features = get_expected_in_features(bb)
        bb.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, n_features)
            out = bb(dummy)
        n_classs = int(out.shape[1]) if (out.ndim == 2 and out.shape[1] > 1) else 2
    return bb

def load_synth_rf(dataset: str,
                  kind: str,
                  guiding_bb: str,
                  percentile: int,
                  base_dir: str | Path = DIR_MODELS_SYNTH1):

    base_dir = Path(base_dir)
    cand_paths = [
        base_dir / dataset / f"rf_synth_{dataset}_{kind}_{guiding_bb}_{percentile}.sav",
        base_dir / dataset / f"rf_{dataset}_{kind}_{guiding_bb}_{percentile}.sav",
        base_dir / dataset / f"rf_{dataset}_wise.sav",
    ]

    model_path = None
    for p in cand_paths:
        if p.exists():
            model_path = p
            break

    if model_path is None:
        raise FileNotFoundError(
            f"Synthetic RF model not found. Checked:\n - " +
            "\n - ".join(str(p) for p in cand_paths)
        )

    with open(model_path, "rb") as f:
        rf = pickle.load(f)


    n_features = int(getattr(rf, "n_features_in_", len(getattr(rf, "feature_names_in_", []))))
    if n_features == 0:
        raise RuntimeError("Synthetic RF model has no n_features_in_/feature_names_in_. Was it fitted correctly?")

    n_classs = len(rf.classs_) if hasattr(rf, "classs_") else 2

    return rf

def ensure_numpy_float32(X):
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if sp.issparse(X):
        X = X.toarray()

    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    X = np.asarray(X, dtype=np.float32, order="C")
    return X

def predict_nn_in_batches(model, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:

    X = ensure_numpy_float32(X)
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            chunk = X[start:start+batch_size]
            xt = torch.tensor(chunk, dtype=torch.float32)
            logits = model(xt)
            if logits.ndim == 2 and logits.shape[1] > 1:
                pred = torch.argmax(logits, dim=1).cpu().numpy()
            else:
                pred = (torch.sigmoid(logits.flatten()) >= 0.5).cpu().numpy().astype(int)
            out.append(pred)
    return np.concatenate(out, axis=0)

def to_dataframe(X, model=None, ref_cols=None):

    if isinstance(X, pd.DataFrame):
        return X

    if ref_cols is None:

        if model is not None and hasattr(model, "feature_names_in_"):
            ref_cols = list(model.feature_names_in_)
        else:
            ref_cols = [f"f{i}" for i in range(X.shape[1])]


    if len(ref_cols) != X.shape[1]:
        ref_cols = [f"f{i}" for i in range(X.shape[1])]

    return pd.DataFrame(X, columns=ref_cols)

def align_to_model_feature_names(X, model):
    import pandas as pd


    if not isinstance(X, pd.DataFrame):
        if hasattr(model, "feature_names_in_"):
            cols = list(model.feature_names_in_)
            if X.shape[1] != len(cols):
                cols = [f"f{i}" for i in range(X.shape[1])]
        else:
            cols = [f"f{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=cols)

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    else:
        expected = list(X.columns)

    X = X.copy()
    missing = [c for c in expected if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    return X.reindex(columns=expected, fill_value=0.0)


def classification_report_dict(y_true, y_pred) -> dict:
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return rep

def safe_predict_proba(model, X: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:
            return None
    return None

def plot_roc_micro(out_path: Path, title: str, y_true: np.ndarray,
                   proba_A: np.ndarray | None, proba_B: np.ndarray | None,
                   classs: np.ndarray, label_A="Original", label_B="Synth RF"):

    plt.figure(figsize=(7,5))

    K = len(np.unique(y_true))

    if K > 2:
        Y = label_binarize(y_true, classs=classs)
        if proba_A is not None:
            fpr, tpr, _ = roc_curve(Y.ravel(), proba_A[:, :K].ravel())  # [:K] per sicurezza
            aucA = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{label_A} (micro AUC={aucA:.3f})")
        if proba_B is not None:
            fpr, tpr, _ = roc_curve(Y.ravel(), proba_B[:, :K].ravel())
            aucB = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{label_B} (micro AUC={aucB:.3f})")
    else:

        pos = classs.max()
        if proba_A is not None:
            idxA = list(classs).index(pos)
            fpr, tpr, _ = roc_curve(y_true, proba_A[:, idxA])
            aucA = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{label_A} (AUC={aucA:.3f})")
        if proba_B is not None:
            idxB = list(classs).index(pos)
            fpr, tpr, _ = roc_curve(y_true, proba_B[:, idxB])
            aucB = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{label_B} (AUC={aucB:.3f})")

    plt.plot([0,1], [0,1], 'k--', lw=1)
    plt.xlim([0,1]); plt.ylim([0,1])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_f1_per_class(out_path: Path, title: str, classs: np.ndarray,
                      f1_A: dict, f1_B: dict,
                      label_A="Original", label_B="Synth RF"):

    cls = [str(c) for c in classs]
    valsA = [f1_A.get(c, {}).get("f1-score", 0.0) for c in cls]
    valsB = [f1_B.get(c, {}).get("f1-score", 0.0) for c in cls]

    x = np.arange(len(cls))
    w = 0.38
    plt.figure(figsize=(max(8, 0.5*len(cls)+3), 5))
    plt.bar(x - w/2, valsA, width=w, label=label_A)
    plt.bar(x + w/2, valsB, width=w, label=label_B)
    plt.xticks(x, cls, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("F1 per class")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_macro_f1_summary(out_path: Path, title: str, summary_rows: list[tuple[str,float]]):

    labels = [r[0] for r in summary_rows]
    values = [r[1] for r in summary_rows]
    x = np.arange(len(labels))
    plt.figure(figsize=(max(7, 1.2*len(labels)), 5))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Macro-F1")
    plt.title(title)
    for i,v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def save_report_txt(path: Path, header: str, rep_dict: dict):
    path.write_text(header + "\n" + json.dumps(rep_dict, indent=2))

def predict_proba_nn_in_batches(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = 4096,
    device: str = "cpu",
    return_2d_binary: bool = True,
) -> np.ndarray:

    model.eval()
    dev = torch.device(device)
    out = []
    X = ensure_numpy_float32(X)
    for start in range(0, len(X), batch_size):
        chunk = X[start:start+batch_size]
        xt = torch.as_tensor(chunk, dtype=torch.float32, device=dev)

        logits = model(xt)
        if logits.ndim == 2 and logits.shape[1] > 1:
            p = torch.softmax(logits, dim=1)
            out.append(to_np(p))
        else:
            p1 = torch.sigmoid(logits.flatten())
            if return_2d_binary:
                p = torch.stack([1.0 - p1, p1], dim=1)

                out.append(to_np(p))
            else:

                out.append(to_np(p))

    proba = np.concatenate(out, axis=0)
    return to_np(proba)



def main():
    DIR_REPORTS.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        out_dir = DIR_REPORTS / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== EVAL: {ds} ===")

        try:

            rf_orig = load_original(ds)
            rf_synth = load_synth_rf(ds, KIND, GUIDING_BB, PERCENTILE)
            print(f"[info] {GUIDING_BB} original")



            Xtr_real, Xte_real, ytr_real, yte_real = load_csv_real_splits(ds)


            try:
                Xsynth_all, ysynth_all = load_wise_synthetic(ds, KIND, GUIDING_BB, PERCENTILE)
                Xtr_synth, Xte_synth, ytr_synth, yte_synth = train_test_split(
                    Xsynth_all, ysynth_all, test_size=SYNTH_TEST_SIZE,
                    random_state=SYNTH_RANDOM_STATE, stratify=ysynth_all
                )
            except Exception as e:
                print(f"[warn] synthetic non disponibile/valido: {e}")
                Xtr_synth = Xte_synth = ytr_synth = yte_synth = None

            Xtr_real_for_orig = align_to_model_feature_names(Xtr_real, rf_orig)
            Xte_real_for_orig = align_to_model_feature_names(Xte_real, rf_orig)
            if Xtr_synth is not None:
                Xtr_synth_for_orig = align_to_model_feature_names(Xtr_synth, rf_orig)
                Xte_synth_for_orig = align_to_model_feature_names(Xte_synth, rf_orig)


            Xtr_real_for_synth = align_to_model_feature_names(Xtr_real, rf_synth)
            Xte_real_for_synth = align_to_model_feature_names(Xte_real, rf_synth)
            if Xtr_synth is not None:
                Xtr_synth_for_synth = align_to_model_feature_names(Xtr_synth, rf_synth)
                Xte_synth_for_synth = align_to_model_feature_names(Xte_synth, rf_synth)

            reports_summary = []

            y_pred = predict_nn_in_batches(rf_orig, Xtr_real_for_orig)
            rep_tr_real_orig = classification_report_dict(ytr_real, y_pred)
            save_report_txt(out_dir / f"report_original_on_real_train_{KIND}_{GUIDING_BB}_{PERCENTILE}.json",
                            f"[Original {GUIDING_BB} on REAL train]",
                            rep_tr_real_orig)
            mF1 = f1_score(ytr_real, y_pred, average="macro")
            reports_summary.append((GUIDING_BB+" REAL train", mF1))

            y_pred = predict_nn_in_batches(rf_orig, Xte_real_for_orig)
            rep_te_real_orig = classification_report_dict(yte_real, y_pred)
            save_report_txt(out_dir / f"report_original_on_real_test_{KIND}_{GUIDING_BB}_{PERCENTILE}.json",
                            f"[Original {GUIDING_BB} on REAL test]",
                            rep_te_real_orig)
            mF1 = f1_score(yte_real, y_pred, average="macro")
            reports_summary.append((GUIDING_BB+" REAL test", mF1))


            if Xtr_synth is not None:
                y_pred = predict_nn_in_batches(rf_orig, Xtr_synth_for_orig)
                rep_tr_synth_orig = classification_report_dict(ytr_synth, y_pred)
                save_report_txt(out_dir / f"report_original_on_synth_train_{KIND}_{GUIDING_BB}_{PERCENTILE}.json",
                                f"[Original {GUIDING_BB} on SYNTH train]",
                                rep_tr_synth_orig)
                mF1 = f1_score(ytr_synth, y_pred, average="macro")
                reports_summary.append((GUIDING_BB+ " SYNTH train", mF1))

                y_pred = predict_nn_in_batches(rf_orig, Xte_synth_for_orig)
                rep_te_synth_orig = classification_report_dict(yte_synth, y_pred)
                save_report_txt(out_dir / f"report_original_on_synth_test_{KIND}_{GUIDING_BB}_{PERCENTILE}.json",
                                f"[Original {GUIDING_BB} on SYNTH test]",
                                rep_te_synth_orig)
                mF1 = f1_score(yte_synth, y_pred, average="macro")
                reports_summary.append((GUIDING_BB+" SYNTH test", mF1))


            y_pred = predict_nn_in_batches(rf_orig, Xtr_real_for_synth)
            rep_tr_real_synth = classification_report_dict(ytr_real, y_pred)
            save_report_txt(out_dir / f"report_synth_on_real_train_{KIND}_{GUIDING_BB}_{PERCENTILE}.json",
                            "[Synth RF on REAL train]",
                            rep_tr_real_synth)
            mF1 = f1_score(ytr_real, y_pred, average="macro")
            reports_summary.append(("PREMs REAL train", mF1))

            y_pred = predict_nn_in_batches(rf_orig, Xte_real_for_synth)
            rep_te_real_synth = classification_report_dict(yte_real, y_pred)
            save_report_txt(out_dir / f"report_synth_on_real_test_{KIND}_{GUIDING_BB}_{PERCENTILE}.json",
                            "[Synth RF on REAL test]",
                            rep_te_real_synth)
            mF1 = f1_score(yte_real, y_pred, average="macro")
            reports_summary.append(("PREMs REAL test", mF1))


            if Xtr_synth is not None:
                y_pred = predict_nn_in_batches(rf_orig, Xtr_synth_for_synth)
                rep_tr_synth_synth = classification_report_dict(ytr_synth, y_pred)
                save_report_txt(out_dir / f"report_synth_on_synth_train_{KIND}_{GUIDING_BB}_{PERCENTILE}.json",
                                "[Synth RF on SYNTH train]",
                                rep_tr_synth_synth)
                mF1 = f1_score(ytr_synth, y_pred, average="macro")
                reports_summary.append(("PREMs SYNTH train", mF1))

                y_pred = predict_nn_in_batches(rf_orig, Xte_synth_for_synth)
                rep_te_synth_synth = classification_report_dict(yte_synth, y_pred)
                save_report_txt(out_dir / f"report_synth_on_synth_test_{KIND}_{GUIDING_BB}_{PERCENTILE}.json",
                                "[Synth RF on SYNTH test]",
                                rep_te_synth_synth)
                mF1 = f1_score(yte_synth, y_pred, average="macro")
                reports_summary.append(("PREMs SYNTH test", mF1))


            plot_macro_f1_summary(
                out_dir / f"macro_f1_summary_{KIND}_{GUIDING_BB}_{PERCENTILE}.png",
                f"{ds} — Macro-F1 (original vs synth RF, real & synthetic splits)",
                reports_summary
            )


            cls_real = np.unique(yte_real)
            f1A = rep_te_real_orig   # original RF on real test
            f1B = rep_te_real_synth  # synth RF on real test
            plot_f1_per_class(
                out_dir / f"f1_per_class_real_test_{KIND}_{GUIDING_BB}_{PERCENTILE}.png",
                f"{ds} — F1 per class (REAL test)",
                cls_real, f1A, f1B,
                label_A=f"Original {GUIDING_BB}", label_B="Synth RF"
            )


            probaA = predict_proba_nn_in_batches(rf_orig, Xte_real_for_orig)
            #probaB = predict_proba_nn_in_batches(rf_synth, Xte_real_for_synth)
            #probaA = safe_predict_proba(rf_orig,  Xte_real_for_orig)
            probaB = safe_predict_proba(rf_synth, Xte_real_for_synth)
            classs_real = np.unique(yte_real)
            if probaA is not None or probaB is not None:
                plot_roc_micro(
                    out_dir / f"roc_micro_real_test_{KIND}_{GUIDING_BB}_{PERCENTILE}.png",
                    f"{ds} — ROC micro-average (REAL test)",
                    yte_real, probaA, probaB, classs_real,
                    label_A= f"Original {GUIDING_BB}", label_B="Synth RF"
                )

            if Xtr_synth is not None:
                probaA = predict_proba_nn_in_batches(rf_orig,  Xte_synth_for_orig)
                #probaB = predict_proba_nn_in_batches(rf_synth, Xte_synth_for_synth)
                #probaA = safe_predict_proba(rf_orig,  Xte_synth_for_orig)
                probaB = safe_predict_proba(rf_synth, Xte_synth_for_synth)
                classs_syn = np.unique(yte_synth)
                if probaA is not None or probaB is not None:
                    plot_roc_micro(
                        out_dir / f"roc_micro_synth_test_{KIND}_{GUIDING_BB}_{PERCENTILE}.png",
                        f"{ds} — ROC micro-average (SYNTH test)",
                        yte_synth, probaA, probaB, classs_syn,
                        label_A= f"Original {GUIDING_BB}", label_B="Synth RF"
                    )

            print(f"Report & plots saved to {out_dir}")

        except Exception as e:
            print(f"[{ds}] {e}")

if __name__ == "__main__":
    main()
