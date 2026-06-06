

import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import classification_report

"""
Train the original Random Forest black-box with an explicit GridSearchCV model
selection step.

The best model is chosen by cross-validation and then saved together with full
training/test reports and the complete CV table.
"""

DATASETS = ["adult", "activity", "pol", "spotify", "spotify-r", "landsat", "landsat2", "electricity", "magic", "credit", "letters", "diamonds_binary", "hypo_binary"]
DIR_DATA = Path("../Data-original")
DIR_OUT  = Path("../Model-original")

# Hyperparameter grid explored with GridSearchCV.
param_grid = {
    "n_estimators": [200, 400, 800],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [10, 20, 40],
    "min_samples_split": [ 8, 16],
    "min_samples_leaf": [8, 15],
    "max_features": ["sqrt", "log2", 0.5],
    "bootstrap": [True],
    "max_samples": [0.6, 0.8, None],
    "class_weight": ["balanced", "balanced_subsample"],
    "min_impurity_decrease": [0.0, 1e-4, 1e-3],
}

def load_split(ds: str):
    ddir = DIR_DATA / ds
    Xtr = pd.read_csv(ddir / f"train_set_{ds}.csv", index_col=0)
    Xte = pd.read_csv(ddir / f"test_set_{ds}.csv",  index_col=0)
    ytr = pd.read_csv(ddir / f"train_labels_{ds}.csv", index_col=0).squeeze()
    yte = pd.read_csv(ddir / f"test_labels_{ds}.csv",  index_col=0).squeeze()
    return Xtr, Xte, ytr, yte

for ds in DATASETS:
    try:
        print(f"\n=== {ds} ===")
        Xtr, Xte, ytr, yte = load_split(ds)

        # Binary tasks use ROC AUC; multiclass tasks use macro F1.
        classes = np.unique(ytr)
        is_binary = (len(classes) == 2)
        scoring = "roc_auc" if is_binary else "f1_macro"

        cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=0)

        base = RandomForestClassifier(
            random_state=0,
            n_jobs=20,
            oob_score=False
        )

        grid = GridSearchCV(
            estimator=base,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=20,
            verbose=1,
            refit=True
        )
        grid.fit(Xtr, ytr)

        best = grid.best_estimator_
        print("Best params:", grid.best_params_)

        outdir = (DIR_OUT / ds)
        outdir.mkdir(parents=True, exist_ok=True)

        with open(outdir / f"rf_{ds}.sav", "wb") as f:
            pickle.dump(best, f)

        pred_tr = best.predict(Xtr)
        rep_tr = classification_report(ytr, pred_tr)
        (outdir / f"rf_{ds}_report_train.txt").write_text(rep_tr)

        pred_te = best.predict(Xte)
        rep_te = classification_report(yte, pred_te)
        (outdir / f"rf_{ds}_report_test.txt").write_text(rep_te)

        cvres = pd.DataFrame(grid.cv_results_)
        cvres.to_csv(outdir / f"rf_{ds}_cv_results.csv", index=False)

        print("[OK] saved in", outdir)
        print("[TEST]\n", rep_te)

    except Exception as e:
        print(f"[ERR] {ds}: {e}")
