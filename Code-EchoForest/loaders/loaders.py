# === Loader robusto per reti addestrate per-dataset ===
from pathlib import Path
import re
import ast
import torch
import torch.nn as nn

# ---- MLP con chiavi 'net.*' (Sequential) ----
class MLPSequential(nn.Module):
    def __init__(self, in_features, n_classes, hidden=(64, 32), dropout=0.2):
        super().__init__()
        layers = []
        last = in_features
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: (N, in_features)
        return self.net(x)

# ---- Rete con layer nominati 'lin1','lin2','lin3',... ----
class LinNet(nn.Module):
    """Crea una rete con layer nominati lin1, lin2, ..., per matchare state_dict con chiavi 'link.*'."""
    def __init__(self, sizes: list[int], dropout=0.2):
        """
        sizes: [in_features, h1, h2, ..., n_classes]
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # costruisce lin1..linK-1; l’ultimo è la testa di output
        self._lins = nn.ModuleList()
        for i in range(len(sizes) - 1):
            setattr(self, f"lin{i+1}", nn.Linear(sizes[i], sizes[i+1]))
            self._lins.append(getattr(self, f"lin{i+1}"))

    def forward(self, x):
        # applica ReLU+dropout su tutti tranne l’ultimo
        for i, lin in enumerate(self._lins, 1):
            x = lin(x)
            if i < len(self._lins):
                x = torch.relu(x)
                x = self.dropout(x)
        return x

# ---- Utilità di parsing e inferenza ----
def _parse_trainconfig_to_dict(txt: str) -> dict:
    """
    Converte 'TrainConfig(a=1, hidden=(64,32), dropout=0.2)' -> {'a':1,'hidden':(64,32),'dropout':0.2}.
    Se non trova TrainConfig(...), prova literal_eval diretto.
    """
    s = txt.strip()
    m = re.search(r"\((.*)\)\s*$", s, flags=re.DOTALL)
    if not m:
        return ast.literal_eval(s)
    inside = m.group(1).strip()
    dict_str = "{" + inside + "}"
    return ast.literal_eval(dict_str)

def _infer_dims_from_state_dict(state: dict, kind: str):
    """
    Ritorna (in_features, hidden_tuple, n_classes) inferiti dallo state_dict.
    kind: 'net' (Sequential) oppure 'lin' (lin1/lin2/..).
    """
    if kind == "net":
        # trova tutti i Linear: net.<idx>.weight (idx = 0,3,6,...) ordinati
        idxs = sorted({int(m.group(1)) for k in state.keys()
                       if (m := re.match(r"net\.(\d+)\.weight$", k))})
        if not idxs:
            raise ValueError("Impossibile inferire layer 'net.*' dallo state_dict.")
        weights = [state[f"net.{i}.weight"] for i in idxs]
        in_features = weights[0].shape[1]
        out_dims = [w.shape[0] for w in weights]
        hidden = tuple(out_dims[:-1])
        n_classes = out_dims[-1]
        return in_features, hidden, n_classes
    else:
        # chiavi lin1.weight, lin2.weight, ..., linK.weight
        lin_idxs = sorted({int(m.group(1)) for k in state.keys()
                           if (m := re.match(r"lin(\d+)\.weight$", k))})
        if not lin_idxs:
            raise ValueError("Impossibile inferire layer 'lin*' dallo state_dict.")
        weights = [state[f"lin{i}.weight"] for i in lin_idxs]
        sizes = [weights[0].shape[1]] + [w.shape[0] for w in weights]
        in_features = sizes[0]
        hidden = tuple(sizes[1:-1])
        n_classes = sizes[-1]
        return in_features, hidden, n_classes

def load_trained_nn(dataset: str, base_dir: str = "../Model-original") -> nn.Module:
    """
    Carica la rete addestrata per `dataset` ricostruendo l'architettura esatta
    in base ai file saved to: {base_dir}/{dataset}/
      - nn_{dataset}.sav  (state_dict torch)
      - nn_{dataset}_best_params.txt  (opzionale: 'TrainConfig(...)')
    Supporta sia checkpoint con chiavi 'net.*' (Sequential) sia 'lin*.*'.
    """
    ddir = Path(base_dir) / dataset
    state_path = ddir / f"nn_{dataset}.sav"
    cfg_path = ddir / f"nn_{dataset}_best_params.txt"

    if not state_path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {state_path}")

    # 1) carica state_dict
    state = torch.load(state_path, map_location="cpu")

    # 2) prova a leggere dropout/hidden dal file config (se disponibile)
    hidden_cfg, dropout_cfg = None, None
    if cfg_path.exists():
        try:
            cfg = _parse_trainconfig_to_dict(cfg_path.read_text())
            if "hidden" in cfg: hidden_cfg = tuple(cfg["hidden"])
            if "dropout" in cfg: dropout_cfg = float(cfg["dropout"])
        except Exception as e:
            print(f"[warn] parsing config fallito ({e}); userò inferenza dai pesi.")

    # 3) decide il “kind” delle chiavi
    has_net = any(k.startswith("net.") for k in state.keys())
    has_lin = any(k.startswith("lin") for k in state.keys())
    if has_net:
        in_features, hidden_inf, n_classes = _infer_dims_from_state_dict(state, kind="net")
        hidden = hidden_cfg or hidden_inf
        dropout = dropout_cfg if dropout_cfg is not None else 0.2
        model = MLPSequential(in_features, n_classes, hidden=hidden, dropout=dropout)
        # carica in modo STRICT: i nomi coincidono (net.*)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    if has_lin:
        in_features, hidden_inf, n_classes = _infer_dims_from_state_dict(state, kind="lin")
        hidden = hidden_cfg or hidden_inf
        dropout = dropout_cfg if dropout_cfg is not None else 0.2
        sizes = [in_features] + list(hidden) + [n_classes]
        model = LinNet(sizes=sizes, dropout=dropout)
        model.load_state_dict(state, strict=True)  # chiavi lin1/lin2/...
        model.eval()
        return model

    # fallback: prova a mappare net.* -> lin*.* (o viceversa) se necessario
    # (di solito non serve; teniamo un messaggio chiaro)
    raise ValueError("Formato state_dict non riconosciuto (niente 'net.*' né 'lin*').")

