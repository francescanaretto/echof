import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# same MLP used during training
class MLP(nn.Module):
    def __init__(self, in_features: int, n_classes: int, hidden=(10,5), dropout=0.2):
        super().__init__()
        layers, prev = [], in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, n_classes)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def _parse_trainconfig_to_dict(txt: str) -> dict:

    m = re.search(r"\((.*)\)\s*$", s, flags=re.DOTALL)
    if not m:

        return ast.literal_eval(s)
    inside = m.group(1).strip()

    dict_str = "{" + inside + "}"
    return ast.literal_eval(dict_str)

def load_nn_model(dataset: str,
                  in_features: int,
                  n_classes: int,
                  base_dir: str = "../Model-original") -> nn.Module:

    path = Path(base_dir) / dataset


    cfg_path = path / f"nn_{dataset}_best_params.txt"
    hidden = (10, 5)
    dropout = 0.2
    if cfg_path.exists():
        cfg_txt = cfg_path.read_text()
        try:
            cfg = _parse_trainconfig_to_dict(cfg_txt)
            hidden = tuple(cfg.get("hidden", hidden))
            dropout = float(cfg.get("dropout", dropout))
        except Exception as e:
            print(f"[warn] parsing config fallito ({e}); uso default hidden={hidden}, dropout={dropout}")
    else:
        print(f"[warn] config non trovato: {cfg_path}; uso default hidden={hidden}, dropout={dropout}")


    model = MLP(in_features, n_classes, hidden=hidden, dropout=dropout)


    weights_path = path / f"nn_{dataset}.sav"
    state = torch.load(weights_path, map_location="cpu")

    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"[OK] NN caricata per {dataset}: hidden={hidden}, dropout={dropout}")
    return model
