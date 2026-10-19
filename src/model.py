import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Mapping, Optional
import torch
from torch import nn

def _cfg_value(cfg: Any, name: str, default: Any = None) -> Any:
    if cfg is None: return default
    if isinstance(cfg, Mapping): return cfg.get(name, default)
    return getattr(cfg, name, default)

class SloshingPINN(nn.Module):
    """forward takes [x/Lref,y/Lref,t/Tref], returns u*, v*, p*, alpha."""
    def __init__(self, cfg: Any = None, width: Optional[int] = None,
                 hidden_layers: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        width = int(width if width is not None else _cfg_value(cfg, "hidden_width", 96))
        hidden_layers = int(hidden_layers if hidden_layers is not None else
                            _cfg_value(cfg, "hidden_layers", 5))
        if width < 1 or hidden_layers < 1:
            raise ValueError("width and hidden_layers must be positive.")
        layers, n_in = [], 3
        for _ in range(hidden_layers):
            layer = nn.Linear(n_in, width)
            nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
            layers += [layer, nn.Tanh()]
            n_in = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(width, 4)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, coords_star: torch.Tensor) -> dict[str, torch.Tensor]:
        if coords_star.ndim != 2 or coords_star.shape[1] != 3:
            raise ValueError("coords_star must have shape (N,3): [x*,y*,t*].")
        raw = self.head(self.backbone(coords_star))
        return {"u": raw[:, 0:1], "v": raw[:, 1:2], "p": raw[:, 2:3],
                "alpha": torch.sigmoid(raw[:, 3:4])}

    def predict_physical(self, coords_si: torch.Tensor) -> dict[str, torch.Tensor]:
        """SI [x,y,t] in; SI u,v,p and dimensionless alpha out."""
        if self.cfg is None:
            raise ValueError("predict_physical requires the Config used to build the model.")
        L = float(_cfg_value(self.cfg, "length_ref"))
        U = float(_cfg_value(self.cfg, "velocity_ref"))
        rho = float(_cfg_value(self.cfg, "rho_ref"))
        T = L / U
        scales = coords_si.new_tensor([L, L, T])
        f = self.forward(coords_si / scales)
        return {"u": U*f["u"], "v": U*f["v"], "p": rho*U**2*f["p"],
                "alpha": f["alpha"]}


# ============================================================
# Utilitaires de configuration
# ============================================================

def _get_config_value(cfg, name, default=None):
    """Lit une propriété d'un dictionnaire ou d'un objet Config."""
    if isinstance(cfg, dict):
        if name in cfg:
            return cfg[name]

        # Accepte aussi une configuration imbriquée dans "domain".
        domain = cfg.get("domain")
        if isinstance(domain, dict) and name in domain:
            return domain[name]

        return default

    if hasattr(cfg, name):
        return getattr(cfg, name)

    domain = getattr(cfg, "domain", None)
    if isinstance(domain, dict):
        return domain.get(name, default)

    if domain is not None and hasattr(domain, name):
        return getattr(domain, name)

    return default


def _load_config(config=None, config_path=None, config_class=None):
    """
    Accepte :
      - une instance de configuration ;
      - une classe Config et config_path ;
      - un chemin JSON directement ;
      - un dictionnaire déjà chargé.

    Si une classe Config est fournie, utilise sa méthode load_json.
    """
    if isinstance(config, (str, Path)):
        if config_path is not None:
            raise ValueError(
                "Ne fournis pas simultanément un chemin dans config "
                "et dans config_path."
            )
        config_path = config
        config = None

    # Autorise config=Config, si Config est une classe.
    if isinstance(config, type):
        config_class = config
        config = None

    if config_path is None:
        if config is None:
            raise ValueError(
                "Fournis une instance Config ou un chemin config_path."
            )
        return config

    config_path = Path(config_path).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {config_path}"
        )

    if config_class is not None:
        loaded = config_class.load_json(str(config_path))
        if loaded is not None:
            return loaded

        # Certaines méthodes load_json modifient l'instance en place.
        if config is not None:
            return config

        raise ValueError(
            "Config.load_json() n'a rien retourné. "
            "Passe une instance Config si load_json modifie celle-ci."
        )

    if config is not None and hasattr(config, "load_json"):
        loaded = config.load_json(str(config_path))
        return loaded if loaded is not None else config

    # Fallback : lecture JSON standard, si aucune classe Config
    # n'est fournie.
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# Génération du dataset SINN
# ============================================================

def generate_sinn_npz(
    openfoam_npz,
    output_npz=None,
    model=None,
    checkpoint=None,
    device=None,
    batch_size=65536,
):
    """
    Évalue le modèle SINN aux mêmes coordonnées et instants
    que le dataset OpenFOAM et sauvegarde les champs prédits.

    Paramètres
    ----------
    openfoam_npz : str | Path
        NPZ produit par l'extracteur OpenFOAM. Doit contenir
        au minimum 'X' et 'time'.

    output_npz : str | Path | None
        Destination du NPZ SINN. Par défaut :
        <dossier du NPZ OpenFOAM>/sinn_data.npz.

    model : SloshingPINN | None
        Modèle déjà instancié. Sinon, le modèle est créé à partir
        de la configuration et des poids du checkpoint.

    checkpoint : str | Path | None
        Chemin du checkpoint .pt. Peut contenir model_state_dict
        et config, ou seulement un state_dict.

    device : str | torch.device | None
        Appareil d'inférence.

    batch_size : int
        Nombre maximal de points traités par lot.

    Retour
    ------
    Path
        Chemin absolu du fichier NPZ créé.

    Notes
    -----
    Les coordonnées x, y et le temps sont passés au réseau dans
    leurs unités physiques, comme dans plot_fields(). Si train.py
    normalise les entrées, il faut appliquer ici exactement la
    même normalisation.
    """

    # --------------------------------------------------------
    # Chemins et données OpenFOAM
    # --------------------------------------------------------
    input_file = Path(openfoam_npz).expanduser().resolve()

    if not input_file.is_file():
        raise FileNotFoundError(
            f"Dataset OpenFOAM introuvable : {input_file}"
        )

    if output_npz is None:
        output_file = input_file.parent / "sinn_data.npz"
    else:
        output_file = Path(output_npz).expanduser().resolve()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with np.load(input_file) as data:
        if "X" not in data or "time" not in data:
            raise KeyError("Le NPZ OpenFOAM doit contenir X et time.")

        X = np.asarray(data["X"], dtype=np.float32)
        times = np.asarray(data["time"], dtype=np.float32).reshape(-1)

    if X.ndim != 3 or X.shape[-1] < 2:
        raise ValueError(
            f"X doit avoir la forme (T, N, 2 ou 3), reçu {X.shape}."
        )

    if X.shape[0] != len(times):
        raise ValueError("Le nombre d'instants ne correspond pas à X.")

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(times)):
        raise ValueError("X ou time contient des valeurs non finies.")

    if batch_size < 1:
        raise ValueError("batch_size doit être supérieur à zéro.")

    n_times, n_points, _ = X.shape

    # --------------------------------------------------------
    # Configuration et checkpoint
    # --------------------------------------------------------
    if checkpoint is not None:
        checkpoint_file = Path(checkpoint).expanduser().resolve()

        if not checkpoint_file.is_file():
            raise FileNotFoundError(
                f"Checkpoint introuvable : {checkpoint_file}"
            )

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            device = torch.device(device)

        model = torch.load(
            checkpoint_file,
            map_location=device,
            weights_only=False,
        )

    # --------------------------------------------------------
    # Initialisation ou réutilisation du modèle
    # --------------------------------------------------------
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    if model is None:
        raise ValueError(
            "Fournis soit une instance model, soit un checkpoint .pt."
        )

    model.eval()  # model.train(False)  # Désactive le dropout et la normalisation par lot
    model.to(device)

    # --------------------------------------------------------
    # Évaluation aux mêmes (t, x, y) que le dataset OpenFOAM
    # --------------------------------------------------------
    field_names = ("u", "v", "p", "alpha")

    predictions = {
        name: np.empty((n_times, n_points), dtype=np.float32)
        for name in field_names
    }

    with torch.inference_mode():
        for i, time_value in enumerate(times):
            # X[i] contient les centres de cellules à cet instant.
            # Le modèle attend uniquement x, y, t.
            xy = X[i, :, :2]

            for start in range(0, n_points, batch_size):
                end = min(start + batch_size, n_points)
                coords_xy = xy[start:end]

                t_column = np.full(
                    (end - start, 1),
                    float(time_value),
                    dtype=np.float32,
                )

                coords_np = np.concatenate(
                    [coords_xy, t_column],
                    axis=1,
                )

                coords = torch.as_tensor(
                    coords_np,
                    dtype=torch.float32,
                    device=device,
                )

                pred = model.predict_physical(coords)

                for name in field_names:
                    if name not in pred:
                        raise KeyError(
                            f"Le modèle ne renvoie pas le champ '{name}'."
                        )

                    values = (
                        pred[name]
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )

                    predictions[name][i, start:end] = values

            print(
                f"[SINN {i + 1}/{n_times}] "
                f"t = {time_value:.6g} s"
            )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------
    np.savez_compressed(
        output_file,
        time=times,
        X=X,
        u=predictions["u"],
        v=predictions["v"],
        p=predictions["p"],
        alpha_eau=predictions["alpha"],
    )

    print("\nDataset SINN généré :")
    print(f"Fichier : {output_file}")
    print(f"time    : {times.shape}")
    print(f"X       : {X.shape}")

    for name, values in predictions.items():
        print(f"{name:8s}: {values.shape}")

    return output_file
