import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Optional
import torch
from torch import nn

class SloshingPINN(nn.Module):
    """Physics-informed neural network for 2D water–air sloshing.

    The forward pass takes non-dimensional coordinates
    ``[x/L_ref, y/L_ref, t/T_ref]`` and returns non-dimensional
    outputs ``(u*, v*, p*, alpha)`` where ``alpha`` is passed through
    a sigmoid activation to keep it in ``[0, 1]``.

    Parameters
    ----------
    state_path : str or pathlib.Path, optional
        Path to a ``.pt`` checkpoint file. If provided, model weights
        are loaded from it. The checkpoint may contain a
        ``model_state_dict`` key, be an ``nn.Module``, or be a raw
        ``state_dict``.
    config : str, pathlib.Path, dict, Config instance, or None, optional
        Configuration source. Can be a JSON file path, a dictionary,
        a ``Config`` instance, or ``None`` (uses default width/layers).
    """

    def __init__(
        self,
        state_path: Optional[str | Path] = None,
        config: Optional[str | Path | Any] = None,
    ):
        super().__init__()

        # Resolve configuration: JSON path, Config/dict object, or None.
        if isinstance(config, (str, Path)):
            config_path = Path(config)
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)

        self.cfg = config

        # Width and hidden_layers come from the configuration only
        # (no double definition between __init__ and config).
        if self.cfg is None:
            width = 96
            hidden_layers = 5
        elif isinstance(self.cfg, dict):
            width = int(self.cfg.get("hidden_width", 96))
            hidden_layers = int(self.cfg.get("hidden_layers", 5))
        else:
            width = int(getattr(self.cfg, "hidden_width", 96))
            hidden_layers = int(getattr(self.cfg, "hidden_layers", 5))
        if width < 1 or hidden_layers < 1:
            raise ValueError("width and hidden_layers must be positive.")

        layers, n_in = [], 3
        for _ in range(hidden_layers):
            layer = nn.Linear(n_in, width)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers += [layer, nn.Tanh()]
            n_in = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(width, 4)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Load weights from a .pt file if a path is provided.
        # Follows PyTorch recommendation: state_dict only.
        if state_path is not None:
            state_path = Path(state_path)
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                self.load_state_dict(state["model_state_dict"])
            elif isinstance(state, nn.Module):
                self.load_state_dict(state.state_dict())
            else:
                self.load_state_dict(state)

    def forward(self, coords_star: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        coords_star : torch.Tensor
            Non-dimensional coordinates, shape ``(N, 3)`` with columns
            ``(x*, y*, t*)``.

        Returns
        -------
        dict of str to torch.Tensor
            Dictionary with keys ``"u"``, ``"v"``, ``"p"``, ``"alpha"``,
            each of shape ``(N, 1)``. ``alpha`` is passed through a
            sigmoid to remain in ``[0, 1]``.
        """
        if coords_star.ndim != 2 or coords_star.shape[1] != 3:
            raise ValueError("coords_star must have shape (N,3): [x*,y*,t*].")
        raw = self.head(self.backbone(coords_star))
        return {"u": raw[:, 0:1], "v": raw[:, 1:2], "p": raw[:, 2:3],
                "alpha": torch.sigmoid(raw[:, 3:4])}

    def predict_physical(self, coords_si: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict physical quantities from SI coordinates.

        Parameters
        ----------
        coords_si : torch.Tensor
            SI coordinates, shape ``(N, 3)`` with columns ``(x, y, t)``.

        Returns
        -------
        dict of str to torch.Tensor
            Dictionary with keys ``"u"``, ``"v"``, ``"p"``, ``"alpha"``.
            ``u`` and ``v`` are in [m/s], ``p`` is in [Pa], and
            ``alpha`` is dimensionless in ``[0, 1]``.

        Raises
        ------
        ValueError
            If the model was built without a configuration.
        """
        if self.cfg is None:
            raise ValueError("predict_physical requires the Config used to build the model.")
        if isinstance(self.cfg, dict):
            L = float(self.cfg.get("length_ref"))
            U = float(self.cfg.get("velocity_ref"))
            rho = float(self.cfg.get("rho_ref"))
        else:
            L = float(getattr(self.cfg, "length_ref"))
            U = float(getattr(self.cfg, "velocity_ref"))
            rho = float(getattr(self.cfg, "rho_ref"))
        T = L / U
        scales = coords_si.new_tensor([L, L, T])
        f = self.forward(coords_si / scales)
        return {"u": U*f["u"], "v": U*f["v"], "p": rho*U**2*f["p"],
                "alpha": f["alpha"]}


# ============================================================
# SINN dataset generation
# ============================================================

def generate_sinn_npz(
    openfoam_npz,
    output_npz=None,
    model=None,
    checkpoint=None,
    device=None,
    batch_size=65536,
):
    """Evaluate the SINN model at the same coordinates and times
    as the OpenFOAM dataset and save the predicted fields.

    Parameters
    ----------
    openfoam_npz : str or pathlib.Path
        NPZ file produced by the OpenFOAM extractor. Must contain at
        least ``'X'`` and ``'time'``.
    output_npz : str, pathlib.Path, or None, optional
        Destination NPZ file. Defaults to
        ``<OpenFOAM NPZ folder>/sinn_data.npz``.
    model : SloshingPINN or None, optional
        Already instantiated model. If None, the model is created from
        the configuration and checkpoint weights.
    checkpoint : str, pathlib.Path, or None, optional
        Path to a ``.pt`` checkpoint. May contain ``model_state_dict``
        and ``config``, or only a state_dict.
    device : str or torch.device or None, optional
        Inference device. Defaults to CUDA if available, else CPU.
    batch_size : int, optional
        Maximum number of points processed per batch. Default is 65536.

    Returns
    -------
    pathlib.Path
        Absolute path to the created NPZ file.

    Notes
    -----
    Coordinates ``x``, ``y`` and time are passed to the network in
    their physical units, as in ``plot_fields()``. If ``train.py``
    normalises the inputs, the same normalisation must be applied here.
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
            raise KeyError("The OpenFOAM NPZ must contain X and time.")

        X = np.asarray(data["X"], dtype=np.float32)
        times = np.asarray(data["time"], dtype=np.float32).reshape(-1)

    if X.ndim != 3 or X.shape[-1] < 2:
        raise ValueError(
            f"X must have shape (T, N, 2 or 3), received {X.shape}."
        )

    if X.shape[0] != len(times):
        raise ValueError("Le nombre d'instants ne correspond pas à X.")

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(times)):
        raise ValueError("X ou time contient des valeurs non finies.")

    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero.")

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
