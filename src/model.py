import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Optional
import torch
from torch import nn

import math

from config import Config
from losses import continuity_loss, volume_fraction_loss, momentum_loss


class FourierFeatures(nn.Module):
    """Encodage sinusoïdal de coordonnées adimensionnelles.

    Parameters
    ----------
    bands : tuple of int, optional
        Frequencies for the sinusoidal encoding. Default is ``(1, 2, 4, 8, 16)``.

    Returns
    -------
    out_features : int
        Number of output features : 3 (coordinates) + 6 * len(bands).
    """

    def __init__(self, bands=(1, 2, 4, 8, 16)):
        super().__init__()
        bands = torch.as_tensor(bands, dtype=torch.float32)
        if bands.ndim != 1 or bands.numel() == 0:
            raise ValueError("bands doit être une séquence non vide.")
        if torch.any(bands <= 0):
            raise ValueError("Toutes les fréquences doivent être positives.")

        self.register_buffer("bands", 2.0 * math.pi * bands)

    @property
    def out_features(self):
        """Number of output features.

        Returns
        -------
        int
            Number of output features : 3 (coordinates) + 6 * len(bands).
        """
        return 3 + 6 * self.bands.numel()

    def forward(self, coords):
        # coords : (N, 3), [x*, y*, t*]
        phase = coords.unsqueeze(-1) * self.bands
        phase = phase.flatten(start_dim=1)
        return torch.cat(
            [coords, torch.sin(phase), torch.cos(phase)],
            dim=1,
        )


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

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with keys ``(u, v, p, alpha)`` containing the
        non-dimensional predicted fields.
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
            config = Config.load_json(config_path)
            # with config_path.open("r", encoding="utf-8") as f:
            #     config = json.load(f)

        self.cfg = config

        if self.cfg is None:
            get_cfg = lambda key, default: default
        elif isinstance(self.cfg, dict):
            get_cfg = lambda key, default: self.cfg.get(key, default)
        else:
            get_cfg = lambda key, default: getattr(
                self.cfg, key, default
            )

        # Width and hidden_layers come from the configuration only
        # (no double definition between __init__ and config).
        width = int(get_cfg("hidden_width", 96))
        hidden_layers = int(get_cfg("hidden_layers", 5))
        if width < 1 or hidden_layers < 1:
            raise ValueError("width and hidden_layers must be positive.")

        self.use_fourier_features = bool(
            get_cfg("use_fourier_features", False)
        )

        if self.use_fourier_features:
            bands = get_cfg("fourier_bands", (1, 2, 4, 8, 16))
            self.encoding = FourierFeatures(bands)
            n_in = self.encoding.out_features
        else:
            self.encoding = nn.Identity()
            n_in = 3

        layers = []
        for _ in range(hidden_layers):
            layer = nn.Linear(n_in, width)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.extend([layer, nn.Tanh()])
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
            # if isinstance(state, dict) and "model_state_dict" in state:
            #     self.load_state_dict(state["model_state_dict"])
            # elif isinstance(state, nn.Module):
            #     self.load_state_dict(state.state_dict())
            # else:
            self.load_state_dict(state["model"])

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
        raw = self.head(self.backbone(self.encoding(coords_star)))
        return {"u": raw[:, 0:1], "v": raw[:, 1:2], "p": raw[:, 2:3],
                "alpha": torch.sigmoid(raw[:, 3:4])}

    def coords_star_to_si(self, coords_star: torch.Tensor) -> torch.Tensor:
        """Convert non-dimensional coordinates to SI units.

        Parameters
        ----------
        coords_star : torch.Tensor
            Non-dimensional coordinates, shape ``(N, 3)`` with columns
            ``(x*, y*, t*)``.

        Returns
        -------
        torch.Tensor
            SI coordinates, shape ``(N, 3)`` with columns ``(x, y, t)``.

        Raises
        ------
        ValueError
            If the model was built without a configuration.
        """
        if self.cfg is None:
            raise ValueError("coords_star_to_si requires the Config used to build the model.")
        if isinstance(self.cfg, dict):
            L = float(self.cfg.get("length_ref"))
            U = float(self.cfg.get("velocity_ref"))
        else:
            L = float(getattr(self.cfg, "length_ref"))
            U = float(getattr(self.cfg, "velocity_ref"))
        T = L / U
        scales = coords_star.new_tensor([L, L, T])
        return coords_star * scales

    def coords_si_to_star(self, coords_si: torch.Tensor) -> torch.Tensor:
        """Convert SI coordinates to non-dimensional.

        Parameters
        ----------
        coords_si : torch.Tensor
            SI coordinates, shape ``(N, 3)`` with columns ``(x, y, t)``.

        Returns
        -------
        torch.Tensor
            Non-dimensional coordinates, shape ``(N, 3)`` with columns
            ``(x*, y*, t*)``.

        Raises
        ------
        ValueError
            If the model was built without a configuration.
        """
        if self.cfg is None:
            raise ValueError("coords_si_to_star requires the Config used to build the model.")
        if isinstance(self.cfg, dict):
            L = float(self.cfg.get("length_ref"))
            U = float(self.cfg.get("velocity_ref"))
        else:
            L = float(getattr(self.cfg, "length_ref"))
            U = float(getattr(self.cfg, "velocity_ref"))
        T = L / U
        scales = coords_si.new_tensor([L, L, T])
        return coords_si / scales

    def _predict_physical_coords(self, coords_si: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict physical quantities from SI coordinates (N, 3).

        Called internally when the model already has properly-shaped (N,3) input.
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

    def _predict_physical_regular_grid(
        self,
        x_1d, y_1d, t_1d,
        *,
        fields=("u", "v", "p", "alpha"),
        batch_points: int = 2**16,   # ~65k points GPU par forward
        dtype=np.float32,
        verbose: bool = False,
    ):
        """Prédit sur une grille cartésienne régulière (nt, ny, nx).

        Mémoire bornée :
        - coords GPU : batch_points × 3 floats
        - sortie CPU : 4 × nt × ny × nx × 4 octets (pré-allouée)
        La boucle sur t n'alloue jamais plus que `batch_points` points GPU.
        """
        x_1d = np.asarray(x_1d, dtype=dtype).ravel()
        y_1d = np.asarray(y_1d, dtype=dtype).ravel()
        t_1d = np.asarray(t_1d, dtype=dtype).ravel()
        nx, ny, nt = x_1d.size, y_1d.size, t_1d.size

        device = next(self.parameters()).device

        # Sortie CPU pré-allouée une fois (contiguë par tranche → reshape(-1) = vue).
        result = {f: np.empty((nt, ny, nx), dtype=dtype) for f in fields}

        # (y, x) aplati, calculé UNE seule fois et partagé entre tous les t.
        Xg, Yg = np.meshgrid(x_1d, y_1d, indexing="xy")   # (ny, nx)
        x_flat = np.ascontiguousarray(Xg.ravel(), dtype=dtype)
        y_flat = np.ascontiguousarray(Yg.ravel(), dtype=dtype)
        n_xy = x_flat.size

        # Buffer GPU réutilisé à chaque chunk (évite les allocs répétées).
        coords = torch.empty(
            (min(batch_points, n_xy), 3), dtype=torch.float32, device=device
        )

        for l in range(nt):
            t_val = float(t_1d[l])
            # Vues aplaties sur la tranche (nt[l, :, :]) — écriture in-place.
            slab = {f: result[f][l].reshape(-1) for f in fields}

            for s in range(0, n_xy, batch_points):
                e = min(s + batch_points, n_xy)
                n = e - s
                cb = coords[:n]
                cb[:, 0] = torch.from_numpy(x_flat[s:e]).to(device, non_blocking=True)
                cb[:, 1] = torch.from_numpy(y_flat[s:e]).to(device, non_blocking=True)
                cb[:, 2] = t_val

                with torch.inference_mode():
                    pred = self._predict_physical_coords(cb)

                for f in fields:
                    slab[f][s:e] = pred[f].squeeze(-1).detach().cpu().numpy()

            if verbose:
                print(f"  t-step {l + 1}/{nt} terminé")

        return result

    def _predict_physical_coords_chunked(self, coords: torch.Tensor,
                                     batch_size: int = 65536):
        device = next(self.parameters()).device
        coords = coords.to(device=device, dtype=torch.float32)
        outs = None
        parts = []
        with torch.inference_mode():
            for s in range(0, coords.shape[0], batch_size):
                p = self._predict_physical_coords(coords[s:s + batch_size])
                parts.append(p)
        keys = parts[0].keys()
        return {k: torch.cat([p[k] for p in parts], dim=0) for k in keys}

    def predict_physical(self, *args, **kwargs) -> dict[str, np.ndarray]:
        """Predict physical quantities from SI or non-dimensional coordinates.

        Two call patterns are supported :

        1. **Single tensor** ``predict_physical(coords_si)``  where
           ``coords_si`` has shape ``(N, 3)`` with columns ``(x, y, t)``.
        2. **Three 1‑D sequences** ``predict_physical(x, y, t)`` where each
           argument is a 1‑D array/sequence of possibly different lengths.
           The method builds a broadcasted grid ``(nt, ny, nx)`` and returns
           fields in that shape.

        Parameters
        ----------
        *args : positional
            - One tensor/array of shape ``(N, 3)``, or
            - Three 1‑D sequences ``x, y, t``.
        **kwargs : optional
            - ``coords_si`` : tensor of shape ``(N, 3)`` (alias for positional).

        Returns
        -------
        dict of str to torch.Tensor
            Keys ``"u"``, ``"v"``, ``"p"``, ``"alpha"``.
            - If called with a single (N,3) tensor, each value has shape ``(N, 1)``.
            - If called with three 1‑D sequences, each value has shape ``(nt, ny, nx)``.

        Raises
        ------
        ValueError
            If the arguments do not match either supported pattern, or the model
            has no configuration.
        """
        # --- resolve arguments -------------------------------------------------
        # allow passing coords_si as keyword
        if "coords_si" in kwargs:
            args = (kwargs["coords_si"],) + args
        elif "x" in kwargs and "y" in kwargs and "t" in kwargs:
            args = (kwargs["x"], kwargs["y"], kwargs["t"]) + args

        # decide which pattern
        if len(args) == 1 and args[0] is not None:
            # pattern A : single (N,3) tensor
            inp = args[0]
            if isinstance(inp, torch.Tensor):
                if inp.ndim == 2 and inp.shape[1] == 3:
                    return self._predict_physical_coords(inp)
                # otherwise fall through to error
            elif isinstance(inp, (np.ndarray, list)):
                arr = np.asarray(inp)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    # convert to tensor and dispatch
                    st = torch.as_tensor(arr, dtype=torch.float32,
                                         device=next(self.parameters()).device)
                    return self._predict_physical_coords(st)
            raise ValueError(
                "predict_physical received a single argument that is not "
                "(N, 3). Expected shape (N, 3) with columns (x, y, t)."
            )

        elif len(args) >= 3:
            grid_kwargs = {
                k: kwargs[k] for k in ("fields", "batch_points", "dtype", "verbose")
                if k in kwargs
            }
            return self._predict_physical_regular_grid(
                args[0], args[1], args[2], **grid_kwargs
            )

        else:
            raise ValueError(
                "predict_physical expects either one (N, 3) tensor or three "
                "1-D sequences x, y, t."
            )

    def predict_physical_grid(
        self, *, nx, ny, nt,
        x_range, y_range, t_range,
        **kwargs,
    ):
        """Construit nx, ny, nt points uniformes sur [min, max] et prédit."""
        def lin(n, r):
            return np.linspace(float(r[0]), float(r[1]), int(n), dtype=np.float32)
        return self.predict_physical(
            lin(nx, x_range), lin(ny, y_range), lin(nt, t_range), **kwargs
        )

    def compute_residuals(self, coords_si: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute physical residuals at given collocation points.

        Parameters
        ----------
        coords_si : torch.Tensor
            SI coordinates, shape ``(N, 3)`` with columns ``(x, y, t)``.

        Returns
        -------
        dict of str to torch.Tensor
            Dictionary with keys :
            - ``"continuity"`` : continuity residual per point, shape ``(N,)``.
            - ``"vof"`` : volume-fraction residual per point, shape ``(N,)``.
            - ``"momentum_x"`` : x-momentum residual per point, shape ``(N,)``.
            - ``"momentum_y"`` : y-momentum residual per point, shape ``(N,)``.

        Notes
        -----
        The model's configuration (``self.cfg``) is used to compute the
        non-dimensional residuals.  All residuals are **signed** (not squared),
        so they can be inspected for positive/negative bias.
        The call internally enables gradients (``torch.enable_grad``) so that
        it works whether the model is in ``eval()`` or ``train()`` mode,
        and regardless of the surrounding ``torch.no_grad()`` context.
        """
        if self.cfg is None:
            raise ValueError(
                "compute_residuals requires the Config used to build the model. "
                "Provide it during model construction (state_path/config)."
            )

        # Ensure coordinates have the expected shape (N, 3).
        if coords_si.ndim != 2 or coords_si.shape[1] != 3:
            raise ValueError(
                f"coords_si must have shape (N, 3) with columns (x, y, t), "
                f"got {coords_si.shape}."
            )

        coords_star = self.coords_si_to_star(coords_si)

        c = coords_star.detach().clone().requires_grad_(True)

        # Force gradient computation even if outside a no_grad context.
        with torch.enable_grad():
            # --- continuity residual ---
            r_cont = continuity_loss(self, c, self.cfg, return_raw=True)  # (N, 1)

            # --- volume-fraction residual ---
            r_vof = volume_fraction_loss(self, c, self.cfg, return_raw=True)  # (N, 1)

            # --- momentum residuals ---
            rx, ry = momentum_loss(self, c, self.cfg, return_raw=True)  # each (N, 1)

        # Squeeze the last dimension for a clean (N,) return.
        return {
            "continuity":  r_cont.squeeze(-1),  # (N,)
            "alpha":         r_vof.squeeze(-1),   # (N,)
            "u":  rx.squeeze(-1),      # (N,)
            "v":  ry.squeeze(-1),      # (N,)
        }

    def compute_residuals_star(self, coords_star: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute physical residuals at given non-dimensional collocation points.

        Parameters
        ----------
        coords_star : torch.Tensor
            Non-dimensional coordinates, shape ``(N, 3)`` with columns
            ``(x*, y*, t*)``.

        Returns
        -------
        dict of str to torch.Tensor
            Dictionary with keys :
            - ``"continuity"`` : continuity residual per point, shape ``(N,)``.
            - ``"alpha"`` : volume-fraction residual per point, shape ``(N,)``.
            - ``"u"`` : x-momentum residual per point, shape ``(N,)``.
            - ``"v"`` : y-momentum residual per point, shape ``(N,)``.

        Notes
        -----
        The model's configuration (``self.cfg``) is used to compute the
        non-dimensional residuals.  All residuals are **signed** (not squared),
        so they can be inspected for positive/negative bias.
        The call internally enables gradients (``torch.enable_grad``) so that
        it works whether the model is in ``eval()`` or ``train()`` mode,
        and regardless of the surrounding ``torch.no_grad()`` context.
        """
        if self.cfg is None:
            raise ValueError(
                "compute_residuals_star requires the Config used to build the model. "
                "Provide it during model construction (state_path/config)."
            )

        # Ensure coordinates have the expected shape (N, 3).
        if coords_star.ndim != 2 or coords_star.shape[1] != 3:
            raise ValueError(
                f"coords_star must have shape (N, 3) with columns (x*, y*, t*), "
                f"got {coords_star.shape}."
            )

        c = coords_star.detach().clone().requires_grad_(True)

        # Force gradient computation even if outside a no_grad context.
        with torch.enable_grad():
            # --- continuity residual ---
            r_cont = continuity_loss(self, c, self.cfg, return_raw=True)  # (N, 1)

            # --- volume-fraction residual ---
            r_vof = volume_fraction_loss(self, c, self.cfg, return_raw=True)  # (N, 1)


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
