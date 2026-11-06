"""Training loop for the dimensionless laminar sloshing PINN."""
from __future__ import annotations

import csv
import json
import random
import re
import shutil
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from adaptive_weights import GradNormWeights
from config import Config
from losses import (
    continuity_loss,
    initial_loss,
    momentum_loss,
    pressure_gauge_loss,
    volume_fraction_loss,
    wall_loss,
)
from model import SloshingPINN
from physics import dimensionalize_points, nondimensionalize_points
from sampling import (
    adaptive_score,
    sample_initial,
    sample_interior,
    sample_mixed_adaptive,
    sample_walls,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_scales(cfg) -> None:
    """Validate that reference scales are present and positive.

    Parameters
    ----------
    cfg : object
        Configuration object.

    Raises
    ------
    AttributeError
        If required reference scales are missing.
    ValueError
        If any reference scale is not positive.
    """
    names = ("length_ref", "velocity_ref", "rho_ref", "mu_ref", "gravity_ref")
    missing = [n for n in names if not hasattr(cfg, n)]
    if missing:
        raise AttributeError(
            "Add these positive reference scales to config.py: "
            + ", ".join(names)
        )
    for n in names:
        if float(getattr(cfg, n)) <= 0:
            raise ValueError(f"{n} must be positive.")
    if not hasattr(cfg, "time_ref"):
        cfg.time_ref = float(cfg.length_ref) / float(cfg.velocity_ref)


def score_candidates(model, candidates, cfg):
    """Évalue la difficulté physique locale sur un pool de candidats."""
    rc = continuity_loss(model, candidates, cfg, return_per_point=True)
    rv = volume_fraction_loss(model, candidates, cfg, return_per_point=True)
    rm = momentum_loss(model, candidates, cfg, return_per_point=True)

    c = candidates.detach().clone().requires_grad_(True)
    alpha = model(c)["alpha"]
    grad_alpha = torch.autograd.grad(alpha.sum(), c, create_graph=False)[0]
    grad_alpha_norm = torch.linalg.vector_norm(grad_alpha[:, :2], dim=1)

    with torch.no_grad():
        alpha_values = model(candidates)["alpha"].reshape(-1)
        scores = adaptive_score(rc, rv, rm, alpha_values, grad_alpha_norm)

    return {
        "scores": scores.detach(),
        "continuity": rc.detach(),
        "vof": rv.detach(),
        "momentum": rm.detach(),
        "grad_alpha_norm": grad_alpha_norm.detach(),
        "alpha": alpha_values.detach(),
    }


def make_time_windows(t_min, t_max, width, overlap=0.0):
    """Create time windows for temporal segmentation training.

    Parameters
    ----------
    t_min : float
        Minimum time value [s].
    t_max : float
        Maximum time value [s].
    width : float
        Width of each time window [s]. Must be positive.
    overlap : float, optional
        Overlap between consecutive windows [s]. Must be in [0, width).
        Default is 0.0 (no overlap).

    Returns
    -------
    list of tuple
        List of (t_left, t_right) time window pairs.

    Raises
    ------
    ValueError
        If ``width <= 0`` or ``overlap < 0`` or ``overlap >= width``.
    """
    if width <= 0:
        raise ValueError("width doit être positif.")
    if overlap < 0 or overlap >= width:
        raise ValueError("overlap doit être dans [0, width).")

    step = width - overlap
    windows = []
    left = t_min

    while left < t_max:
        right = min(left + width, t_max)
        windows.append((left, right))
        if right >= t_max:
            break
        left += step

    return windows


def _split_epochs(total: int, n: int) -> list[int]:
    """Distribue ``total`` époques sur ``n`` fenêtres de manière équilibrée.

    Le reste de la division est distribué aux premières fenêtres, ce qui
    garantit ``sum(result) == total``.
    """
    if n <= 0:
        raise ValueError("n doit être strictement positif.")
    base, extra = divmod(int(total), int(n))
    return [base + (1 if i < extra else 0) for i in range(n)]


def write_history_csv(history, run_dir):
    """Write training history to a CSV file.

    Parameters
    ----------
    history : list of dict
        List of dictionaries containing training metrics. Each dictionary
        must have the same keys as the first entry.
    run_dir : str or pathlib.Path
        Directory where the history CSV file will be saved.

    Raises
    ------
    ValueError
        If dictionaries have inconsistent keys.
    """
    if not history:
        return

    fieldnames = list(history[0].keys())
    for i, row in enumerate(history):
        if list(row.keys()) != fieldnames:
            raise ValueError(
                f"Inconsistent keys at row {i}: "
                f"{list(row.keys())} != {fieldnames}"
            )

    path = Path(run_dir) / "history.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(
    path,
    model,
    optimizer,
    weight_balancer,
    weights,
    window_idx,
    local_epoch,
    global_epoch,
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "weight_balancer": (
            weight_balancer.state_dict() if weight_balancer is not None else None
        ),
        "weights": weights,
        "window_idx": window_idx,
        "local_epoch": local_epoch,
        "global_epoch": global_epoch,
        # États aléatoires pour une reprise aussi fidèle que possible
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, weight_balancer, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if weight_balancer is not None and checkpoint["weight_balancer"] is not None:
        weight_balancer.load_state_dict(checkpoint["weight_balancer"], device)

    random.setstate(checkpoint["python_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    
    torch_rng = checkpoint["torch_rng"]
    if not isinstance(torch_rng, torch.Tensor):
        torch_rng = torch.tensor(torch_rng, dtype=torch.uint8)
    else:
        torch_rng = torch_rng.detach().cpu().to(torch.uint8)
    torch.set_rng_state(torch_rng)

    if torch.cuda.is_available() and checkpoint.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(
            [s.cpu() if isinstance(s, torch.Tensor) else s for s in checkpoint["cuda_rng"]]
        )

    return (
        checkpoint["weights"],
        checkpoint["window_idx"],
        checkpoint["local_epoch"],
        checkpoint["global_epoch"],
    )


def _backup_resume_files(run_dir: Path, history_path: Path, ckpt_path: Path) -> Path:
    """Crée une sauvegarde numérotée de ``history.json`` + ``checkpoint_final.pt``.

    Retourne le chemin de la copie de checkpoint créée.
    """
    run_dir = Path(run_dir)
    pattern_ckpt = re.compile(r"^checkpoint(\d+)\.pt$")
    pattern_hist = re.compile(r"^history(\d+)\.json$")
    nums_ckpt, nums_hist = [], []
    for p in run_dir.iterdir():
        m = pattern_ckpt.match(p.name)
        if m:
            nums_ckpt.append(int(m.group(1)))
        m = pattern_hist.match(p.name)
        if m:
            nums_hist.append(int(m.group(1)))
    next_idx = max(max(nums_ckpt, default=0), max(nums_hist, default=0)) + 1

    backup_ckpt = run_dir / f"checkpoint{next_idx:02d}.pt"
    backup_hist = run_dir / f"history{next_idx:02d}.json"

    shutil.copy2(str(ckpt_path), str(backup_ckpt))
    shutil.copy2(str(history_path), str(backup_hist))
    return backup_ckpt


def train(
    cfg: Config | None = None,
    output_dir: str | Path = "runs",
    model: SloshingPINN | None = None,
    state_path: str | Path | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    epochs: int | None = None,
    lbfgs_epochs: int | None = None,
    model_name: str | None = None,
    resume: bool = False,
):
    """Train the dimensionless laminar sloshing PINN.

    Parameters
    ----------
    cfg : Config, optional
        Configuration object. If None, a default ``Config()`` is used.
    output_dir : str or pathlib.Path, optional
        Root output directory. A timestamped subdirectory is created
        inside it. Default is ``"runs"``.
    model : SloshingPINN, optional
        Pre-instantiated model. If None, a new model is built from
        ``cfg`` and ``state_path``.
    state_path : str or pathlib.Path, optional
        Path to a ``.pt`` checkpoint to initialise model weights.
        Ignored if ``model`` is provided.
    optimizer : torch.optim.Optimizer, optional
        Optimiser for training. If None, Adam is used with
        ``cfg.learning_rate``.
    epochs : int, optional
        Number of training epochs. If None, ``cfg.epochs`` is used.
    lbfgs_epochs : int, optional
        Number of L-BFGS passes at the end. If None, ``cfg.lbfgs_epochs``
        is used.
    model_name : str, optional
        Name of the model / folder under which results are saved.
        If provided, the run directory is ``<output_dir>/<model_name>``
        instead of the default ``<output_dir>/sloshing_YYYYMMDD_HHMMSS``.
        If omitted and ``cfg`` carries a ``model_name`` attribute, that
        value is used. Otherwise the default timestamped name is kept.
    resume : bool, optional
        If True, resume training from a previously saved state in
        ``run_dir`` (load ``history.json`` and ``checkpoint_final.pt``,
        back up existing files, and continue where training left off).
        Default is ``False``.

    Returns
    -------
    model : SloshingPINN
        Trained model.
    history : list of dict
        Training history with one entry per epoch containing keys
        ``epoch``, ``total``, ``continuity``, ``vof``, ``momentum_x``,
        ``momentum_y``, ``initial_u``, ``initial_v``, ``initial_alpha``,
        ``wall_u``, ``wall_v``, ``pressure_gauge``, ``window``,
        ``t_left``, ``t_right``.
    run_dir : pathlib.Path
        Directory where checkpoints and logs are saved.
    """
    # --- Résolution du répertoire de run -----------------------------------
    chosen = model_name
    if not chosen and cfg is not None:
        chosen = getattr(cfg, "model_name", None)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if chosen:
        run_dir = root / str(chosen)
    else:
        run_dir = root / ("sloshing_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Résolution de la configuration -----------------------------------
    if cfg is None and model is not None:
        cfg = getattr(model, "cfg", None)
    cfg = cfg or Config()

    if hasattr(cfg, "validate"):
        cfg.validate()
    _validate_scales(cfg)
    set_seed(int(cfg.seed))
    device = torch.device(cfg.device)

    # --- Modèle ------------------------------------------------------------
    if model is None:
        if state_path is not None:
            model = SloshingPINN(config=cfg, state_path=state_path)
        else:
            model = SloshingPINN(config=cfg)
    model = model.to(device)
    cfg.save_json(run_dir / "config.json")

    # --- Optimiseur --------------------------------------------------------
    adam = optimizer or torch.optim.Adam(
        model.parameters(), lr=float(cfg.learning_rate)
    )

    # --- Fenêtres temporelles ---------------------------------------------
    if cfg.use_temporal_segmentation:
        windows = make_time_windows(
            cfg.t_min,
            cfg.t_max,
            cfg.temporal_window_size,
            cfg.temporal_overlap,
        )
    else:
        windows = [(cfg.t_min, cfg.t_max)]

    if not windows:
        raise ValueError(
            "Aucune fenêtre temporelle générée : vérifier t_min / t_max / "
            "temporal_window_size dans la configuration."
        )

    # --- Budget d'époques : distribué équitablement entre les fenêtres -----
    total_budget = int(epochs if epochs is not None else cfg.epochs)
    window_budgets = _split_epochs(total_budget, len(windows))
    n_lbfgs_epochs = int(
        lbfgs_epochs if lbfgs_epochs is not None else cfg.lbfgs_epochs
    )

    # Print configuration parameters once at the start
    print(
        f"Training config:\n\tepochs={total_budget}\n"
        f"\tfourier features={cfg.use_fourier_features}\n"
        f"\tadaptive sampling={cfg.use_adaptive_sampling}\n"
        f"\tadaptive loss weights={cfg.use_adaptive_loss_weights}\n"
        f"\tuse lbfgs={cfg.use_lbfgs}\n"
        f"\ttemporal segmentation={cfg.use_temporal_segmentation}\n"
        f"\twindows={len(windows)}\n"
        f"\twindow_budgets={window_budgets}\n"
        f"\tlearning_rate={float(cfg.learning_rate):.2e}\n"
        f"\tdevice={device}\n"
        f"\tseed={int(cfg.seed)}"
    )

    # --- Poids adaptatifs --------------------------------------------------
    weight_names = (
        "continuity",
        "vof",
        "momentum_x",
        "momentum_y",
        "wall_u",
        "wall_v",
    )
    weight_balancer = GradNormWeights(
        weight_names,
        alpha=getattr(cfg, "gradnorm_alpha", 0.5),
        lr=getattr(cfg, "gradnorm_lr", 1e-3),
        min_weight=cfg.min_adaptive_weight,
        max_weight=cfg.max_adaptive_weight,
        shared_params=lambda m: list(m.backbone[-2].parameters()),
    )

    # Poids fixes par défaut (mapping des nouveaux termes vers les vieux poids cfg)
    weights = {
        "continuity": cfg.w_continuity,
        "vof": cfg.w_vof,
        "momentum_x": cfg.w_momentum,
        "momentum_y": cfg.w_momentum,
        "initial_u": cfg.w_initial,
        "initial_v": cfg.w_initial,
        "initial_alpha": cfg.w_initial,
        "wall_u": cfg.w_wall,
        "wall_v": cfg.w_wall,
        "pressure_gauge": cfg.w_pressure_gauge,
    }

    # --- Reprise éventuelle ------------------------------------------------
    start_epoch = 0
    resumed = False
    if resume:
        history_path = run_dir / "history.json"
        ckpt_path = run_dir / "checkpoint_final.pt"
        if history_path.is_file() and ckpt_path.is_file():
            with open(history_path, "r", encoding="utf-8") as f:
                existing_history = json.load(f)

            if existing_history:
                start_epoch = max(int(r["epoch"]) for r in existing_history)

            backup_ckpt = _backup_resume_files(run_dir, history_path, ckpt_path)
            print(
                f"Resuming training from epoch {start_epoch} "
                f"(backup: {backup_ckpt.name})"
            )

            try:
                load_checkpoint(ckpt_path, model, adam, weight_balancer, device)
            except Exception as exc:  # pragma: no cover - fallback legacy
                print(
                    f"Warning: load_checkpoint failed ({exc}); "
                    f"falling back to model-only load."
                )
                model.load_state_dict(
                    torch.load(str(ckpt_path), map_location=device)
                )

            resumed = True
        else:
            print(
                "Resume requested but no previous state found; "
                "starting from scratch."
            )

    history = existing_history if resumed else []
    adaptive_weight_history: list[dict] = []
    replay_si = None

    # Historisation des points adaptatifs et de la perte locale.
    use_adaptive_history = bool(getattr(cfg, "use_adaptive_sampling", False))
    adaptive_history_epochs: list[int] = []
    adaptive_history_windows: list[int] = []
    adaptive_history_tleft: list[float] = []
    adaptive_history_tright: list[float] = []
    adaptive_points: list = []
    adaptive_scores: list = []
    adaptive_continuity: list = []
    adaptive_vof: list = []
    adaptive_momentum: list = []
    adaptive_alpha: list = []

    gauge_si = torch.tensor(
        [
            [
                0.5 * (cfg.x_min + cfg.x_max),
                0.5 * (cfg.y_min + cfg.y_max),
                cfg.t_min,
            ]
        ],
        dtype=torch.float32,
        device=device,
    )
    gauge = nondimensionalize_points(gauge_si, cfg)

    # Valeurs de secours pour la sauvegarde finale
    last_window_idx = 0
    last_local_epoch = 0
    global_epoch = start_epoch

    # --- Boucle d'entraînement --------------------------------------------
    for window_id, (t_left, t_right) in enumerate(windows):
        window_offset = sum(window_budgets[:window_id])
        n_window_epochs = window_budgets[window_id]
        already_done = max(0, min(n_window_epochs, start_epoch - window_offset))

        if cfg.use_adaptive_loss_weights and window_id > 0:
            weight_balancer.reset_baselines()

        window_cfg = replace(cfg, t_min=t_left, t_max=t_right)

        # Fenêtre entièrement terminée : on ne saute que l'entraînement,
        # mais on rafraîchit les points de replay pour la fenêtre suivante.
        if already_done >= n_window_epochs:
            replay_si = sample_interior(
                cfg.n_collocation, window_cfg, device=device
            ).detach()
            continue

        desc = (
            f"Training on window {window_id + 1}/{len(windows)}, "
            f"t=[{t_left}, {t_right}]"
        ) if cfg.use_temporal_segmentation else f"Training for t=[{t_left}, {t_right}]"
        pbar = tqdm(
            total=n_window_epochs,
            initial=already_done,
            desc=desc,
            leave=True,
            dynamic_ncols=True,
        )

        cached_interior_si = None
        cached_walls_si = None
        cached_initial_si = None

        include_top = not bool(getattr(cfg, "top_is_open", True))

        for local_epoch in range(n_window_epochs):
            if local_epoch < already_done:
                continue

            global_epoch = window_offset + local_epoch + 1
            last_window_idx = window_id
            last_local_epoch = local_epoch
            model.train()

            sampling_refreshed = False

            # ---------------- Rafraîchissement adaptatif ------------------
            if cfg.use_adaptive_sampling and (
                cached_interior_si is None
                or local_epoch % cfg.sampling_update_interval == 0
            ):
                sampling_refreshed = True
                model.eval()
                with torch.enable_grad():
                    # Points intérieurs
                    candidates_si = sample_interior(
                        cfg.n_candidate_points, window_cfg, device=device
                    )
                    candidates_nd = nondimensionalize_points(candidates_si, cfg)
                    score_result = score_candidates(model, candidates_nd, cfg)

                    scores = score_result["scores"]
                    selected_nd, selected_idx = sample_mixed_adaptive(
                        candidates_nd,
                        scores,
                        cfg.n_collocation,
                        adaptive_fraction=cfg.adaptive_fraction,
                        return_indices=True,
                    )
                    cached_interior_si = dimensionalize_points(
                        selected_nd, cfg
                    ).detach()

                    # Points de paroi
                    candidates_walls_si = sample_walls(
                        cfg.n_candidate_points,
                        window_cfg,
                        device=device,
                        include_top=include_top,
                    )
                    candidates_walls_nd = nondimensionalize_points(
                        candidates_walls_si, cfg
                    )
                    with torch.no_grad():
                        wall_pred = model(candidates_walls_nd)
                        wall_score = (
                            wall_pred["u"].pow(2) + wall_pred["v"].pow(2)
                        ).reshape(-1)
                    selected_walls_nd, _ = sample_mixed_adaptive(
                        candidates_walls_nd,
                        wall_score,
                        cfg.n_wall,
                        adaptive_fraction=cfg.adaptive_fraction,
                        return_indices=True,
                    )
                    cached_walls_si = dimensionalize_points(
                        selected_walls_nd, cfg
                    ).detach()

                    # Conditions initiales (fenêtre 0 uniquement)
                    if window_id == 0:
                        candidates_init_si = sample_initial(
                            cfg.n_candidate_points, window_cfg, device=device
                        )
                        candidates_init_nd = nondimensionalize_points(
                            candidates_init_si, cfg
                        )
                        with torch.no_grad():
                            init_pred = model(candidates_init_nd)
                            target_alpha = torch.sigmoid(
                                (
                                    cfg.initial_level / cfg.length_ref
                                    - candidates_init_nd[:, 1:2]
                                )
                                / (cfg.interface_smoothing / cfg.length_ref)
                            )
                            init_score = (
                                init_pred["u"].pow(2)
                                + init_pred["v"].pow(2)
                                + (init_pred["alpha"] - target_alpha).pow(2)
                            ).reshape(-1)
                        selected_init_nd, _ = sample_mixed_adaptive(
                            candidates_init_nd,
                            init_score,
                            cfg.n_initial,
                            adaptive_fraction=cfg.adaptive_fraction,
                            return_indices=True,
                        )
                        cached_initial_si = dimensionalize_points(
                            selected_init_nd, cfg
                        ).detach()

                model.train()

                if use_adaptive_history:
                    adaptive_history_epochs.append(global_epoch)
                    adaptive_history_windows.append(window_id)
                    adaptive_history_tleft.append(float(t_left))
                    adaptive_history_tright.append(float(t_right))
                    adaptive_points.append(
                        selected_nd.detach().cpu().numpy().copy()
                    )
                    adaptive_scores.append(
                        scores[selected_idx].detach().cpu().numpy().copy()
                    )
                    adaptive_continuity.append(
                        score_result["continuity"][selected_idx]
                        .detach().cpu().numpy().copy()
                    )
                    adaptive_vof.append(
                        score_result["vof"][selected_idx]
                        .detach().cpu().numpy().copy()
                    )
                    adaptive_momentum.append(
                        score_result["momentum"][selected_idx]
                        .detach().cpu().numpy().copy()
                    )
                    adaptive_alpha.append(
                        score_result["alpha"][selected_idx]
                        .detach().cpu().numpy().copy()
                    )

            # ---------------- Échantillonnage intérieur -------------------
            if cfg.use_adaptive_sampling and cached_interior_si is not None:
                interior_si = cached_interior_si.clone()
            else:
                interior_si = sample_interior(
                    cfg.n_collocation, window_cfg, device=device
                )

            # Replay temporel : maintenir des contraintes sur le passé.
            if replay_si is not None and replay_si.numel() > 0:
                n_replay = min(
                    replay_si.shape[0],
                    int(cfg.temporal_replay_fraction * cfg.n_collocation),
                )
                replay_idx = torch.randperm(
                    replay_si.shape[0], device=device
                )[:n_replay]
                interior_si = torch.cat(
                    [interior_si, replay_si[replay_idx]], dim=0
                )

            # ---------------- Échantillonnage parois ----------------------
            if cfg.use_adaptive_sampling and cached_walls_si is not None:
                walls_si = cached_walls_si.clone()
            else:
                walls_si = sample_walls(
                    cfg.n_wall, window_cfg, device=device, include_top=include_top
                )

            # ---------------- Échantillonnage conditions initiales --------
            if window_id == 0:
                if cfg.use_adaptive_sampling and cached_initial_si is not None:
                    initial_si = cached_initial_si.clone()
                else:
                    initial_si = sample_initial(
                        cfg.n_initial, window_cfg, device=device
                    )
            else:
                initial_si = None

            interior = nondimensionalize_points(interior_si, cfg)
            walls = nondimensionalize_points(walls_si, cfg)

            # ---------------- Calcul des pertes ---------------------------
            lc = continuity_loss(model, interior, cfg)
            lv = volume_fraction_loss(model, interior, cfg)
            lmx, lmy = momentum_loss(model, interior, cfg)
            lwu, lwv = wall_loss(model, walls, cfg)

            if initial_si is not None:
                initial = nondimensionalize_points(initial_si, cfg)
                liu, liv, lia = initial_loss(model, initial, cfg)
            else:
                zero = lc.new_zeros(())
                liu, liv, lia = zero, zero, zero

            lg = pressure_gauge_loss(model, gauge, cfg)

            loss_terms = {
                "continuity": lc,
                "vof": lv,
                "momentum_x": lmx,
                "momentum_y": lmy,
                "initial_u": liu,
                "initial_v": liv,
                "initial_alpha": lia,
                "wall_u": lwu,
                "wall_v": lwv,
                "pressure_gauge": lg,
            }

            # Poids fixes (réinitialisés à chaque époque)
            weights = {
                "continuity": cfg.w_continuity,
                "vof": cfg.w_vof,
                "momentum_x": cfg.w_momentum,
                "momentum_y": cfg.w_momentum,
                "initial_u": cfg.w_initial,
                "initial_v": cfg.w_initial,
                "initial_alpha": cfg.w_initial,
                "wall_u": cfg.w_wall,
                "wall_v": cfg.w_wall,
                "pressure_gauge": cfg.w_pressure_gauge,
            }

            # Option adaptative : mise à jour périodique des poids.
            if cfg.use_adaptive_loss_weights and (
                local_epoch % cfg.weight_update_interval == 0
                and not sampling_refreshed
                and local_epoch > 0
            ):
                weights.update(weight_balancer.update(loss_terms, model))

            if cfg.use_adaptive_loss_weights:
                adaptive_weight_history.append(
                    {
                        "epoch": global_epoch,
                        "window": window_id,
                        **{name: float(weights[name]) for name in weight_names},
                    }
                )

            total = sum(weights[k] * loss_terms[k] for k in loss_terms)

            adam.zero_grad(set_to_none=True)
            total.backward()
            adam.step()

            row = {
                "epoch": global_epoch,
                "window": window_id,
                "t_left": float(t_left),
                "t_right": float(t_right),
                "total": float(total.detach().cpu()),
                **{
                    key: float(value.detach().cpu())
                    for key, value in loss_terms.items()
                },
            }
            history.append(row)

            pbar.update(1)
            pbar.set_postfix(
                total=f"{float(total.detach().cpu()):.4e}",
                continuity=f"{float(loss_terms['continuity'].detach().cpu()):.4e}",
                alpha=f"{float(loss_terms['vof'].detach().cpu()):.4e}",
                u=f"{float(loss_terms['momentum_x'].detach().cpu()):.4e}",
                v=f"{float(loss_terms['momentum_y'].detach().cpu()):.4e}",
                lr=f"{adam.param_groups[0]['lr']:.2e}",
                window=f"w{window_id}",
            )

            if global_epoch % cfg.checkpoint_interval == 0:
                save_checkpoint(
                    path=run_dir / f"checkpoint_epoch{global_epoch:04d}.pt",
                    model=model,
                    optimizer=adam,
                    weight_balancer=weight_balancer,
                    weights=weights,
                    window_idx=window_id,
                    local_epoch=local_epoch,
                    global_epoch=global_epoch,
                )

        pbar.close()

        # Points de replay pour la fenêtre suivante : plusieurs instants
        # répartis dans la fenêtre, meilleurs qu'un seul point de raccord.
        replay_si = sample_interior(
            cfg.n_collocation, window_cfg, device=device
        ).detach()

    # --- L-BFGS sur points fixes ------------------------------------------
    if cfg.use_lbfgs and n_lbfgs_epochs > 0:
        fixed_interior = nondimensionalize_points(
            sample_interior(cfg.n_collocation, cfg, device=device), cfg
        )
        fixed_walls = nondimensionalize_points(
            sample_walls(
                cfg.n_wall,
                cfg,
                device=device,
                include_top=not bool(getattr(cfg, "top_is_open", True)),
            ),
            cfg,
        )
        fixed_initial = nondimensionalize_points(
            sample_initial(cfg.n_initial, cfg, device=device), cfg
        )

        frozen_weights = dict(weights)
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=cfg.lbfgs_lr,
            max_iter=cfg.lbfgs_max_iter,
            history_size=cfg.lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            lc = continuity_loss(model, fixed_interior, cfg)
            lv = volume_fraction_loss(model, fixed_interior, cfg)
            lmx, lmy = momentum_loss(model, fixed_interior, cfg)
            lwu, lwv = wall_loss(model, fixed_walls, cfg)
            liu, liv, lia = initial_loss(model, fixed_initial, cfg)
            lg = pressure_gauge_loss(model, gauge, cfg)

            total = (
                frozen_weights["continuity"] * lc
                + frozen_weights["vof"] * lv
                + frozen_weights["momentum_x"] * lmx
                + frozen_weights["momentum_y"] * lmy
                + frozen_weights["initial_u"] * liu
                + frozen_weights["initial_v"] * liv
                + frozen_weights["initial_alpha"] * lia
                + frozen_weights["wall_u"] * lwu
                + frozen_weights["wall_v"] * lwv
                + frozen_weights["pressure_gauge"] * lg
            )
            total.backward()
            return total

        for _ in range(n_lbfgs_epochs):
            model.train()
            lbfgs.step(closure)

    # --- Sauvegarde historique adaptatif ----------------------------------
    if use_adaptive_history and adaptive_points:
        points_arr = np.stack(adaptive_points, axis=0)          # (M, N, 3)
        scores_arr = np.stack(adaptive_scores, axis=0)          # (M, N)
        cont_arr = np.stack(adaptive_continuity, axis=0)
        vof_arr = np.stack(adaptive_vof, axis=0)
        mom_arr = np.stack(adaptive_momentum, axis=0)
        alpha_arr = np.stack(adaptive_alpha, axis=0)
        np.savez_compressed(
            run_dir / "adaptive_history.npz",
            epochs=np.asarray(adaptive_history_epochs, dtype=int),
            windows=np.asarray(adaptive_history_windows, dtype=int),
            t_left=np.asarray(adaptive_history_tleft, dtype=float),
            t_right=np.asarray(adaptive_history_tright, dtype=float),
            points=points_arr,
            scores=scores_arr,
            continuity=cont_arr,
            vof=vof_arr,
            momentum=mom_arr,
            alpha=alpha_arr,
        )
        summary = [
            {
                "epoch": int(e),
                "window": int(w),
                "t_left": float(tl),
                "t_right": float(tr),
                "n_points": int(points_arr[i].shape[0]),
                "mean_score": float(scores_arr[i].mean()),
                "mean_continuity": float(cont_arr[i].mean()),
                "mean_vof": float(vof_arr[i].mean()),
                "mean_momentum": float(mom_arr[i].mean()),
            }
            for i, (e, w, tl, tr) in enumerate(
                zip(
                    adaptive_history_epochs,
                    adaptive_history_windows,
                    adaptive_history_tleft,
                    adaptive_history_tright,
                )
            )
        ]
        (run_dir / "adaptive_history.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(
            f"Adaptive history saved: {len(adaptive_points)} updates, "
            f"{points_arr.shape} points."
        )

    # --- Sauvegarde historique principal ----------------------------------
    write_history_csv(history, run_dir)
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    # Snapshot numéroté des époques réellement nouvelles lors de cette session.
    if resumed:
        new_history = [r for r in history if int(r["epoch"]) > start_epoch]
        if new_history:
            pattern_ckpt = re.compile(r"^checkpoint(\d+)\.pt$")
            pattern_hist = re.compile(r"^history(\d+)\.json$")
            nums_ckpt, nums_hist = [], []
            for p in run_dir.iterdir():
                m = pattern_ckpt.match(p.name)
                if m:
                    nums_ckpt.append(int(m.group(1)))
                m = pattern_hist.match(p.name)
                if m:
                    nums_hist.append(int(m.group(1)))
            next_idx = max(
                max(nums_ckpt, default=0), max(nums_hist, default=0)
            ) + 1

            new_hist_file = run_dir / f"history{next_idx:02d}.json"
            new_hist_file.write_text(
                json.dumps(new_history, indent=2), encoding="utf-8"
            )
            # Le checkpoint correspondant est checkpoint_final.pt qui sera
            # écrit ci-dessous ; on en garde une copie numérotée cohérente.
            # (Effectué après sauvegarde du checkpoint final.)
            pending_ckpt_backup = run_dir / f"checkpoint{next_idx:02d}.pt"
        else:
            pending_ckpt_backup = None
    else:
        pending_ckpt_backup = None

    # --- Poids adaptatifs (seulement si activés) --------------------------
    if cfg.use_adaptive_loss_weights and adaptive_weight_history:
        (run_dir / "adaptive_weights.json").write_text(
            json.dumps(adaptive_weight_history, indent=2), encoding="utf-8"
        )
        final_adaptive_weights = {
            name: float(weights[name]) for name in weight_names
        }
        (run_dir / "adaptive_weights_final.json").write_text(
            json.dumps(final_adaptive_weights, indent=2), encoding="utf-8"
        )

    print(f"Training finished. Results saved to: {run_dir}")

    # --- Checkpoint final -------------------------------------------------
    save_checkpoint(
        path=run_dir / "checkpoint_final.pt",
        model=model,
        optimizer=adam,
        weight_balancer=weight_balancer,
        weights=weights,
        window_idx=last_window_idx,
        local_epoch=last_local_epoch,
        global_epoch=global_epoch,
    )

    if pending_ckpt_backup is not None:
        shutil.copy2(
            str(run_dir / "checkpoint_final.pt"), str(pending_ckpt_backup)
        )

    return model, history, run_dir


if __name__ == "__main__":
    train()