"""Training loop for the dimensionless laminar sloshing PINN."""
from __future__ import annotations
import json, random
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from config import Config
from model import SloshingPINN
from sampling import sample_initial, sample_interior, sample_walls, adaptive_score, sample_mixed_adaptive
from physics import nondimensionalize_points, dimensionalize_points
from losses import (continuity_loss, volume_fraction_loss, momentum_loss,
                    initial_loss, wall_loss, pressure_gauge_loss)


import csv
import time
from tqdm import trange, tqdm
from dataclasses import replace

from adaptive_weights import GradNormWeights



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

def config_dict(cfg) -> dict:
    """Convert a configuration object to a dictionary.

    Parameters
    ----------
    cfg : object
        Configuration object (dataclass or object with ``__dict__``).

    Returns
    -------
    dict
        Dictionary representation of the configuration.

    Raises
    ------
    TypeError
        If ``cfg`` is neither a dataclass nor exposes ``__dict__``.
    """
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    raise TypeError("Config must be a dataclass or expose __dict__.")

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
    rc = continuity_loss(
        model, candidates, cfg, return_per_point=True
    )
    rv = volume_fraction_loss(
        model, candidates, cfg, return_per_point=True
    )
    rm = momentum_loss(
        model, candidates, cfg, return_per_point=True
    )

    c = candidates.detach().clone().requires_grad_(True)
    alpha = model(c)["alpha"]
    grad_alpha = torch.autograd.grad(
        alpha.sum(), c, create_graph=False
    )[0]
    grad_alpha_norm = torch.linalg.vector_norm(
        grad_alpha[:, :2], dim=1
    )

    with torch.no_grad():
        alpha_values = model(candidates)["alpha"].reshape(-1)
        scores = adaptive_score(
            rc, rv, rm, alpha_values, grad_alpha_norm
        )

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



def write_history_csv(history, run_dir):
    """Write training history to a CSV file.

    Parameters
    ----------
    history : list of dict
        List of dictionaries containing training metrics. Each dictionary
        should have the same keys (e.g., ``epoch``, ``continuity``, etc.).
    run_dir : str or pathlib.Path
        Directory where the history CSV file will be saved.

    Raises
    ------
    ValueError
        If ``history`` is empty or dictionaries have inconsistent keys.
    """
    if not history:
        return

    path = Path(run_dir) / "history.csv"
    fieldnames = list(history[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _next_numbered_file(run_dir, prefix, suffix):
    """Return the next available index for numbered files like prefixNNsuffix.

    Scans ``run_dir`` for files whose name matches ``prefix`` + digits + ``suffix``
    and returns ``max(existing) + 1`` (or 1 when none exist). Used to avoid
    overwriting previously created checkpoint/history backups on resume.
    """
    import re
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    nums = []
    for path in Path(run_dir).iterdir():
        m = pattern.match(path.name)
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


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
    optimizer : torch.optim.Optimizer, optional
        Optimiser for training. If None, Adam is used with
        ``cfg.learning_rate``.
    epochs : int, optional
        Number of training epochs. If None, ``cfg.epochs`` is used.
    model_name : str, optional
        Name of the model / folder under which results are saved.
        If provided, the run directory is ``<output_dir>/<model_name>``
        instead of the default ``<output_dir>/sloshing_YYYYMMDD_HHMMSS``.
        If omitted and ``cfg`` carries a ``model_name`` attribute, that
        value is used.  Otherwise the default timestamped name is kept.
    resume : bool, optional
        If True, resume training from a previously saved state in
        ``run_dir`` (load ``history.json`` and ``checkpoint_final.pt``,
        back up existing files, and continue from the last completed epoch).
        Default is ``False``.

    Returns
    -------
    model : SloshingPINN
        Trained model.
    history : list of dict
        Training history with one entry per epoch containing keys
        ``epoch``, ``total``, ``continuity``, ``vof``, ``momentum``,
        ``initial``, ``wall``, ``pressure_gauge``.
    run_dir : pathlib.Path
        Directory where checkpoints and logs are saved.
    """
    # Resolve model_name from argument, then cfg, then fall back to timestamp
    chosen = model_name
    if not chosen:
        chosen = getattr(cfg, "model_name", None) if cfg else None
    if chosen:
        root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
        run_dir = root / str(chosen)
    else:
        root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
        run_dir = root / ("sloshing_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resolve cfg: prefer provided cfg, then fall back to model.cfg, then default Config()
    if cfg is None and model is not None:
        cfg = getattr(model, "cfg", None)
    cfg = cfg or Config()
    if hasattr(cfg, "validate"): cfg.validate()
    _validate_scales(cfg)
    set_seed(int(cfg.seed))
    device = torch.device(cfg.device)
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)

    if model is None:
        model = SloshingPINN(config=cfg, state_path=state_path).to(device)
        cfg.save_json(run_dir / "config.json")
    else:
        model = model.to(device)
        cfg.save_json(run_dir / "config.json")

    
    # Conserver une instance unique du modèle et un seul optimiseur Adam.
    adam = optimizer or torch.optim.Adam(
        model.parameters(), lr=float(cfg.learning_rate)
    )

    if cfg.use_temporal_segmentation:
        windows = make_time_windows(
            cfg.t_min,
            cfg.t_max,
            cfg.temporal_window_size,
            cfg.temporal_overlap,
        )
    else:
        windows = [(cfg.t_min, cfg.t_max)]

    epochs_total = int(epochs if epochs is not None else cfg.epochs)
    epochs_per_window = max(1, epochs_total // len(windows))

    # Print configuration parameters once at the start
    print(f"Training config:\n\tepochs={epochs_total}\n"
          f"\tfourier features={cfg.use_fourier_features}\n"
          f"\tadaptive sampling={cfg.use_adaptive_sampling}\n"
          f"\tadaptive loss weights={cfg.use_adaptive_loss_weights}\n"
          f"\tuse lbfgs={cfg.use_lbfgs}\n"
          f"\ttemporal segmentation={cfg.use_temporal_segmentation}\n"
          f"\twindows={len(windows)}\n"
          f"\tepochs_per_window={epochs_per_window}\n"
          f"\tlearning_rate={float(cfg.learning_rate):.2e}\n"
          f"\tdevice={device}\n"
          f"\tseed={int(cfg.seed)}")

    
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
        # Use the last linear layer of the backbone as shared parameters
        # (recommended in the GradNorm paper for efficiency and representativeness).
        shared_params=lambda m: list(m.backbone[-2].parameters()),
    )  # end of weight_balancer setup

    # === Resume from a previous training state (if requested) ===
    start_epoch = 0
    resumed = False
    if resume:
        history_path = run_dir / "history.json"
        ckpt_path = run_dir / "checkpoint_final.pt"
        if history_path.is_file() and ckpt_path.is_file():
            # Load existing history to find how many epochs have been completed
            with open(history_path, "r", encoding="utf-8") as f:
                existing_history = json.load(f)
            # Count completed epochs from history
            if existing_history:
                start_epoch = max(int(r["epoch"]) for r in existing_history)
            resumed = True
            global_epoch = start_epoch
            # Determine if this is the 1st resume (no numbered backups exist yet)
            # or a subsequent resume.
            ckpt01 = run_dir / "checkpoint01.pt"
            hist01 = run_dir / "history01.json"
            is_first_resume = not (ckpt01.is_file() and hist01.is_file())
            if is_first_resume:
                # 1st resume: copy checkpoint and rename history, then load
                import shutil
                # Copy (not move) so original remains for model loading
                shutil.copy2(str(ckpt_path), str(ckpt01))
                # Rename history.json to history01.json
                hist_backup = run_dir / "history01.json"
                shutil.move(str(history_path), str(hist_backup))
                # Load existing history from the backup
                with open(str(hist_backup), "r", encoding="utf-8") as f:
                    existing_history = json.load(f)
                print(f"Resuming training from epoch {start_epoch} (1st resume)")
                print(f"Checkpoint backed up to: {ckpt01}")
            else:
                # Subsequent resume: just load existing history from history.json
                with open(str(history_path), "r", encoding="utf-8") as f:
                    existing_history = json.load(f)
                print(f"Resuming training from epoch {start_epoch} (subsequent resume)")
            # Load model weights from the original checkpoint
            model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
        else:
            print("Resume requested but no previous state found; starting from scratch.")
            existing_history = []
    else:
        existing_history = []

    # Initialize history: use existing if resuming, otherwise empty list
    if resumed:
        history = existing_history
    else:
        history = []

    adaptive_weight_history = []

    global_epoch = start_epoch

    # Recompute epochs_total after we know start_epoch
    # So that only remaining epochs are trained
    epochs_total = int(epochs if epochs is not None else cfg.epochs)
    if start_epoch > 0:
        epochs_total = max(0, epochs_total - start_epoch)
    epochs_per_window = max(1, epochs_total // len(windows))
    replay_si = None

    # Historisation des points adaptatifs et de la perte locale.
    use_adaptive_history = bool(getattr(cfg, "use_adaptive_sampling", False))
    adaptive_history_epochs = []
    adaptive_history_windows = []
    adaptive_history_tleft = []
    adaptive_history_tright = []
    adaptive_points = []
    adaptive_scores = []
    adaptive_continuity = []
    adaptive_vof = []
    adaptive_momentum = []
    adaptive_alpha = []

    gauge_si = torch.tensor([[0.5*(cfg.x_min+cfg.x_max),
                              0.5*(cfg.y_min+cfg.y_max), cfg.t_min]],
                            dtype=torch.float32, device=device)
    gauge = nondimensionalize_points(gauge_si, cfg)

    for window_id, (t_left, t_right) in enumerate(windows):
        # Reset GradNorm baselines at the start of each new temporal window (except the first)
        if cfg.use_adaptive_loss_weights and window_id > 0:
            weight_balancer.reset_baselines()

        window_cfg = replace(cfg, t_min=t_left, t_max=t_right)

        # Progress bar for total epochs
        if cfg.use_temporal_segmentation:
            pbar = tqdm(total=epochs_per_window, desc=f"Training on window {window_id+1}/{len(windows)}, t=[{t_left}, {t_right}]", leave=True, dynamic_ncols=True)
        else:
            pbar = tqdm(total=epochs_per_window, desc=f"Training for t=[{t_left}, {t_right}]", leave=True, dynamic_ncols=True)


        # Les poids du réseau et l'état d'Adam sont conservés.
        cached_interior_si = None
        cached_walls_si = None
        fixed_lbfgs_points = None

        for local_epoch in range(epochs_per_window):
            global_epoch += 1
            model.train()

            sampling_refreshed = False

            if cfg.use_adaptive_sampling and (
                cached_interior_si is None
                or local_epoch % cfg.sampling_update_interval == 0
            ):
                sampling_refreshed = True

                model.eval()
                with torch.enable_grad():
                    # Interior candidates
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

                    # Wall candidates and adaptive selection
                    candidates_walls_si = sample_walls(
                        cfg.n_candidate_points,
                        window_cfg,
                        device=device,
                        include_top=not bool(getattr(cfg, "top_is_open", True)),
                    )
                    candidates_walls_nd = nondimensionalize_points(candidates_walls_si, cfg)
                    # Score walls by u^2 + v^2 (no-slip violation)
                    with torch.no_grad():
                        wall_pred = model(candidates_walls_nd)
                        wall_score = wall_pred["u"].pow(2) + wall_pred["v"].pow(2)
                    # Flatten for scoring
                    wall_scores_flat = wall_score.reshape(-1)
                    selected_walls_nd, wall_selected_idx = sample_mixed_adaptive(
                        candidates_walls_nd,
                        wall_scores_flat,
                        cfg.n_wall,
                        adaptive_fraction=cfg.adaptive_fraction,
                        return_indices=True,
                    )
                    cached_walls_si = dimensionalize_points(
                        selected_walls_nd, cfg
                    ).detach()

                    # Initial candidates (window 0 only)
                    if window_id == 0:
                        candidates_init_si = sample_initial(
                            cfg.n_candidate_points, window_cfg, device=device
                        )
                        candidates_init_nd = nondimensionalize_points(candidates_init_si, cfg)
                        with torch.no_grad():
                            init_pred = model(candidates_init_nd)
                            # Score by u^2 + v^2 + (alpha - alpha_0)^2
                            target_alpha = torch.sigmoid(
                                (cfg.initial_level / cfg.length_ref
                                - candidates_init_nd[:, 1:2])
                                / (cfg.interface_smoothing / cfg.length_ref)
                            )
                            init_score = (
                                init_pred["u"].pow(2)
                                + init_pred["v"].pow(2)
                                + (init_pred["alpha"] - target_alpha).pow(2)
                            ).reshape(-1)
                        selected_init_nd, init_selected_idx = sample_mixed_adaptive(
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

                # Historisation des points adaptatifs et de la perte locale.
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
                replay_batch = replay_si[replay_idx]
                interior_si = torch.cat(
                    [interior_si, replay_batch], dim=0
                )

            if cfg.use_adaptive_sampling and cached_walls_si is not None:
                walls_si = cached_walls_si.clone()
            else:
                walls_si = sample_walls(
                    cfg.n_wall,
                    window_cfg,
                    device=device,
                    include_top=not bool(getattr(cfg, "top_is_open", True)),
                )

            if window_id == 0 and cfg.use_adaptive_sampling and cached_initial_si is not None:
                initial_si = cached_initial_si.clone()
            else:
                initial_si = (
                    sample_initial(cfg.n_initial, cfg, device=device)
                    if window_id == 0 else None
                )

            interior = nondimensionalize_points(interior_si, cfg)
            walls = nondimensionalize_points(walls_si, cfg)


            lc = continuity_loss(model, interior, cfg)
            lv = volume_fraction_loss(model, interior, cfg)
            lmx, lmy = momentum_loss(model, interior, cfg)
            lwu, lwv = wall_loss(model, walls, cfg)

            if initial_si is not None:
                initial = nondimensionalize_points(initial_si, cfg)
                liu, liv, lia = initial_loss(model, initial, cfg)
            else:
                liu = lc.new_zeros(())
                liv = lc.new_zeros(())
                lia = lc.new_zeros(())

            lg = pressure_gauge_loss(model, gauge, cfg)

            # Split-term dictionary for GradNorm and weighted sum
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

            # Option adaptative : remplacer les poids périodiquement.
            if cfg.use_adaptive_loss_weights and (
                local_epoch % cfg.weight_update_interval == 0
                and not sampling_refreshed
                and local_epoch > 0
            ):
                adaptive_weights = weight_balancer.update(
                    loss_terms, model
                )

                weights.update(adaptive_weights)


            adaptive_weight_history.append({
                "epoch": global_epoch,
                "window": window_id,
                **{
                    name: float(weights[name])
                    for name in weight_names
                },
            })


            total = sum(weights[k] * loss_terms[k] for k in loss_terms)

            adam.zero_grad(set_to_none=True)
            total.backward()
            adam.step()

            row = {
                "epoch": global_epoch,
                "window": window_id,
                "t_left": t_left,
                "t_right": t_right,
                "total": float(total.detach().cpu()),
                **{
                    key: float(value.detach().cpu())
                    for key, value in loss_terms.items()
                },
            }
            history.append(row)

            # Update progress bar with losses
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

        # Les derniers points peuvent servir de points de replay
        # à la fenêtre suivante. Échantillonner plusieurs temps
        # est préférable à un seul instant de raccord.
        replay_si = sample_interior(
            cfg.n_collocation, window_cfg, device=device
        ).detach()

    
        # Fixed points for L-BFGS: interior + walls + initial
        fixed_interior = nondimensionalize_points(
            sample_interior(cfg.n_collocation, cfg, device=device),
            cfg,
        )
        fixed_walls = nondimensionalize_points(
            sample_walls(cfg.n_wall, cfg, device=device, include_top=False),
            cfg,
        )
        fixed_initial = nondimensionalize_points(
            sample_initial(cfg.n_initial, cfg, device=device),
            cfg,
        )

    if cfg.use_lbfgs:

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

        for _ in range(cfg.lbfgs_epochs):
            model.train()
            lbfgs.step(closure)

    # Sauvegarde de l'historique des points adaptatifs et de la perte locale.
    if use_adaptive_history and adaptive_points:
        points_arr = np.stack(adaptive_points, axis=0)          # (M, N, 3)
        scores_arr = np.stack(adaptive_scores, axis=0)          # (M, N)
        cont_arr = np.stack(adaptive_continuity, axis=0)        # (M, N)
        vof_arr = np.stack(adaptive_vof, axis=0)                # (M, N)
        mom_arr = np.stack(adaptive_momentum, axis=0)           # (M, N)
        alpha_arr = np.stack(adaptive_alpha, axis=0)            # (M, N)
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

    write_history_csv(history, run_dir)
    (run_dir/"history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Also save a snapshot as historyNN.json / checkpointNN.json with only the NEW epochs
    # trained in this session - only when resuming, and with incremental numbering.
    if resumed and start_epoch > 0 and len(history) > start_epoch:
        new_history = history[start_epoch:]
        # Assign new epoch numbers starting from start_epoch + 1
        for i, entry in enumerate(new_history):
            entry["epoch"] = start_epoch + i + 1
        new_hist_json = json.dumps(new_history, indent=2)
        # Use incremental numbering: find the next available number from existing
        # numbered backups (checkpoint01.pt, history01.json, etc.)
        import re
        pattern_ckpt = re.compile(r"^checkpoint(\d+)\.pt$")
        pattern_hist = re.compile(r"^history(\d+)\.json$")
        nums_ckpt = []
        nums_hist = []
        for p in run_dir.iterdir():
            m = pattern_ckpt.match(p.name)
            if m:
                nums_ckpt.append(int(m.group(1)))
            m = pattern_hist.match(p.name)
            if m:
                nums_hist.append(int(m.group(1)))
        next_idx = 1
        if nums_ckpt or nums_hist:
            next_idx = max(max(nums_ckpt, default=0), max(nums_hist, default=0)) + 1
        new_ckpt_file = run_dir / f"checkpoint{next_idx:02d}.pt"
        new_hist_file = run_dir / f"history{next_idx:02d}.json"
        # Copy checkpoint_final.pt to the numbered backup
        import shutil
        shutil.copy2(str(run_dir / "checkpoint_final.pt"), str(new_ckpt_file))
        # Write the new history snapshot
        with open(str(new_hist_file), "w", encoding="utf-8") as f:
            f.write(new_hist_json)
    else:
        # Not resuming: no numbered snapshots created
        new_hist_json = json.dumps(history, indent=2)

    if adaptive_weight_history:
        with open(
            run_dir / "adaptive_weights.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                adaptive_weight_history,
                f,
                indent=2,
            )
        
        final_adaptive_weights = {
            name: float(weights[name])
            for name in weight_names
        }

        with open(
            run_dir / "adaptive_weights_final.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                final_adaptive_weights,
                f,
                indent=2,
            )



    print(f"Training finished. Results saved to: {run_dir}")

    # Save model weights as checkpoint (weights-only, compatible with SloshingPINN.load)
    torch.save(model.state_dict(), run_dir / "checkpoint_final.pt")

    return model, history, run_dir

if __name__ == "__main__":
    train()
