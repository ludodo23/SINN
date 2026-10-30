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
    if not history:
        return

    path = Path(run_dir) / "benchmark.csv"
    fieldnames = list(history[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def train(
    cfg: Config | None = None,
    output_dir: str | Path = "runs",
    model: SloshingPINN | None = None,
    state_path: str | Path | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    epochs: int | None = None,
    lbfgs_epochs: int | None = None,
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
    cfg = cfg or Config()
    if hasattr(cfg, "validate"): cfg.validate()
    _validate_scales(cfg)
    set_seed(int(cfg.seed))
    device = torch.device(cfg.device)
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    run_dir = root / ("sloshing_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    if model is None:
        model = SloshingPINN(config=cfg, state_path=state_path).to(device)
        (run_dir/"config.json").write_text(json.dumps(config_dict(cfg), indent=2), encoding="utf-8")
    else:
        model = model.to(device)

    
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
        "continuity", "vof", "momentum",
        "initial", "wall", "pressure_gauge",
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
    )

    history = []
    global_epoch = 0
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
        window_cfg = replace(cfg, t_min=t_left, t_max=t_right)

        # Progress bar for total epochs
        if cfg.use_temporal_segmentation:
            pbar = tqdm(total=epochs_per_window, desc=f"Training on window {window_id+1}/{len(windows)}, t=[{t_left}, {t_right}]", leave=True, dynamic_ncols=True)
        else:
            pbar = tqdm(total=epochs_per_window, desc=f"Training for t=[{t_left}, {t_right}]", leave=True, dynamic_ncols=True)


        # Les poids du réseau et l'état d'Adam sont conservés.
        cached_interior_si = None
        fixed_lbfgs_points = None

        for local_epoch in range(epochs_per_window):
            global_epoch += 1
            model.train()

            # Collocation adaptative : mise à jour périodique du pool.
            if cfg.use_adaptive_sampling and (
                cached_interior_si is None
                or local_epoch % cfg.sampling_update_interval == 0
            ):
                model.eval()
                with torch.enable_grad():
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

            initial_si = (
                sample_initial(cfg.n_initial, cfg, device=device)
                if window_id == 0 else None
            )
            walls_si = sample_walls(
                cfg.n_wall,
                window_cfg,
                device=device,
                include_top=not bool(getattr(cfg, "top_is_open", True)),
            )

            interior = nondimensionalize_points(interior_si, cfg)
            walls = nondimensionalize_points(walls_si, cfg)

            lc = continuity_loss(model, interior, cfg)
            lv = volume_fraction_loss(model, interior, cfg)
            lm = momentum_loss(model, interior, cfg)
            lw = wall_loss(model, walls, cfg)

            if initial_si is not None:
                initial = nondimensionalize_points(initial_si, cfg)
                li = initial_loss(model, initial, cfg)
            else:
                li = lc.new_zeros(())

            lg = pressure_gauge_loss(model, gauge, cfg)

            losses = {
                "continuity": lc,
                "vof": lv,
                "momentum": lm,
                "initial": li,
                "wall": lw,
                "pressure_gauge": lg,
            }

            # Poids fixes par défaut.
            weights = {
                "continuity": cfg.w_continuity,
                "vof": cfg.w_vof,
                "momentum": cfg.w_momentum,
                "initial": cfg.w_initial,
                "wall": cfg.w_wall,
                "pressure_gauge": cfg.w_pressure_gauge,
            }

            # Option adaptative : remplacer les poids périodiquement.
            if cfg.use_adaptive_loss_weights and (
                local_epoch % cfg.weight_update_interval == 0
            ):
                weights.update(weight_balancer.update(losses, model))

            total = sum(weights[k] * losses[k] for k in losses)

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
                    for key, value in losses.items()
                },
            }
            history.append(row)

            # Update progress bar with losses
            pbar.update(1)
            pbar.set_postfix(
                total=f"{float(total.detach().cpu()):.4e}",
                continuity=f"{float(losses['continuity'].detach().cpu()):.4e}",
                volume_fraction=f"{float(losses['vof'].detach().cpu()):.4e}",
                momentum=f"{float(losses['momentum'].detach().cpu()):.4e}",
                lr=f"{adam.param_groups[0]['lr']:.2e}",
                window=f"w{window_id}",
            )

        # Les derniers points peuvent servir de points de replay
        # à la fenêtre suivante. Échantillonner plusieurs temps
        # est préférable à un seul instant de raccord.
        replay_si = sample_interior(
            cfg.n_collocation, window_cfg, device=device
        ).detach()

    
    if cfg.use_lbfgs:
        print("Starting L-BFGS optimization...", flush=True)
        # Use lbfgs_epochs parameter if provided, otherwise fall back to cfg
        lbfgs_epochs_here = lbfgs_epochs if lbfgs_epochs is not None else cfg.lbfgs_epochs
        # Ces points et les poids des pertes doivent rester fixes
        # pendant toutes les évaluations de la closure.
        fixed_points = nondimensionalize_points(
            sample_interior(cfg.n_collocation, cfg, device=device),
            cfg,
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

            lc = continuity_loss(model, fixed_points, cfg)
            lv = volume_fraction_loss(model, fixed_points, cfg)
            lm = momentum_loss(model, fixed_points, cfg)

            total = (
                frozen_weights["continuity"] * lc
                + frozen_weights["vof"] * lv
                + frozen_weights["momentum"] * lm
            )
            total.backward()
            return total

        for _ in range(lbfgs_epochs_here):
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

    print(f"Training finished. Results saved to: {run_dir}")

    return model, history, run_dir

if __name__ == "__main__":
    train()
