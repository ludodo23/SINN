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
from sampling import sample_initial, sample_interior, sample_walls
from physics import nondimensionalize_points
from losses import (continuity_loss, vof_loss, momentum_loss, initial_loss,
                    wall_loss, pressure_gauge_loss)

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def config_dict(cfg) -> dict:
    if is_dataclass(cfg): return asdict(cfg)
    if hasattr(cfg, "__dict__"): return dict(cfg.__dict__)
    raise TypeError("Config must be a dataclass or expose __dict__.")

def _validate_scales(cfg) -> None:
    names = ("length_ref", "velocity_ref", "rho_ref", "mu_ref", "gravity_ref")
    missing = [n for n in names if not hasattr(cfg, n)]
    if missing:
        raise AttributeError("Add these positive reference scales to config.py: " +
                             ", ".join(missing))
    for n in names:
        if float(getattr(cfg, n)) <= 0: raise ValueError(f"{n} must be positive.")
    if not hasattr(cfg, "time_ref"):
        cfg.time_ref = float(cfg.length_ref)/float(cfg.velocity_ref)

def train(cfg: Config | None = None, output_dir: str | Path = "runs"):
    cfg = cfg or Config()
    if hasattr(cfg, "validate"): cfg.validate()
    _validate_scales(cfg)
    set_seed(int(cfg.seed))
    device = torch.device(cfg.device)
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    run_dir = root / ("sloshing_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    model = SloshingPINN(cfg=cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate))

    gauge_si = torch.tensor([[0.5*(cfg.x_min+cfg.x_max),
                              0.5*(cfg.y_min+cfg.y_max), cfg.t_min]],
                            dtype=torch.float32, device=device)
    gauge = nondimensionalize_points(gauge_si, cfg)
    history = []
    # The tank top is open: do not impose no-slip there.
    include_top = not bool(getattr(cfg, "top_is_open", True))

    for epoch in range(1, int(cfg.epochs)+1):
        model.train()
        interior_si = sample_interior(int(cfg.n_collocation), cfg, device=device)
        initial_si = sample_initial(int(cfg.n_initial), cfg, device=device)
        walls_si = sample_walls(int(cfg.n_wall), cfg, device=device,
                                include_top=include_top)
        interior = nondimensionalize_points(interior_si, cfg)
        initial = nondimensionalize_points(initial_si, cfg)
        walls = nondimensionalize_points(walls_si, cfg)

        lc = continuity_loss(model, interior, cfg)
        lv = vof_loss(model, interior, cfg)
        lm = momentum_loss(model, interior, cfg)

        li = initial_loss(model, initial, cfg)
        lw = wall_loss(model, walls, cfg)
        lg = pressure_gauge_loss(model, gauge, cfg)
        total = (cfg.w_continuity*lc + cfg.w_vof*lv + cfg.w_momentum*lm
                 + cfg.w_initial*li + cfg.w_wall*lw
                 + cfg.w_pressure_gauge*lg)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        row = {"epoch": epoch, "total": float(total.detach().cpu()),
               "continuity": float(lc.detach().cpu()), "vof": float(lv.detach().cpu()),
               "momentum": float(lm.detach().cpu()), "initial": float(li.detach().cpu()),
               "wall": float(lw.detach().cpu()), "pressure_gauge": float(lg.detach().cpu())}
        history.append(row)
        if epoch == 1 or epoch % int(cfg.print_every) == 0:
            print(f"[{epoch:6d}/{cfg.epochs}] total={row['total']:.4e} "
                  f"cont={row['continuity']:.3e} vof={row['vof']:.3e} "
                  f"mom={row['momentum']:.3e} init={row['initial']:.3e} "
                  f"wall={row['wall']:.3e}")
        if epoch % max(1, int(cfg.print_every)) == 0 or epoch == int(cfg.epochs):
            payload = {"model_state_dict": model.state_dict(), "config": config_dict(cfg),
                       "epoch": epoch, "loss": row["total"]}
            torch.save(payload, run_dir/"checkpoint_latest.pt")
            (run_dir/"history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            (run_dir/"config.json").write_text(json.dumps(config_dict(cfg), indent=2), encoding="utf-8")

    # torch.save({"model_state_dict": model.state_dict(), "config": config_dict(cfg),
    #             "epoch": int(cfg.epochs), "loss": history[-1]["total"] if history else None},
    #            run_dir/"checkpoint_final.pt")
    torch.save(model, run_dir/"checkpoint_final.pt")
    print(f"Training finished. Results saved to: {run_dir}")
    return model, history, run_dir

if __name__ == "__main__":
    train()
