import json
import datetime
from pathlib import Path

import torch

from config import Config
from model import SloshingPINN
from sampling import (
    sample_interior,
    sample_initial,
    sample_walls,
)
from losses import (
    continuity_loss,
    vof_loss,
    momentum_loss,
    turbulence_losses,
    initial_loss,
    wall_loss,
    pressure_gauge_loss,
)


def train(cfg=None, output_directory="outputs", name="sinn"):
    cfg = (cfg or Config()).validate()

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    model = SloshingPINN(
        width=cfg.hidden_width,
        hidden_layers=cfg.hidden_layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
    )

    history = []

    gauge_point = torch.tensor(
        [[
            (cfg.x_min + cfg.x_max) / 2,
            (cfg.y_min + cfg.y_max) / 2,
            cfg.t_min,
        ]],
        dtype=torch.float32,
        device=device,
    )

    for epoch in range(1, cfg.epochs + 1):
        interior = sample_interior(
            cfg.n_collocation, cfg, device
        )
        initial = sample_initial(
            cfg.n_initial, cfg, device
        )
        walls = sample_walls(
            cfg.n_wall, cfg, device
        )

        optimizer.zero_grad(set_to_none=True)

        loss_cont = continuity_loss(model, interior, cfg)
        loss_vof = vof_loss(model, interior, cfg)
        loss_mom = momentum_loss(model, interior, cfg)

        loss_k, loss_eps = turbulence_losses(
            model, interior, cfg
        )

        loss_init = initial_loss(model, initial, cfg)
        loss_wall = wall_loss(model, walls)
        loss_pressure = pressure_gauge_loss(
            model, gauge_point
        )

        total = (
            cfg.w_continuity * loss_cont
            + cfg.w_vof * loss_vof
            + cfg.w_momentum * loss_mom
            + cfg.w_k * loss_k
            + cfg.w_epsilon * loss_eps
            + cfg.w_initial * loss_init
            + cfg.w_wall * loss_wall
            + cfg.w_pressure_gauge * loss_pressure
        )

        total.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=100.0
        )
        optimizer.step()

        row = {
            "epoch": epoch,
            "total": float(total.detach().cpu()),
            "continuity": float(loss_cont.detach().cpu()),
            "vof": float(loss_vof.detach().cpu()),
            "momentum": float(loss_mom.detach().cpu()),
            "k": float(loss_k.detach().cpu()),
            "epsilon": float(loss_eps.detach().cpu()),
            "initial": float(loss_init.detach().cpu()),
            "wall": float(loss_wall.detach().cpu()),
            "pressure_gauge": float(loss_pressure.detach().cpu()),
        }
        history.append(row)

        if epoch == 1 or epoch % cfg.print_every == 0:
            print(
                f"Epoch {epoch:5d} | "
                f"total={row['total']:.4e} | "
                f"continuity={row['continuity']:.3e} | "
                f"VOF={row['vof']:.3e} | "
                f"momentum={row['momentum']:.3e}"
            )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
    }

    output_dir = Path(output_directory)
    output_dir.mkdir(exist_ok=True)

    torch.save(
        checkpoint,
        output_dir / f"{name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt",
    )

    with (output_dir / f"loss_history_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(history, file, indent=2)

    cfg.save_json(output_dir / f"config_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")

    return model, history


if __name__ == "__main__":
    train()