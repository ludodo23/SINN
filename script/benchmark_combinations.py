import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import Config
from train import train
from model import SloshingPINN


def make_config(use_fourier_features=False,
                use_adaptive_sampling=False,
                use_temporal_segmentation=False,
                use_adaptive_loss_weights=False,
                **kwargs):
    cfg = Config()
    cfg.use_fourier_features = bool(use_fourier_features)
    cfg.use_adaptive_sampling = bool(use_adaptive_sampling)
    cfg.use_temporal_segmentation = bool(use_temporal_segmentation)
    cfg.use_adaptive_loss_weights = bool(use_adaptive_loss_weights)
    cfg.use_lbfgs = False
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def run_inference_and_error(model, cfg, data_npz_path, device):
    data = np.load(data_npz_path, allow_pickle=True)
    X = data["X"]
    times = data["time"]
    n_times, n_points, _ = X.shape

    model.eval()
    model.to(device)

    rmse_u = np.empty(n_times, dtype=np.float64)
    rmse_v = np.empty(n_times, dtype=np.float64)
    rmse_p = np.empty(n_times, dtype=np.float64)
    rmse_alpha = np.empty(n_times, dtype=np.float64)

    batch_size = 2048

    with torch.inference_mode():
        for i in range(n_times):
            xy = X[i, :, :2]
            t_val = float(times[i])
            t_col = np.full((n_points, 1), t_val, dtype=np.float32)
            coords_np = np.concatenate([xy, t_col], axis=1)
            coords = torch.as_tensor(coords_np, dtype=torch.float32, device=device)

            pred = model.predict_physical(coords)

            ref_u = data["u"][i]
            ref_v = data["v"][i]
            ref_p = data["p"][i]
            ref_alpha = data["alpha"][i]

            err_u = (pred["u"].squeeze(-1).float().cpu().numpy() - ref_u) ** 2
            err_v = (pred["v"].squeeze(-1).float().cpu().numpy() - ref_v) ** 2
            err_p = (pred["p"].squeeze(-1).float().cpu().numpy() - ref_p) ** 2
            err_alpha = (pred["alpha"].squeeze(-1).float().cpu().numpy() - ref_alpha) ** 2

            rmse_u[i] = float(np.sqrt(err_u.mean()))
            rmse_v[i] = float(np.sqrt(err_v.mean()))
            rmse_p[i] = float(np.sqrt(err_p.mean()))
            rmse_alpha[i] = float(np.sqrt(err_alpha.mean()))

    return {
        "timestamps": times.copy(),
        "rmse_u": rmse_u,
        "rmse_v": rmse_v,
        "rmse_p": rmse_p,
        "rmse_alpha": rmse_alpha,
    }
def main():
    results_dir = ROOT / "reports"
    models_dir = ROOT / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    data_path = ROOT / "data" / "sloshing_data.npz"
    if not data_path.is_file():
        print(f"Reference data not found at {data_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_kwargs = dict(
        epochs=50,
        seed=1234,
    )

    flags = [
        "use_fourier_features",
        "use_adaptive_sampling",
        "use_temporal_segmentation",
        "use_adaptive_loss_weights",
    ]

    all_combos = list(itertools.product([False, True], repeat=4))

    summary = []
    model_table = []

    for idx, combo in enumerate(all_combos, start=1):
        ff, asamp, ts, alw = combo
        model_name = f"model_ff{int(ff)}_as{int(asamp)}_ts{int(ts)}_alw{int(alw)}"

        print(f"\n=== Combination {idx}/{len(all_combos)}: "
              f"use_fourier_features={ff}, use_adaptive_sampling={asamp}, "
              f"use_temporal_segmentation={ts}, use_adaptive_loss_weights={alw} "
              f"(model_name={model_name}) ===")

        cfg = make_config(
            use_fourier_features=ff,
            use_adaptive_sampling=asamp,
            use_temporal_segmentation=ts,
            use_adaptive_loss_weights=alw,
            **base_kwargs,
        )
        cfg.model_name = model_name

        t0 = time.perf_counter()
        model, history, run_dir = train(
            cfg,
            output_dir=str(models_dir),
            model_name=model_name,
        )
        train_time = time.perf_counter() - t0
        print(f"  Training time: {train_time:.2f}s, final continuity loss: {history[-1].get('continuity', 'N/A'):.6e}")

        inf = run_inference_and_error(model, cfg, data_path, device)

        out_npz = results_dir / f"errors_{model_name}.npz"
        np.savez_compressed(
            out_npz,
            timestamps=inf["timestamps"],
            rmse_u=inf["rmse_u"],
            rmse_v=inf["rmse_v"],
            rmse_p=inf["rmse_p"],
            rmse_alpha=inf["rmse_alpha"],
        )
        print(f"  Errors saved to: {out_npz}")

        mean_rmse = {
            "u": float(np.mean(inf["rmse_u"])),
            "v": float(np.mean(inf["rmse_v"])),
            "p": float(np.mean(inf["rmse_p"])),
            "alpha": float(np.mean(inf["rmse_alpha"])),
        }
        summary.append({
            "model_name": model_name,
            "combination": f"ff{ff}_asamp{asamp}_ts{ts}_alw{alw}",
            "use_fourier_features": ff,
            "use_adaptive_sampling": asamp,
            "use_temporal_segmentation": ts,
            "use_adaptive_loss_weights": alw,
            "train_time_s": train_time,
            "mean_rmse_u": mean_rmse["u"],
            "mean_rmse_v": mean_rmse["v"],
            "mean_rmse_p": mean_rmse["p"],
            "mean_rmse_alpha": mean_rmse["alpha"],
        })

        model_table.append({
            "model_name": model_name,
            "use_fourier_features": bool(ff),
            "use_adaptive_sampling": bool(asamp),
            "use_temporal_segmentation": bool(ts),
            "use_adaptive_loss_weights": bool(alw),
        })

        del model, history, inf

    # Model name <-> features table
    table_json = results_dir / "model_table.json"
    with open(table_json, "w", encoding="utf-8") as f:
        json.dump(model_table, f, indent=2, ensure_ascii=False)
    print(f"\n Model table written to: {table_json}")

    table_csv = results_dir / "model_table.csv"
    with open(table_csv, "w", encoding="utf-8") as f:
        f.write("model_name,use_fourier_features,use_adaptive_sampling,use_temporal_segmentation,use_adaptive_loss_weights\n")
        for m in model_table:
            f.write(f"{m['model_name']},{int(m['use_fourier_features'])},"
                    f"{int(m['use_adaptive_sampling'])},{int(m['use_temporal_segmentation'])},"
                    f"{int(m['use_adaptive_loss_weights'])}\n")
    print(f" Model table CSV written to: {table_csv}")

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n Summary written to: {summary_path}")

    csv_path = results_dir / "timings.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("model_name,train_time_s,mean_rmse_u,mean_rmse_v,mean_rmse_p,mean_rmse_alpha\n")
        for s in summary:
            f.write(f"{s['model_name']},{s['train_time_s']:.4f},"
                    f"{s['mean_rmse_u']:.6e},{s['mean_rmse_v']:.6e},"
                    f"{s['mean_rmse_p']:.6e},{s['mean_rmse_alpha']:.6e}\n")
    print(f" Timings CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
