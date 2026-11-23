"""
Analyse de convergence d'un PINN à partir de :
  - losses.json : historique des pertes par epoch
  - model.pt    : modèle sauvegardé (state_dict)
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Chargement et inspection des pertes
# ---------------------------------------------------------------
def load_losses(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)

    # Normalise en dict de listes numpy
    # Accepte soit {"loss": [...], "loss_pde": [...], ...}
    # soit une liste de dicts [{"epoch":0,"loss":...}, ...]
    if isinstance(data, list):
        keys = data[0].keys()
        data = {k: [d[k] for d in data] for k in keys}

    return {k: np.asarray(v, dtype=float) for k, v in data.items()}


def plot_losses(losses: dict, outdir: Path, logy=True):
    """Trace chaque composante de perte, en échelle log optionnelle."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- vue globale ---
    ax = axes[0]
    for k, v in losses.items():
        if k.lower() in ("epoch", "window", "t_left", "t_right"):
            continue
        ax.plot(v, label=k, alpha=0.85)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Pertes (échelle linéaire)")
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- zoom sur les dernières epochs ---
    ax = axes[1]
    n = len(next(iter(losses.values())))
    start = int(0.7 * n)
    for k, v in losses.items():
        if k.lower() in ("epoch", "window", "t_left", "t_right"):
            continue
        ax.plot(range(start, n), v[start:], label=k, alpha=0.85)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"Zoom dernières 30% (epoch {start}→{n})")
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "losses.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------
# 2. Indicateurs quantitatifs de convergence
# ---------------------------------------------------------------
def convergence_metrics(losses: dict, window: int = 50) -> dict:
    """
    Calcule des indicateurs de convergence :
      - valeur finale, min, pente récente (log-log)
      - stabilité (écart-type glissant)
      - plateau detection
    """
    metrics = {}
    for k, v in losses.items():
        if k.lower() in ("epoch", "window", "t_left", "t_right"):
            continue
        v_safe = np.maximum(v, 1e-12)
        recent = v_safe[-window:]

        # pente en log-log sur la fenêtre récente
        x = np.arange(len(recent))
        slope = np.polyfit(x, np.log(recent), 1)[0] if len(recent) > 2 else np.nan

        # stabilité relative
        rel_std = np.std(recent) / np.mean(recent)

        metrics[k] = {
            "final": float(v_safe[-1]),
            "min": float(v_safe.min()),
            "min_epoch": int(np.argmin(v_safe)),
            "slope_log_recent": float(slope),   # <0 => décroît encore
            "rel_std_recent": float(rel_std),    # proche de 0 => plateau
        }
    return metrics


def detect_plateau(v: np.ndarray, window: int = 50,
                   tol: float = 1e-3) -> int | None:
    """Retourne l'epoch à partir de laquelle la perte stagne (ou None)."""
    v = np.maximum(v, 1e-12)
    for i in range(len(v) - window):
        seg = v[i:i + window]
        if np.std(seg) / np.mean(seg) < tol:
            return i
    return None


# ---------------------------------------------------------------
# 3. Diagnostic NTK (optionnel, nécessite le modèle + les points)
# ---------------------------------------------------------------
def ntk_diagnostics(model, collocation_pts, residual_fn, max_pts=500):
    """
    Calcule un NTK approché sur un sous-échantillon de points de collocation.
    residual_fn(x) doit retourner le résidu de l'EDP pour x (tensor).
    """
    import torch

    model.eval()
    pts = torch.as_tensor(collocation_pts[:max_pts],
                          dtype=torch.float32, requires_grad=True)

    # gradient du résidu par rapport aux paramètres
    params = [p for p in model.parameters() if p.requires_grad]
    res = residual_fn(pts)                      # (N, d_out)
    grads = []
    for i in range(res.shape[1]):
        g = torch.autograd.grad(res[:, i].sum(), params,
                                retain_graph=True, create_graph=False)
        grads.append(torch.cat([gi.flatten() for gi in g]))
    J = torch.stack(grads, dim=1)               # (P, N)

    K = J.T @ J                                  # (N, N)
    eigvals = torch.linalg.eigvalsh(K).clamp_min(0).numpy()[::-1]

    r_eff = eigvals.sum() / (eigvals[0] + 1e-12)
    return {
        "eigenvalues": eigvals,
        "effective_rank": float(r_eff),
        "cond_number": float(eigvals[0] / (eigvals[-1] + 1e-12)),
    }


def plot_spectrum(eigvals: np.ndarray, outdir: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(eigvals, ".-", markersize=3)
    ax.set_xlabel("indice")
    ax.set_ylabel("valeur propre")
    ax.set_title("Spectre du NTK (résidu)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "ntk_spectrum.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------
# 4. Rapport texte
# ---------------------------------------------------------------
def write_report(metrics: dict, plateaus: dict, outdir: Path,
                 ntk: dict | None = None):
    lines = ["=== Rapport d'analyse de convergence ===\n"]
    for k, m in metrics.items():
        lines.append(
            f"[{k}]\n"
            f"  final          = {m['final']:.3e}\n"
            f"  min            = {m['min']:.3e}  (epoch {m['min_epoch']})\n"
            f"  pente log réc. = {m['slope_log_recent']:+.3e}\n"
            f"  rel. std réc.  = {m['rel_std_recent']:.3e}\n"
            f"  plateau depuis = {plateaus.get(k)}\n"
        )
    if ntk:
        lines.append(
            f"\n[NTK]\n"
            f"  rang effectif  = {ntk['effective_rank']:.2f}\n"
            f"  cond. number   = {ntk['cond_number']:.3e}\n"
        )
    (outdir / "report.txt").write_text("\n".join(lines))
    print("\n".join(lines))


# ---------------------------------------------------------------
# 5. Point d'entrée
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--losses", default="losses.json")
    parser.add_argument("--model",  default="model.pt")
    parser.add_argument("--outdir", default="analysis/deepseek")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--ntk", action="store_true",
                        help="active le calcul NTK (nécessite hooks)")
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True, parents=True)

    losses = load_losses(args.losses)
    plot_losses(losses, outdir)
    metrics = convergence_metrics(losses, window=args.window)

    plateaus = {}
    for k, v in losses.items():
        if k.lower() in ("epoch", "window", "t_left", "t_right"):
            continue
        plateaus[k] = detect_plateau(v, window=args.window)

    ntk = None
    if args.ntk:
        # À adapter : charger le modèle, récupérer les points et
        # définir residual_fn selon ton EDP.
        import torch
        model = torch.load(args.model, map_location="cpu")
        # ntk = ntk_diagnostics(model, collocation_pts, residual_fn)
        # plot_spectrum(ntk["eigenvalues"], outdir)
        print("[NTK] brancher les points de collocation + residual_fn")

    write_report(metrics, plateaus, outdir, ntk)


if __name__ == "__main__":
    main()