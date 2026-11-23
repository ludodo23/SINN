#!/usr/bin/env python3
"""
Analyse de convergence d'un PINN à partir de :
  - un JSON de loss par epoch
  - un checkpoint .pt (optionnel)

Formats JSON acceptés :
  {"loss_pde": [...], "loss_ic": [...], "total": [...]}          (dict de listes)
  [{"epoch": 0, "loss_pde": ..., "loss_ic": ...}, ...]           (liste de dicts)
  {"history": <un des deux formats ci-dessus>}                   (clé englobante)

Usage :
  python analyse_pinn.py --json losses.json --model model.pt --out analyse/
  python analyse_pinn.py --json losses.json --total-key total --window 200
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Chargement et normalisation de l'historique
# --------------------------------------------------------------------------- #
def load_history(path):
    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, dict) and len(raw) == 1 and isinstance(next(iter(raw.values())), (dict, list)):
        inner = next(iter(raw.values()))
        if isinstance(inner, list) and inner and isinstance(inner[0], dict):
            raw = inner
        elif isinstance(inner, dict):
            raw = inner

    if isinstance(raw, list):  # liste de dicts -> dict de listes
        keys = [k for k in raw[0] if k != "epoch"]
        hist = {k: [r.get(k, np.nan) for r in raw] for k in keys}
    else:
        hist = dict(raw)

    out = {}
    for k, v in hist.items():
        try:
            arr = np.asarray(v, dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.ndim == 1 and arr.size > 1:
            out[k] = arr
    if not out:
        raise ValueError("Aucune série numérique exploitable dans le JSON.")
    return out


def moving_avg(x, w):
    w = max(1, min(w, len(x)))
    return np.convolve(x, np.ones(w) / w, mode="valid")


def is_loss_key(k):
    kl = k.lower()
    return not any(s in kl for s in ("lr", "learning", "time", "epoch", "grad"," window", "t_left", "t_right"))


# --------------------------------------------------------------------------- #
# Diagnostics sur les courbes de loss
# --------------------------------------------------------------------------- #
def diagnostics(hist, total_key, window):
    lines = []
    n = len(next(iter(hist.values())))
    lines.append(f"Epochs : {n}")

    for k, v in hist.items():
        if not is_loss_key(k):
            continue
        nan = int(np.isnan(v).sum() + np.isinf(v).sum())
        tail = v[-max(1, n // 10):]
        # pente log sur les 10 % finaux : < 0 => encore en décroissance
        t = np.arange(len(tail))
        pos = tail > 0
        slope = np.polyfit(t[pos], np.log10(tail[pos]), 1)[0] if pos.sum() > 2 else np.nan
        # sauts : rapport max entre epochs successives (instabilités)
        ratio = v[1:] / np.maximum(v[:-1], 1e-30)
        spikes = int((ratio > 10).sum())
        lines.append(
            f"{k:>18s} | init {v[0]:.3e} | final {v[-1]:.3e} | min {np.nanmin(v):.3e} "
            f"(epoch {int(np.nanargmin(v))}) | pente log10/epoch (10% fin) {slope:+.2e} "
            f"| pics x10 : {spikes} | NaN/Inf : {nan}"
        )

    # Détection de plateau sur la loss totale
    if total_key in hist:
        tot = hist[total_key]
        if len(tot) > 2 * window:
            ma = moving_avg(tot, window)
            rel = abs(ma[-1] - ma[-window]) / max(abs(ma[-window]), 1e-30) if len(ma) > window else np.nan
            lines.append(
                f"\nVariation relative de la loss lissée sur les {window} derniers epochs : {rel:.2%}"
                + ("  -> plateau probable" if rel < 0.01 else "")
            )

    # Terme dominant en fin d'entraînement
    terms = {k: v[-1] for k, v in hist.items() if is_loss_key(k) and k != total_key}
    if terms:
        s = sum(terms.values())
        if s > 0:
            lines.append("\nContribution finale par terme :")
            for k, val in sorted(terms.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {k:>18s} : {val / s:6.1%}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_history(hist, total_key, window, out):
    loss_keys = [k for k in hist if is_loss_key(k)]
    other_keys = [k for k in hist if not is_loss_key(k)]
    n = len(hist[loss_keys[0]])
    ep = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) toutes les loss, échelle log
    ax = axes[0, 0]
    for k in loss_keys:
        ax.semilogy(ep, np.abs(hist[k]) + 1e-30, alpha=0.35, lw=0.8)
        if n > 2 * window:
            ax.semilogy(ep[window - 1:], np.abs(moving_avg(hist[k], window)) + 1e-30, lw=1.6, label=k)
        else:
            ax.lines[-1].set_label(k)
    ax.set(title="Loss par terme (brut + lissé)", xlabel="epoch", ylabel="loss")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # (b) contributions relatives
    ax = axes[0, 1]
    terms = [k for k in loss_keys if k != total_key]
    if terms:
        mat = np.vstack([np.abs(hist[k]) for k in terms])
        mat = mat / np.maximum(mat.sum(axis=0), 1e-30)
        ax.stackplot(ep, mat, labels=terms, alpha=0.8)
        ax.set(title="Contribution relative de chaque terme", xlabel="epoch", ylim=(0, 1))
        ax.legend(fontsize=8, loc="upper right")

    # (c) ratios vs terme résidu (si identifiable)
    ax = axes[1, 0]
    pde = next((k for k in terms if any(s in k.lower() for s in ("pde", "res", "phys"))), None)
    if pde:
        for k in terms:
            if k != pde:
                ax.semilogy(ep, np.abs(hist[k]) / np.maximum(np.abs(hist[pde]), 1e-30), label=f"{k} / {pde}")
        ax.set(title="Équilibre entre termes (ratio au résidu)", xlabel="epoch")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    else:
        ax.axis("off")

    # (d) grandeurs auxiliaires (lr, poids adaptatifs...)
    ax = axes[1, 1]
    if other_keys:
        for k in other_keys:
            ax.plot(ep, hist[k], label=k)
        ax.set(title="Autres grandeurs loggées", xlabel="epoch")
        ax.legend(fontsize=8)
    else:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out / "convergence_loss.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Analyse du checkpoint
# --------------------------------------------------------------------------- #
def analyse_checkpoint(path, out):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
    if hasattr(ckpt, "state_dict"):  # modèle complet sérialisé
        ckpt = ckpt.state_dict()

    names, norms, max_abs, bad = [], [], [], 0
    n_params = 0
    lines = []
    for name, t in ckpt.items():
        if not hasattr(t, "numel") or not t.is_floating_point():
            continue
        n_params += t.numel()
        bad += int((~torch.isfinite(t)).sum())
        if t.ndim >= 2:  # poids uniquement
            names.append(name)
            norms.append(float(t.norm()))
            max_abs.append(float(t.abs().max()))
        lines.append(f"{name:>40s} {tuple(t.shape)!s:>14s} | norme {float(t.norm()):.3e} | max|w| {float(t.abs().max()):.3e}")

    header = f"Paramètres : {n_params:,} | valeurs non finies : {bad}"
    (out / "checkpoint_resume.txt").write_text(header + "\n" + "\n".join(lines))

    if names:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].bar(range(len(names)), norms)
        ax[0].set(title="Norme des poids par couche", xlabel="couche", ylabel="||W||")
        ax[1].bar(range(len(names)), max_abs, color="tab:orange")
        ax[1].set(title="max |w| par couche", xlabel="couche")
        for a in ax:
            a.set_xticks(range(len(names)))
            a.set_xticklabels(range(len(names)))
        fig.tight_layout()
        fig.savefig(out / "checkpoint_poids.png", dpi=150)
        plt.close(fig)
    return header


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, help="JSON des loss par epoch")
    p.add_argument("--model", help="checkpoint .pt (optionnel)")
    p.add_argument("--out", default="analysis/claude", help="dossier de sortie")
    p.add_argument("--total-key", default="total", help="clé de la loss totale")
    p.add_argument("--window", type=int, default=100, help="fenêtre de lissage (epochs)")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    hist = load_history(a.json)
    report = diagnostics(hist, a.total_key, a.window)
    plot_history(hist, a.total_key, a.window, out)

    if a.model:
        report += "\n\n" + analyse_checkpoint(a.model, out)

    (out / "rapport.txt").write_text(report)
    print(report)
    print(f"\nFichiers écrits dans {out.resolve()}")


if __name__ == "__main__":
    main()
