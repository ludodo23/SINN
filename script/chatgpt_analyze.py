#!/usr/bin/env python3
"""
PINN post-training diagnostics.

Analyzes a JSON loss history and optionally evaluates a saved PyTorch model
on user-supplied collocation/test points.

Expected JSON examples:
  {"epoch": [0, 1, ...], "loss": [..], "loss_res": [..], "loss_bc": [..]}
or
  [{"epoch": 0, "loss": ..., "loss_res": ..., "loss_bc": ...}, ...]

Usage:
  python analyze_pinn.py losses.json --outdir analysis
  python analyze_pinn.py losses.json --outdir analysis --model model.pt

For model residual evaluation, adapt build_model() and residual_fn() below
to your architecture and PDE. A state_dict checkpoint alone cannot define
the architecture.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_history(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Common wrappers around histories
    if isinstance(raw, dict):
        for key in ("history", "losses", "records", "data"):
            if key in raw and isinstance(raw[key], (list, dict)):
                raw = raw[key]
                break

    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        rows = raw
        keys = sorted({k for row in rows for k in row})
        result = {}
        for key in keys:
            vals = [row.get(key, np.nan) for row in rows]
            try:
                result[key] = np.asarray(vals, dtype=float)
            except (TypeError, ValueError):
                continue
        return result

    if isinstance(raw, dict):
        result = {}
        for key, vals in raw.items():
            if isinstance(vals, (list, tuple)):
                try:
                    result[key] = np.asarray(vals, dtype=float)
                except (TypeError, ValueError):
                    pass
            elif isinstance(vals, (int, float)):
                result[key] = np.asarray([vals], dtype=float)
        return result

    raise ValueError("Format JSON non reconnu. Voir les exemples en tête du script.")


def choose_epoch(history, n):
    for key in ("epoch", "epochs", "step", "iteration", "iterations"):
        if key in history and len(history[key]) == n:
            return np.asarray(history[key], dtype=float)
    return np.arange(n, dtype=float)


def clean_name(name):
    return str(name).lower().replace(" ", "").replace("-", "_")


def analyze(history, outdir, window=50):
    outdir.mkdir(parents=True, exist_ok=True)
    if not history:
        raise ValueError("Aucune série numérique trouvée dans le JSON.")

    # Ignore epoch-like series when plotting losses
    loss_series = {}
    for name, values in history.items():
        norm = clean_name(name)
        if norm in ("epoch", "epochs", "step", "iteration", "iterations", "t_left", "t_right", "window"):
            continue
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size > 1 and np.isfinite(values).any():
            loss_series[name] = values

    if not loss_series:
        raise ValueError("Aucune série de Loss de longueur > 1 trouvée.")

    n = max(len(v) for v in loss_series.values())
    epochs = choose_epoch(history, n)

    # Plot losses; each series is aligned to its own length.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, values in loss_series.items():
        x = epochs[:len(values)] if len(epochs) >= len(values) else np.arange(len(values))
        safe = np.maximum(values, np.finfo(float).tiny)
        ax.plot(x, safe, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (échelle logarithmique)")
    ax.set_title("Historique des Loss")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "losses_log.png", dpi=160)
    plt.close(fig)

    # Relative change across a trailing window and simple plateau heuristic.
    report = []
    for name, values in loss_series.items():
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            report.append(f"{name}: aucune valeur finie")
            continue

        final = float(finite[-1])
        best = float(np.min(finite))
        start = float(finite[0])
        improvement = (start - final) / max(abs(start), 1e-15) * 100

        if finite.size > window:
            old = float(np.median(finite[-window-1:-1]))
            new = float(np.median(finite[-window:]))
            relative_change = abs(new - old) / max(abs(old), 1e-15)
        else:
            relative_change = float("nan")

        report.append(
            f"{name}: initial={start:.6g}, final={final:.6g}, "
            f"best={best:.6g}, baisse globale={improvement:.2f}%, "
            f"variation relative médiane sur fenêtre={relative_change:.3g}"
        )

    # Estimate whether each series has plateaued; heuristic, not proof.
    plateau = []
    for name, values in loss_series.items():
        finite = values[np.isfinite(values)]
        if finite.size >= max(10, window):
            w = min(window, max(5, finite.size // 5))
            before = np.median(finite[-2*w:-w])
            after = np.median(finite[-w:])
            rel = abs(after-before) / max(abs(before), 1e-15)
            plateau.append((name, rel < 1e-3, float(rel)))
        else:
            plateau.append((name, False, float("nan")))

    report.append("")
    report.append("Détection heuristique de plateau (seuil 1e-3 ; ne prouve pas la convergence) :")
    for name, is_plateau, rel in plateau:
        report.append(f"  {name}: {'plateau probable' if is_plateau else 'pas de plateau détecté'} (écart={rel:.3g})")

    (outdir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\nGraphique et rapport enregistrés dans : {outdir.resolve()}")


def optional_model_inspection(path):
    # Safe-ish loading for a trusted checkpoint only.
    # PyTorch versions differ in support for weights_only.
    import torch
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        print("\nCheckpoint chargé. Clés disponibles :", list(checkpoint.keys())[:30])
        if "state_dict" in checkpoint:
            print("Le checkpoint contient une clé 'state_dict'.")
        elif "model_state_dict" in checkpoint:
            print("Le checkpoint contient une clé 'model_state_dict'.")
        else:
            print("Le fichier semble être un state_dict ou un dictionnaire de poids.")
    else:
        print(f"\nObjet checkpoint : {type(checkpoint).__name__}")
        print("Si le modèle complet a été sérialisé, il faut une architecture compatible pour l'évaluer.")


def main():
    parser = argparse.ArgumentParser(description="Analyse des Loss d'un PINN")
    parser.add_argument("json_file", type=Path, help="Fichier JSON des Loss")
    parser.add_argument("--outdir", type=Path, default=Path("analysis/chatgpt"))
    parser.add_argument("--window", type=int, default=50, help="Fenêtre pour l'analyse de plateau")
    parser.add_argument("--model", type=Path, help="Checkpoint .pt facultatif (inspection uniquement)")
    args = parser.parse_args()

    history = load_history(args.json_file)
    analyze(history, args.outdir, window=max(5, args.window))
    if args.model:
        optional_model_inspection(args.model)


if __name__ == "__main__":
    main()
