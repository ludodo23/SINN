import json
import glob
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch

# SINN project modules
import sys
ROOT = Path(__file__).resolve().parent.parent
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from visualization import load_model, build_sinn_data_from_model, plot_fields


# ---------------------------------------------------------------------------
# Helper: liste des modèles entraînés
# ---------------------------------------------------------------------------
def list_trained_models(base_dir: Path = Path("models")):
    """Return a list of model dicts with name, checkpoint, config, history."""
    models = []
    print(f"Searching for trained models in {base_dir}")
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist.")
        return models
    for model_dir in sorted(base_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        ckpt = model_dir / "checkpoint_final.pt"
        cfg = model_dir / "config.json"
        hist = model_dir / "history.json"
        if ckpt.exists() and cfg.exists():
            # extraire quelques infos du config
            try:
                with open(cfg, encoding="utf-8") as f:
                    cfg_obj = json.load(f)
                use_fourier = cfg_obj.get("use_fourier_features", False)
                use_adaptive = cfg_obj.get("use_adaptive_sampling", False)
                models.append(
                    {
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "checkpoint": str(ckpt),
                        "config": str(cfg),
                        "use_fourier_features": use_fourier,
                        "use_adaptive_sampling": use_adaptive,
                        "history": str(hist) if hist.exists() else None,
                    }
                )
            except Exception:
                models.append(
                    {
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "checkpoint": str(ckpt),
                        "config": str(cfg),
                    }
                )
    print(f"Found {len(models)} trained models.")
    return models
# Page : Pertes d'entraînement
# ---------------------------------------------------------------------------
def page_losses(models):
    st.subheader("📊 Historique des pertes d'entraînement")
    if not models:
        st.info("Aucun modèle entraîné trouvé.")
        return
    # Sélection du modèle
    model_names = [m["name"] for m in models]
    selected_name = st.selectbox("Modèle", model_names, key="loss_model")
    selected = [m for m in models if m["name"] == selected_name][0]

    hist_path = selected.get("history")
    if hist_path and Path(hist_path).exists():
        try:
            with open(hist_path, encoding="utf-8") as f:
                history = json.load(f)
            # history attendu: [{'epoch': 1, 'total': ..., ...}, ...]
            epochs = [row.get("epoch", i) for i, row in enumerate(history)]
            # Ontrace les principales pertes
            loss_keys = [k for k in history[0].keys() if k != "epoch"]
            fig_loss, ax = plt.subplots(figsize=(10, 5))
            for key in loss_keys:
                vals = [row.get(key, 0) for row in history]
                ax.plot(epochs, vals, label=key)
            ax.set_xlabel("Époque")
            ax.set_ylabel("Perte")
            ax.set_title(f"Historique des pertes – {selected_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig_loss)
        except Exception as e:
            st.error(f"Erreur lors du chargement de l'historique : {e}")
    else:
        st.info("Aucun fichier history.json trouvé pour ce modèle.")


# ---------------------------------------------------------------------------
# Page : Champs (single plot)
# ---------------------------------------------------------------------------
def page_fields(models):
    st.subheader("🌊 Champs prédits par le PINN")
    if not models:
        st.info("Aucun modèle entraîné trouvé.")
        return

    model_names = [m["name"] for m in models]
    selected_name = st.selectbox("Modèle", model_names, key="field_model")
    selected = [m for m in models if m["name"] == selected_name][0]

    # Charger le modèle
    try:
        ckpt = selected["checkpoint"]
        cfg_path = selected["config"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, cfg = load_model(checkpoint=ckpt, config=cfg_path, device=device)
        model.eval()
    except Exception as e:
        st.error(f"Impossible de charger le modèle : {e}")
        return

    # Choix du champ
    field = st.selectbox("Champ", ("u", "v", "p", "alpha"), key="field_champ")
    t_default = float(cfg.t_min) if cfg else 0.0
    t = st.slider("Temps t", float(cfg.t_min), float(cfg.t_max), value=t_default, key="field_t")

    # Charger les données OpenFOAM de référence (si disponibles)
    openfoam_data = Path("data/sloshing_data.npz")
    of_available = openfoam_data.exists()

    if st.button("Générer le champ", key="field_btn"):
        with st.spinner("Chargement en cours..."):
            try:
                if of_available:
                    # On utilise plot_fields qui gère OpenFOAM + PINN en 3 panneaux
                    fig = plot_fields(
                        model=model,
                        t=t,
                        nx=180,
                        ny=80,
                        fields=field,
                        openfoam_data=str(openfoam_data),
                    )
                    st.pyplot(fig)
                else:
                    # Simple affichage PINN seulement
                    # Créer un mesh grid
                    x_min = -1.0 if cfg is None else float(cfg.x_min)
                    x_max = 6.0 if cfg is None else float(cfg.x_max)
                    y_min = -1.0 if cfg is None else float(cfg.y_min)
                    y_max = 1.0 if cfg is None else float(cfg.y_max)

                    nx, ny = 200, 100
                    x = np.linspace(x_min, x_max, nx)
                    y = np.linspace(y_min, y_max, ny)
                    X, Y = np.meshgrid(x, y)
                    T = np.full_like(X, t)
                    coords_np = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1)
                    coords = torch.tensor(
                        coords_np, dtype=torch.float32, device=device
                    )
                    with torch.no_grad():
                        prediction = model.predict_physical(coords)
                    Z = prediction[field].cpu().numpy().reshape(ny, nx)

                    fig, ax = plt.subplots(figsize=(8, 5))
                    im = ax.contourf(X, Y, Z, levels=40)
                    cbar = fig.colorbar(im, ax=ax, label=field)
                    ax.set_xlabel("x [m]")
                    ax.set_ylabel("y [m]")
                    ax.set_title(f"{field} t = {t:.4f} s – {selected_name}")
                    ax.set_aspect("equal", adjustable="box")
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Erreur : {e}")


# ---------------------------------------------------------------------------
# Page : Comparaison OpenFOAM vs PINN (animation)
# ---------------------------------------------------------------------------
def page_comparison(models):
    st.subheader("📈 Comparaison OpenFOAM ↔ PINN")

    if not models:
        st.info("Aucun modèle entraîné trouvé.")
        return

    model_names = [m["name"] for m in models]
    selected_name = st.selectbox("Modèle", model_names, key="comp_model")
    selected = [m for m in models if m["name"] == selected_name][0]

    # Choix du champ
    field = st.selectbox("Champ", ("alpha", "u", "v", "p"), key="comp_field")

    # Charger le modèle
    try:
        ckpt = selected["checkpoint"]
        cfg_path = selected["config"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, cfg = load_model(checkpoint=ckpt, config=cfg_path, device=device)
        model.eval()
    except Exception as e:
        st.error(f"Impossible de charger le modèle : {e}")
        return

    # Données OpenFOAM
    openfoam_data = Path("data/sloshing_data.npz")
    of_available = openfoam_data.exists()

    if st.button("Générer l'animation", key="comp_btn"):
        with st.spinner("Génération de l'animation en cours..."):
            try:
                if not of_available:
                    st.error(
                        "Fichier OpenFOAM introuvable : `data/sloshing_data.npz`."
                    )
                    return

                # Préparer les données SINN via build_sinn_data_from_model
                sinn_data = build_sinn_data_from_model(model, str(openfoam_data))

                # Lancer l'animation
                from visualization import animate_comparison

                # On génère l'animation mais on l'exporte en GIF et on l'affiche en HTML
                anim, fig = animate_comparison(
                    openfoam_data=str(openfoam_data),
                    sinn_data=None,
                    model=model,
                    save_path=None,
                    field=field,
                    interval=200,
                    show_forces=True,
                    display_animation=False,
                )

                # Afficher le HTML de l'animation
                st.success("Animation générée !")
                # Convertir en HTML pour Streamlit
                html = anim.to_jshtml()
                st.components.v1.html(html, height=600, width=None)

                # Optionnel : exporter en GIF
                gif_path = f"comparison_{selected_name}_{field}.gif"
                try:
                    anim.save(gif_path, writer="pillow", fps=1000 / 200)
                    st.download_button(
                        "Télécharger le GIF",
                        data=open(gif_path, "rb").read(),
                        file_name=gif_path,
                        mime="image/gif",
                    )
                except Exception as e:
                    st.warning(f"Impossible d'exporter le GIF : {e}")

            except Exception as e:
                st.error(f"Erreur lors de la génération : {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="SINN Dashboard", layout="wide")
    st.title("🧊 SINN — Tableau de bord des résultats")

    # Liste des modèles
    models = list_trained_models()

    # Onglets
    tab1, tab2, tab3 = st.tabs(["📊 Pertes", "🌊 Champs", "📈 Comparaison"])

    with tab1:
        page_losses(models)
    with tab2:
        page_fields(models)
    with tab3:
        page_comparison(models)


if __name__ == "__main__":
    main()