import json
import glob
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import torch

from plotly.subplots import make_subplots

# SINN project modules
import sys
ROOT = Path(__file__).resolve().parent.parent
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from visualization import load_model, build_sinn_data_from_model


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

# ---------------------------------------------------------------------------
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
            
            # Créer la figure Plotly
            fig = go.Figure()
            for key in loss_keys:
                vals = [row.get(key, 0) for row in history]
                fig.add_trace(go.Scatter(
                    x=epochs, 
                    y=vals, 
                    mode='lines+markers', 
                    name=key,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                xaxis_title="Époque",
                yaxis_title="Perte",
                title=f"Historique des pertes – {selected_name}",
                hovermode='x unified',
                template="streamlit",
                showlegend=True,
                # grid=dict(theta=0)
            )
            fig.update_xaxes(gridcolor='rgba(0,0,0,0.1)')
            fig.update_yaxes(gridcolor='rgba(0,0,0,0.1)')
            
            st.plotly_chart(fig, width='stretch')
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
                    # Charger les données OpenFOAM pour calibration du grid
                    of = np.load(openfoam_data, allow_pickle=True)
                    X_of = of['X']  # shape (T, N, 3) avec x, y, t
                    t_of = of['time']
                    # Grid calibration: unique sorted x and y
                    xs = np.unique(np.round(X_of[0, :, 0], 9))
                    ys = np.unique(np.round(X_of[0, :, 1], 9))
                    x = xs
                    y = ys
                    nx = len(x)
                    ny = len(y)

                    # Prédiction PINN sur le grid calibré
                    X, Y = np.meshgrid(x, y)
                    T = np.full_like(X, t)
                    coords_np = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1)
                    coords = torch.tensor(
                        coords_np, dtype=torch.float32, device=device
                    )
                    with torch.no_grad():
                        prediction = model.predict_physical(coords)

                    Z_pinn = prediction[field].cpu().numpy().reshape(ny, nx)

                    # Données OpenFOAM correspondantes
                    i_t = int(np.argmin(np.abs(t_of - t)))
                    Z_of = np.asarray(of[field])[i_t].reshape(ny, nx, order='F')

                    # Relative error
                    with np.errstate(divide='ignore', invalid='ignore'):
                        denom = np.maximum(np.abs(Z_of), 1e-12)
                        rel_err = np.abs(Z_pinn - Z_of) / denom
                        rel_err = np.clip(rel_err, 0.0, 1.0)

                    # Figure Plotly avec 3 panneaux contour
                    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=('OpenFOAM', 'PINN', 'Relative error'),
                        horizontal_spacing=0.05)

                    # OpenFOAM contour
                    fig.add_trace(go.Contour(
                        z=Z_of,
                        x=x,
                        y=y,
                        colorscale='RdBu',
                        showscale=True,
                        colorbar=dict(title=field, xanchor='left'),
                        name='OpenFOAM'
                    ), row=1, col=1)

                    # PINN contour
                    fig.add_trace(go.Contour(
                        z=Z_pinn,
                        x=x,
                        y=y,
                        colorscale='RdBu',
                        showscale=True,
                        colorbar=dict(title=field, xanchor='center'),
                        name='PINN'
                    ), row=1, col=2)

                    # Relative error contour
                    fig.add_trace(go.Contour(
                        z=rel_err,
                        x=x,
                        y=y,
                        colorscale='RdBu_r',
                        showscale=False,
                        name='Error'
                    ), row=1, col=3)

                    fig.update_layout(
                        title=f"{field} t = {t:.4f} s – {selected_name}",
                        template="streamlit",
                        height=500,
                    )

                    fig.update_xaxes(title_text="x [m]", row=1, col=1)
                    fig.update_yaxes(title_text="y [m]", row=1, col=1)
                    fig.update_xaxes(title_text="x [m]", row=1, col=2)
                    fig.update_yaxes(title_text="y [m]", row=1, col=2)
                    fig.update_xaxes(title_text="x [m]", row=1, col=3)
                    fig.update_yaxes(title_text="y [m]", row=1, col=3)

                    st.plotly_chart(fig, width='stretch')
                else:
                    # Simple affichage PINN seulement avec Plotly - contour
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

                    # Figure Plotly contour
                    fig = go.Figure(data=go.Contour(
                        z=Z,
                        x=x,
                        y=y,
                        colorscale='RdBu',
                        colorbar=dict(label=field),
                    ))
                    fig.update_layout(
                        title=f"{field} t = {t:.4f} s – {selected_name}",
                        xaxis_title="x [m]",
                        yaxis_title="y [m]",
                        xaxis=dict(scaleanchor="y", scaleratio=1),
                        template="streamlit",
                        height=500,
                    )
                    fig.update_yaxes(side="right")
                    st.plotly_chart(fig, width='stretch')
            except Exception as e:
                st.error(f"Erreur : {e}")


# ---------------------------------------------------------------------------
# Fonction d'animation Plotly interactive
def display_plotly_animation(openfoam_data, model, field, interval=200, show_forces=True):
    """Afficher une animation Plotly interactive comparant OpenFOAM et PINN avec contours."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    # Charger les données OpenFOAM
    of = np.load(openfoam_data, allow_pickle=True)
    X_of = of['X']  # shape (T, N, 3) avec x, y, t
    t_of = of['time']
    A_of = of[field]
    
    # Générer les prédictions SINN
    sinn_data = build_sinn_data_from_model(model, str(openfoam_data))
    X_sinn = sinn_data['X']
    A_sinn = sinn_data[field]
    t_sinn = sinn_data['time']
    
    # Calibration du grid sur les données OpenFOAM (mêmes x,y pour les deux)
    xs = np.unique(np.round(X_of[0, :, 0], 9))
    ys = np.unique(np.round(X_of[0, :, 1], 9))
    x = xs
    y = ys
    nx = len(x)
    ny = len(y)
    
    # Reshape les données en grille pour contour plot
    # Les données OpenFOAM sont stockées sous forme de points irréguliers,
    # on les reshape en utilisant l'ordre Fortran pour matcher la création du grid
    A_of_grid = np.zeros((len(t_of), ny, nx))
    A_sinn_grid = np.zeros((len(t_sinn), ny, nx))
    
    for i in range(len(t_of)):
        # OpenFOAM: reshape avec order='F' pour matcher meshgrid (x varies slowest)
        A_of_grid[i] = A_of[i].reshape(ny, nx, order='F')
    
    for i in range(len(t_sinn)):
        A_sinn_grid[i] = A_sinn[i].reshape(ny, nx, order='F')
    
    n_frames = min(len(t_of), len(t_sinn))
    
    # Créer la figure avec 2 sous-graphiques pour les contours
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('OpenFOAM', 'PINN'),
                        horizontal_spacing=0.1)
    
    # Contour plots initiaux (frame 0)
    trace_of = go.Contour(
        z=A_of_grid[0],
        x=x,
        y=y,
        colorscale='RdBu',
        showscale=True,
        colorbar=dict(title=field, xanchor='left'),
        name='OpenFOAM'
    )
    trace_sinn = go.Contour(
        z=A_sinn_grid[0],
        x=x,
        y=y,
        colorscale='RdBu',
        showscale=True,
        name='PINN'
    )
    
    fig.add_trace(trace_of, row=1, col=1)
    fig.add_trace(trace_sinn, row=1, col=2)
    
    # Fonction pour créer les traces d'une frame donnée
    def make_frame_traces(frame_idx):
        trace_of = go.Contour(
            z=A_of_grid[frame_idx],
            x=x,
            y=y,
            colorscale='RdBu',
            showscale=False,
            name='OF',
            showlegend=False
        )
        trace_sinn = go.Contour(
            z=A_sinn_grid[frame_idx],
            x=x,
            y=y,
            colorscale='RdBu',
            showscale=False,
            name='PINN',
            showlegend=False
        )
        return [trace_of, trace_sinn]
    
    # Créer les frames
    frames = []
    for i in range(n_frames):
        frame_data = make_frame_traces(i)
        frame = go.Frame(data=frame_data, name=str(i))
        frames.append(frame)
    
    fig.frames = frames
    
    # Mise en page avec slider et boutons de contrôle
    fig.update_layout(
        title=f"Comparaison {field} - Animation interactive",
        template="streamlit",
        sliders=[{
            'steps': [{'label': f'T={t_of[i]:.2f}s', 'method': 'animate', 'args': [[i]]} for i in range(n_frames)],
            'currentvalue': {'prefix': 'Temps: ', 'visible': True},
            'len': 0.9,
            'x': 0.1,
            'yanchor': 'middle'
        }],
        updatemenus=[{
            'type': 'buttons',
            'direction': 'left',
            'buttons': [
                {'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': interval, 'redraw': True}}]},
                {'label': '⏸ Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}}]},
            ]
        }],
        height=600,
        showlegend=False
    )
    
    # Mettre à jour les axes
    fig.update_xaxes(title_text="x [m]", row=1, col=1)
    fig.update_yaxes(title_text="y [m]", row=1, col=1)
    fig.update_xaxes(title_text="x [m]", row=1, col=2)
    fig.update_yaxes(title_text="y [m]", row=1, col=2)
    
    return fig


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

                # Utiliser display_plotly_animation pour une animation Plotly interactive
                fig_plotly = display_plotly_animation(
                    openfoam_data=str(openfoam_data),
                    model=model,
                    field=field,
                    interval=200,
                    show_forces=True,
                )
                st.success("Animation générée !")
                st.plotly_chart(fig_plotly, width='stretch')
            
            except Exception as e:
                st.error(f"Erreur lors de la génération : {e}")


# ---------------------------------------------------------------------------
# Page : Benchmarks
# ---------------------------------------------------------------------------
def page_benchmarks(models):
    """Page de visualisation des benchmarks des modèles."""
    st.subheader("📉 Benchmarks des modèles")
    
    # Vérifier la présence des fichiers de benchmark
    from pathlib import Path
    import json
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    
    summary_path = Path("reports/summary.json")
    timings_path = Path("reports/timings.csv")
    model_table_path = Path("reports/model_table.json")
    model_table_csv_path = Path("reports/model_table.csv")
    
    files_available = []
    if summary_path.exists():
        files_available.append("summary.json")
    if timings_path.exists():
        files_available.append("timings.csv")
    if model_table_path.exists():
        files_available.append("model_table.json")
    if model_table_csv_path.exists():
        files_available.append("model_table.csv")
    
    if not files_available:
        st.info("Aucune donnée de benchmark disponible dans le dossier reports/.")
        st.write("Fichiers attendus : " + ", ".join(files_available))
        return
    
    st.write("Fichiers disponibles : " + ", ".join(files_available))
    
    # Charger les données
    df_summary = None
    df_timings = None
    df_model_table = None
    
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            df_summary = pd.DataFrame(json.load(f))
    
    if timings_path.exists():
        df_timings = pd.read_csv(timings_path)
    
    if model_table_path.exists():
        with open(model_table_path, encoding="utf-8") as f:
            df_model_table = pd.DataFrame(json.load(f))
    elif model_table_csv_path.exists():
        df_model_table = pd.read_csv(model_table_csv_path)
    
    # Section 1: Métriques du modèle sélectionné
    st.write("### Métriques du modèle sélectionné")
    
    if df_model_table is not None and "model_name" in df_model_table.columns:
        model_names = df_model_table["model_name"].tolist()
    elif df_summary is not None:
        model_names = df_summary["model_name"].tolist()
    else:
        model_names = [m["name"] for m in models] if models else []
    
    if not model_names:
        st.info("Aucun modèle trouvé.")
        return
    
    selected_model = st.selectbox("Modèle", model_names, key="bench_select_model")
    
    if df_summary is not None and selected_model in df_summary["model_name"].values:
        model_data = df_summary[df_summary["model_name"] == selected_model].iloc[0]
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("RMSE u", f"{float(model_data['mean_rmse_u']):.4f}")
        with c2: st.metric("RMSE v", f"{float(model_data['mean_rmse_v']):.4f}")
        with c3: st.metric("RMSE p", f"{float(model_data['mean_rmse_p']):.1f}")
        with c4: st.metric("RMSE α", f"{float(model_data['mean_rmse_alpha']):.4f}")
        
        if df_timings is not None and selected_model in df_timings["model_name"].values:
            t_data = df_timings[df_timings["model_name"] == selected_model].iloc[0]
            st.metric("Temps entraînement (s)", f"{float(t_data['train_time_s']):.2f}")
    
    # Section 2: Comparaison RMSE α
    st.write("### Comparaison RMSE α moyen par modèle")
    
    if df_summary is not None:
        fig_bar = go.Figure(go.Bar(
            y=df_summary["model_name"].tolist(),
            x=df_summary["mean_rmse_alpha"].astype(float).tolist(),
            orientation='h',
            marker_color='rgba(54, 162, 235, 0.6)'
        ))
        fig_bar.update_layout(
            title="RMSE α moyen par modèle",
            xaxis_title="RMSE α",
            yaxis_title="Modèle",
            template="streamlit",
            height=500
        )
        st.plotly_chart(fig_bar, width='stretch')
    
    # Section 3: Courbes d'erreur temporelles
    st.write("### Courbes d'erreur temporelles")
    
    if df_summary is not None and selected_model:
        errors_path = Path(f"reports/errors_{selected_model}.npz")
        if errors_path.exists():
            try:
                errors = np.load(errors_path, allow_pickle=True)
                t = errors["timestamps"].astype(float)
                for name, data in [("u", errors["rmse_u"]), ("v", errors["rmse_v"]), 
                                  ("p", errors["rmse_p"]), ("α", errors["rmse_alpha"])]:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=t, y=data.astype(float), mode='lines', name=name))
                    fig2.update_layout(title=f"Erreurs {selected_model}", template="streamlit")
                    st.plotly_chart(fig2, width='stretch')
            except Exception as e:
                st.warning(f"Impossible de charger les courbes d'erreur : {e}")
    
    # Section 4: Tableau des configurations
    st.write("### Tableau des configurations de modèles")
    
    if df_model_table is not None:
        st.dataframe(df_model_table, width='stretch')
    elif df_summary is not None:
        st.dataframe(df_summary, width='stretch')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="SINN Dashboard", layout="wide")
    st.title("🧊 SINN — Tableau de bord des résultats")

    # Liste des modèles
    models = list_trained_models()

    # Onglets
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Pertes", "🌊 Champs", "📈 Comparaison", "📉 Benchmarks"])

    with tab1:
        page_losses(models)
    with tab2:
        page_fields(models)
    with tab3:
        page_comparison(models)
    with tab4:
        page_benchmarks(models)



if __name__ == "__main__":
    main()