import json

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import numpy as np
import torch

from IPython.display import HTML, display

from model import SloshingPINN


def plot_history(path="outputs/loss_history.json", use_same_figure=True):
    with open(path, encoding="utf-8") as file:
        history = json.load(file)

    epochs = [row["epoch"] for row in history]

    loss_names = [
        key for key in history[0]
        if key != "epoch"
    ]

    if use_same_figure:
        fig, ax = plt.subplots()


    for name in loss_names:
        values = [
            max(row[name], 1e-16)
            for row in history
        ]

        if not use_same_figure:
            ax, fig = plt.subplots()
            ax.set_ylabel(name)
        else:
            ax.set_ylabel("Perte (échelle logarithmique)")

        ax.semilogy(epochs, values, label=name)
        ax.set_xlabel("Époque")
        ax.set_title(f"Historique de la perte : {name}")
        ax.grid(True)
        fig.tight_layout()
    if use_same_figure:
        ax.legend()
    plt.show()


def load_model(
    checkpoint="outputs/sloshing_pinn.pt",
    config=None,
    device=None,
):
    device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Construit le modèle et charge les poids via l'API recommandée PyTorch
    # (state_dict) directement dans __init__.
    model = SloshingPINN(state_path=checkpoint, config=config).to(device)

    # Si aucune config n'a été fournie, tente de la récupérer depuis le checkpoint.
    if model.cfg is None:
        saved = torch.load(
            checkpoint,
            map_location=device,
            weights_only=False,
        )
        if isinstance(saved, dict) and "config" in saved:
            model.cfg = saved["config"]

    model.eval()
    return model, device, model.cfg


@torch.no_grad()
def plot_fields(
    model: SloshingPINN,
    t=0.0,
    nx=180,
    ny=80,
    fields=("u", "v", "p", "alpha"),
    openfoam_data=None,
):
    """Affiche les champs prédits par le modèle PINN à l'instant t.

    Paramètres
    ----------
    model : SloshingPINN
        Modèle déjà chargé (obligatoire). La configuration est récupérée
        depuis model.cfg et les prédictions passent par predict_physical
        (coordonnées SI).
    t : float
        Instant d'affichage en secondes.
    nx, ny : int
        Résolution de la grille d'affichage.
    fields : tuple[str]
        Noms des champs à afficher (parmi "u", "v", "p", "alpha").
    openfoam_data : str | dict | None
        Si fourni, pour chaque champ affiche trois graphes en ligne :
        OpenFOAM, modèle PINN et erreur relative (échelle 0 à 1).
    """
    cfg = model.cfg
    if cfg is None:
        raise ValueError(
            "Le modèle ne possède pas de configuration (model.cfg est None)."
        )

    if isinstance(cfg, dict):
        x_min = float(cfg["x_min"])
        x_max = float(cfg["x_max"])
        y_min = float(cfg["y_min"])
        y_max = float(cfg["y_max"])
    else:
        x_min = float(cfg.x_min)
        x_max = float(cfg.x_max)
        y_min = float(cfg.y_min)
        y_max = float(cfg.y_max)

    device = next(model.parameters()).device
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)

    X, Y = np.meshgrid(x, y)
    T = np.full_like(X, t)

    coords_np = np.stack(
        [X.ravel(), Y.ravel(), T.ravel()],
        axis=1,
    )

    coords = torch.tensor(
        coords_np,
        dtype=torch.float32,
        device=device,
    )

    prediction = model.predict_physical(coords)

    # Chargement éventuel des données OpenFOAM pour comparaison.
    of = None
    if openfoam_data is not None:
        of = _load_animation_data(openfoam_data)
        if "X" not in of or "time" not in of:
            raise ValueError(
                "openfoam_data doit contenir les clés 'X' et 'time'."
            )
        X_of = np.asarray(of["X"])
        t_of = np.asarray(of["time"]).reshape(-1)
        # Trouver l'instant OpenFOAM le plus proche de t.
        i_t = int(np.argmin(np.abs(t_of - t)))

    for field in fields:
        Z = prediction[field].cpu().numpy().reshape(ny, nx)

        if of is None:
            plt.figure()
            plt.contourf(X, Y, Z, levels=40)
            plt.colorbar(label=field)
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.title(f"{field} à t = {t:g} s")
            plt.tight_layout()
            plt.show()
        else:
            # Trois graphes : OpenFOAM, modèle, erreur relative.
            Z_of = np.asarray(of[field])[i_t].reshape(ny, nx)
            with np.errstate(divide="ignore", invalid="ignore"):
                denom = np.maximum(np.abs(Z_of), 1e-12)
                err = np.abs(Z - Z_of) / denom
                err = np.clip(err, 0.0, 1.0)

            fig, axes = plt.subplots(
                1, 3, figsize=(15, 4), sharex=True, sharey=True
            )

            im0 = axes[0].contourf(X, Y, Z_of, levels=40)
            axes[0].set_title("OpenFOAM")
            fig.colorbar(im0, ax=axes[0], label=field, fraction=0.046)

            im1 = axes[1].contourf(X, Y, Z, levels=40)
            axes[1].set_title("SINN (modèle)")
            fig.colorbar(im1, ax=axes[1], label=field, fraction=0.046)

            im2 = axes[2].contourf(X, Y, err, levels=40, vmin=0, vmax=1)
            axes[2].set_title("Erreur relative")
            fig.colorbar(im2, ax=axes[2], label="|m−o|/max(|o|,eps)", fraction=0.046)

            for ax in axes:
                ax.set_xlabel("x [m]")
                ax.set_ylabel("y [m]")
                ax.set_aspect("equal", adjustable="box")
                ax.grid(alpha=0.15)

            fig.suptitle(f"{field} à t = {t:g} s")
            fig.tight_layout()
            plt.show()


def _load_animation_data(data):
    """Charge un dataset depuis un chemin .npz ou un dictionnaire."""
    if isinstance(data, (str, bytes)):
        with np.load(data) as file:
            return {key: file[key] for key in file.files}

    if isinstance(data, dict):
        return data

    # Permet aussi de passer directement un objet NpzFile
    return {key: data[key] for key in data.files}


def build_sinn_data_from_model(
    model: SloshingPINN,
    openfoam_data,
    batch_size: int = 65536,
) -> dict:
    """Évalue le modèle PINN aux mêmes (x, y, t) que le dataset OpenFOAM.

    Les coordonnées sont en unités SI ; l'appel passe par
    model.predict_physical. Retourne un dictionnaire sinn_data utilisable
    directement par animate_comparison.

    Paramètres
    ----------
    model : SloshingPINN
        Modèle déjà chargé (doit posséder une configuration model.cfg).
    openfoam_data : str | dict
        Chemin vers un .npz ou dictionnaire contenant X et time.
    batch_size : int
        Taille des lots pour l'inférence.

    Retour
    ------
    dict
        Dictionnaire avec les clés "time", "X", "alpha_eau".
    """
    of = _load_animation_data(openfoam_data)
    X_of = np.asarray(of["X"], dtype=np.float32)      # (T, N, >=2)
    times = np.asarray(of["time"], dtype=np.float32).reshape(-1)

    n_times, n_points = X_of.shape[0], X_of.shape[1]
    device = next(model.parameters()).device
    model.eval()

    field_names = ("u", "v", "p", "alpha")
    predictions = {
        name: np.empty((n_times, n_points), dtype=np.float32)
        for name in field_names
    }

    with torch.inference_mode():
        for i, t in enumerate(times):
            xy = X_of[i, :, :2]
            for start in range(0, n_points, batch_size):
                end = min(start + batch_size, n_points)
                coords_xy = xy[start:end]
                t_column = np.full(
                    (end - start, 1), float(t), dtype=np.float32
                )
                coords_np = np.concatenate([coords_xy, t_column], axis=1)
                coords = torch.as_tensor(
                    coords_np, dtype=torch.float32, device=device
                )
                pred = model.predict_physical(coords)
                for name in field_names:
                    values = (
                        pred[name].detach().cpu().numpy().reshape(-1)
                    )
                    predictions[name][i, start:end] = values

    return {
        "time": times,
        "X": X_of,
        "alpha_eau": predictions["alpha"],
    }


def animate_comparison(
    openfoam_data,
    sinn_data=None,
    model=None,
    save_path=None,
    field="alpha_eau",
    xlim=(-2, 8),
    ylim=None,
    interval=30,
    point_size=10,
    show_forces=True,
    display_animation=True,
):
    """
    Compare les champs OpenFOAM et SINN côte à côte.

    Paramètres
    ----------
    openfoam_data : str ou dict
        Chemin vers un .npz ou dictionnaire contenant X, alpha_eau,
        time et éventuellement Fx.
    sinn_data : str ou dict, optionnel
        Même structure pour les prédictions SINN. Si None, un modèle
        doit être fourni via model.
    model : SloshingPINN, optionnel
        Modèle déjà chargé. Si sinn_data est None, les prédictions SINN
        sont générées à la volée à partir de ce modèle (via
        build_sinn_data_from_model).
    save_path : str ou None
        Chemin de sortie .gif ou .mp4.
    field : str
        Nom du champ scalaire à comparer, par défaut 'alpha_eau'.
    xlim, ylim : tuple
        Limites spatiales communes aux deux panneaux.
    interval : int
        Durée entre images en millisecondes.
    point_size : float
        Taille des points du scatter.
    show_forces : bool
        Affiche les courbes Fx si les deux datasets les contiennent.
    display_animation : bool
        Affiche l'animation dans un notebook Jupyter.

    Retour
    ------
    animation : FuncAnimation
    fig : Figure
    """

    if sinn_data is None and model is None:
        raise ValueError(
            "Fournir soit sinn_data, soit model (au moins l'un des deux)."
        )
    if sinn_data is not None and model is not None:
        raise ValueError(
            "Fournir soit sinn_data, soit model, mais pas les deux."
        )

    of = _load_animation_data(openfoam_data)

    if sinn_data is None:
        sinn = build_sinn_data_from_model(model, openfoam_data)
    else:
        sinn = _load_animation_data(sinn_data)

    # --------------------------------------------------------
    # Chargement des données
    # --------------------------------------------------------
    for name, data in [("OpenFOAM", of), ("SINN", sinn)]:
        if "X" not in data or "time" not in data:
            raise ValueError(
                f"{name} doit contenir les clés 'X' et 'time'."
            )

        if field not in data:
            raise ValueError(
                f"Champ '{field}' absent du dataset {name}."
            )

    X_of = np.asarray(of["X"])
    X_sinn = np.asarray(sinn["X"])

    A_of = np.asarray(of[field])
    A_sinn = np.asarray(sinn[field])

    t_of = np.asarray(of["time"]).reshape(-1)
    t_sinn = np.asarray(sinn["time"]).reshape(-1)

    if X_of.ndim != 3 or X_of.shape[-1] < 2:
        raise ValueError("OpenFOAM : X doit avoir la forme (T, N, 2).")

    if X_sinn.ndim != 3 or X_sinn.shape[-1] < 2:
        raise ValueError("SINN : X doit avoir la forme (T, N, 2).")

    if A_of.shape[:2] != X_of.shape[:2]:
        raise ValueError("OpenFOAM : dimensions de X et du champ incompatibles.")

    if A_sinn.shape[:2] != X_sinn.shape[:2]:
        raise ValueError("SINN : dimensions de X et du champ incompatibles.")

    if len(t_of) != len(X_of) or len(t_sinn) != len(X_sinn):
        raise ValueError("Les vecteurs de temps doivent correspondre à X.")

    if len(t_of) == 0 or len(t_sinn) == 0:
        raise ValueError("Les datasets ne doivent pas être vides.")

    if not np.all(np.isfinite(t_of)) or not np.all(np.isfinite(t_sinn)):
        raise ValueError("Les vecteurs de temps contiennent des valeurs non finies.")

    # On garde les temps OpenFOAM compris dans l'intervalle SINN.
    valid = (t_of >= t_sinn.min()) & (t_of <= t_sinn.max())
    frame_indices = np.flatnonzero(valid)

    if len(frame_indices) == 0:
        raise ValueError(
            "Les intervalles de temps OpenFOAM et SINN ne se recouvrent pas."
        )

    times = t_of[frame_indices]

    # Associe à chaque temps OpenFOAM l'instant SINN le plus proche.
    sinn_indices = np.abs(
        times[:, None] - t_sinn[None, :]
    ).argmin(axis=1)

    # --------------------------------------------------------
    # Figure : deux champs en haut, forces en bas
    # --------------------------------------------------------
    has_forces = (
        show_forces
        and "Fx" in of
        and "Fx" in sinn
    )

    if has_forces:
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(
            2, 2,
            height_ratios=[3, 1],
            hspace=0.28,
            wspace=0.12,
        )
        ax_of = fig.add_subplot(gs[0, 0])
        ax_sinn = fig.add_subplot(gs[0, 1])
        ax_force = fig.add_subplot(gs[1, :])
    else:
        fig, (ax_of, ax_sinn) = plt.subplots(
            1, 2, figsize=(15, 5),
            sharex=True, sharey=True,
        )

    # --------------------------------------------------------
    # Champs scalaires : même échelle de couleurs
    # --------------------------------------------------------
    i0 = frame_indices[0]
    j0 = sinn_indices[0]

    scatter_of = ax_of.scatter(
        X_of[i0, :, 0],
        X_of[i0, :, 1],
        c=A_of[i0],
        vmin=0,
        vmax=1,
        s=point_size,
        cmap="coolwarm",
    )

    scatter_sinn = ax_sinn.scatter(
        X_sinn[j0, :, 0],
        X_sinn[j0, :, 1],
        c=A_sinn[j0],
        vmin=0,
        vmax=1,
        s=point_size,
        cmap="coolwarm",
    )

    ax_of.set_title("OpenFOAM — interFoam")
    ax_sinn.set_title("SINN — PINN")

    for ax in (ax_of, ax_sinn):
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.15)

    fig.colorbar(
        scatter_sinn,
        ax=[ax_of, ax_sinn],
        label=field,
        fraction=0.025,
        pad=0.03,
    )

    title = fig.suptitle(f"Comparaison à t = {times[0]:.4f} s")

    # --------------------------------------------------------
    # Forces Fx
    # --------------------------------------------------------
    if has_forces:
        Fx_of = np.asarray(of["Fx"]).reshape(-1)
        Fx_sinn = np.asarray(sinn["Fx"]).reshape(-1)

        if len(Fx_of) != len(t_of) or len(Fx_sinn) != len(t_sinn):
            raise ValueError("Chaque série Fx doit avoir la longueur de son vecteur time.")

        ax_force.plot(t_of, Fx_of, label="OpenFOAM", linewidth=1.5)
        ax_force.plot(t_sinn, Fx_sinn, label="SINN", linewidth=1.5)

        marker_of, = ax_force.plot(
            [times[0]], [Fx_of[i0]], "o", markersize=6
        )
        marker_sinn, = ax_force.plot(
            [times[0]], [Fx_sinn[j0]], "o", markersize=6
        )

        ax_force.set_xlim(times[0], times[-1])
        ax_force.set_xlabel("t [s]")
        ax_force.set_ylabel("$F_x$ [N]")
        ax_force.grid(True, alpha=0.3)
        ax_force.legend()

        fmin = min(np.nanmin(Fx_of), np.nanmin(Fx_sinn))
        fmax = max(np.nanmax(Fx_of), np.nanmax(Fx_sinn))
        margin = 0.05 * (fmax - fmin) if fmax > fmin else 1.0
        ax_force.set_ylim(fmin - margin, fmax + margin)

    # --------------------------------------------------------
    # Mise à jour de l'animation
    # --------------------------------------------------------
    def update(frame):
        i = frame_indices[frame]
        j = sinn_indices[frame]

        scatter_of.set_offsets(X_of[i, :, :2])
        scatter_of.set_array(A_of[i].reshape(-1))

        scatter_sinn.set_offsets(X_sinn[j, :, :2])
        scatter_sinn.set_array(A_sinn[j].reshape(-1))

        title.set_text(
            f"Comparaison — t = {t_of[i]:.4f} s "
            f"(SINN : {t_sinn[j]:.4f} s)"
        )

        artists = [scatter_of, scatter_sinn, title]

        if has_forces:
            marker_of.set_data([t_of[i]], [Fx_of[i]])
            marker_sinn.set_data([t_of[i]], [Fx_sinn[j]])
            artists.extend([marker_of, marker_sinn])

        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=interval,
        blit=False,
    )

    # --------------------------------------------------------
    # Export facultatif
    # --------------------------------------------------------
    if save_path is not None:
        path = str(save_path)
        if path.lower().endswith(".gif"):
            animation.save(path, writer=PillowWriter(fps=1000 / interval))
        elif path.lower().endswith(".mp4"):
            animation.save(path, writer=FFMpegWriter(fps=1000 / interval))
        else:
            raise ValueError("Le fichier de sortie doit être .gif ou .mp4")

    if display_animation:
        plt.close(fig)
        display(HTML(animation.to_jshtml()))

    return animation, fig
