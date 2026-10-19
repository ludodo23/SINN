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
    device=None,
):
    device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    saved = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    cfg = saved["config"]

    model = SloshingPINN(
        width=cfg["hidden_width"],
        hidden_layers=cfg["hidden_layers"],
    ).to(device)

    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    return model, device, cfg


@torch.no_grad()
def plot_fields(
    t=0.0,
    nx=180,
    ny=80,
    checkpoint="outputs/sloshing_pinn.pt",
):
    model, device, cfg = load_model(checkpoint)

    x = np.linspace(cfg["x_min"], cfg["x_max"], nx)
    y = np.linspace(cfg["y_min"], cfg["y_max"], ny)

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

    prediction = model(coords)

    for field in ["u", "v", "p", "alpha"]: #, "k", "epsilon"]:
        Z = prediction[field].cpu().numpy().reshape(ny, nx)

        plt.figure()
        plt.contourf(X, Y, Z, levels=40)
        plt.colorbar(label=field)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title(f"{field} à t = {t:g} s")
        plt.tight_layout()
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


def animate_comparison(
    openfoam_data,
    sinn_data,
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
    sinn_data : str ou dict
        Même structure pour les prédictions SINN.
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

    of = _load_animation_data(openfoam_data)
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
