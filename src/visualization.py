import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import numpy as np
import torch

from IPython.display import HTML, display

from model import SloshingPINN


def plot_history(path="outputs/loss_history.json", use_same_figure=True):
    """Plot training loss history.

    Parameters
    ----------
    path : str, optional
        Path to the loss history JSON file. Default is
        ``"outputs/loss_history.json"``.
    use_same_figure : bool, optional
        If True, plot all losses in a single figure. If False, create
        a separate figure for each loss. Default is True.
    """
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
            ax.set_ylabel("Loss (log scale)")

        ax.semilogy(epochs, values, label=name)
        ax.set_xlabel("Epoch")
        ax.set_title(f"Loss history: {name}")
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
    """Load a trained SloshingPINN model from a checkpoint.

    Parameters
    ----------
    checkpoint : str, optional
        Path to the ``.pt`` checkpoint file. Default is
        ``"outputs/sloshing_pinn.pt"``.
    config : object, optional
        Configuration object. If None, attempts to recover it from
        the checkpoint.
    device : str, optional
        Device for the model. Defaults to CUDA if available, else CPU.

    Returns
    -------
    model : SloshingPINN
        Loaded model in evaluation mode.
    device : torch.device
        Device the model is on.
    cfg : object
        Configuration associated with the model.
    """
    device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Build the model and load weights via the recommended PyTorch API
    # (state_dict) directly in __init__.
    model = SloshingPINN(state_path=checkpoint, config=config).to(device)

    # If no config was provided, try to recover it from the checkpoint.
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
    """Plot fields predicted by the PINN model at time t.

    Parameters
    ----------
    model : SloshingPINN
        Loaded model (required). The configuration is retrieved from
        ``model.cfg`` and predictions go through ``predict_physical``
        (SI coordinates).
    t : float, optional
        Time in seconds for display. Default is 0.0.
    nx, ny : int, optional
        Grid resolution for display. Defaults are 180 and 80.
    fields : tuple of str, optional
        Names of fields to display (among "u", "v", "p", "alpha").
        Default is ``("u", "v", "p", "alpha")``.
    openfoam_data : str or dict or None, optional
        If provided, for each field displays three panels side by side:
        OpenFOAM, PINN model, and relative error (scale 0 to 1).
    """
    cfg = model.cfg
    if cfg is None:
        raise ValueError(
            "The model has no configuration (model.cfg is None)."
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

    # Load OpenFOAM data for comparison if provided, so that the
    # display grid (nx, ny) can be calibrated on the OpenFOAM "X".
    of = None
    X_of = None
    t_of = None
    i_t = None
    if openfoam_data is not None:
        of = _load_animation_data(openfoam_data)
        if "X" not in of or "time" not in of:
            raise ValueError(
                "openfoam_data must contain keys 'X' and 'time'."
            )
        X_of = np.asarray(of["X"])
        t_of = np.asarray(of["time"]).reshape(-1)
        # Find the OpenFOAM time closest to t.
        i_t = int(np.argmin(np.abs(t_of - t)))

    if of is not None:
        # Calibrate nx, ny on the OpenFOAM X (unique sorted coordinates).
        xs = np.unique(np.round(X_of[0, :, 0], 9))
        ys = np.unique(np.round(X_of[0, :, 1], 9))
        x = xs
        y = ys
        nx = int(len(x))
        ny = int(len(y))
    else:
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

    if isinstance(fields, str):
        fields = (fields,)

    for field in fields:
        Z = prediction[field].cpu().numpy().reshape(ny, nx)

        if of is None:
            plt.figure()
            plt.contourf(X, Y, Z, levels=40)
            plt.colorbar(label=field)
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.title(f"{field} at t = {t:g} s")
            plt.tight_layout()
            plt.show()
        else:
            # Three panels: OpenFOAM, model, relative error.
            Z_of = np.asarray(of[field])[i_t].reshape(ny, nx, order="F")
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
            axes[1].set_title("SINN (model)")
            fig.colorbar(im1, ax=axes[1], label=field, fraction=0.046)

            im2 = axes[2].contourf(X, Y, err, levels=40, vmin=0, vmax=1)
            axes[2].set_title("Relative error")
            fig.colorbar(im2, ax=axes[2], label="|m−o|/max(|o|,eps)", fraction=0.046)

            for ax in axes:
                ax.set_xlabel("x [m]")
                ax.set_ylabel("y [m]")
                ax.set_aspect("equal", adjustable="box")
                ax.grid(alpha=0.15)

            fig.suptitle(f"{field} at t = {t:g} s")
            fig.tight_layout()
            plt.show()


def _load_animation_data(data):
    """Load a dataset from an .npz path or a dictionary.

    Parameters
    ----------
    data : str, bytes, dict, or NpzFile
        Path to an .npz file, a dictionary, or an NpzFile object.

    Returns
    -------
    dict
        Dictionary mapping array names to array values.
    """
    if isinstance(data, (str, bytes)):
        with np.load(data) as file:
            return {key: file[key] for key in file.files}

    if isinstance(data, dict):
        return data

    # Handle Path objects by converting to string (must be before str check)
    if isinstance(data, Path):
        data = str(data)

    if isinstance(data, (str, bytes)):
        with np.load(data) as file:
            return {key: file[key] for key in file.files}

    if isinstance(data, dict):
        return data

    # Allow also passing an NpzFile object directly
    return {key: data[key] for key in data.files}


def build_sinn_data_from_model(
    model: SloshingPINN,
    openfoam_data,
    batch_size: int = 65536,
) -> dict:
    """Evaluate the PINN model at the same (x, y, t) as the OpenFOAM dataset.

    Coordinates are in SI units; the call goes through
    ``model.predict_``. Returns a dictionary usable directly
    directement par animate_comparison.

    Parameters
    ----------
    model : SloshingPINN
        Loaded model (must have ``model`` configuration).
    openfoam_data : str or dict
        Path to an .npz file or dictionary containing ``X`` and
        ``time``.
    batch_size : int, optional
        Batch size for inference. Default is 65536.

    Returns
    -------
    dict
        Dictionary with keys ``"time"``, ``"X"``, ``"alpha"``.
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
        "alpha": predictions["alpha"],
    }


def animate_comparison(
    openfoam_data,
    sinn_data=None,
    model=None,
    save_path=None,
    field="alpha",
    xlim=(-2, 8),
    ylim=None,
    interval=30,
    point_size=10,
    show_forces=True,
    display_animation=True,
):
    """
    Compare OpenFOAM and SINN fields side by side.

    Parameters
    ----------
    openfoam_data : str or dict
        Path to an .npz file or dictionary containing X, alpha,
        time and optionally Fx.
    sinn_data : str or dict, optional
        Same structure for SINN predictions. If None, a model must
        be provided via model.
    model : SloshingPINN, optional
        Loaded model. If sinn_data is None, SINN predictions are
        generated on the fly from this model (via
        build_sinn_data_from_model).
    save_path : str or None, optional
        Output path for .gif or .mp4.
    field : str, optional
        Name of the scalar field to compare, default 'alpha'.
    xlim, ylim : tuple, optional
        Common spatial limits for both panels.
    interval : int, optional
        Duration between frames in milliseconds.
    point_size : float, optional
        Size of scatter points.
    show_forces : bool, optional
        Display Fx curves if both datasets contain them.
    display_animation : bool, optional
        Display the animation in a Jupyter notebook.

    Returns
    -------
    animation : matplotlib.animation.FuncAnimation
    fig : matplotlib.figure.Figure
    """

    if sinn_data is None and model is None:
        raise ValueError(
            "Provide either sinn_data or model (at least one of them)."
        )
    if sinn_data is not None and model is not None:
        raise ValueError(
            "Provide either sinn_data or model, but not both."
        )

    of = _load_animation_data(openfoam_data)

    if sinn_data is None:
        sinn = build_sinn_data_from_model(model, openfoam_data)
    else:
        sinn = _load_animation_data(sinn_data)

    # --------------------------------------------------------
    # Data loading
    # --------------------------------------------------------
    for name, data in [("OpenFOAM", of), ("SINN", sinn)]:
        if "X" not in data or "time" not in data:
            raise ValueError(
                f"{name} must contain keys 'X' and 'time'."
            )

        if field not in data:
            raise ValueError(
                f"Field '{field}' missing from dataset {name}."
            )

    X_of = np.asarray(of["X"])
    X_sinn = np.asarray(sinn["X"])

    A_of = np.asarray(of[field])
    A_sinn = np.asarray(sinn[field])

    t_of = np.asarray(of["time"]).reshape(-1)
    t_sinn = np.asarray(sinn["time"]).reshape(-1)

    if X_of.ndim != 3 or X_of.shape[-1] < 2:
        raise ValueError("OpenFOAM: X must have shape (T, N, 2).")

    if X_sinn.ndim != 3 or X_sinn.shape[-1] < 2:
        raise ValueError("SINN: X must have shape (T, N, 2).")

    if A_of.shape[:2] != X_of.shape[:2]:
        raise ValueError("OpenFOAM: X and field dimensions are incompatible.")

    if A_sinn.shape[:2] != X_sinn.shape[:2]:
        raise ValueError("SINN: X and field dimensions are incompatible.")

    if len(t_of) != len(X_of) or len(t_sinn) != len(X_sinn):
        raise ValueError("Time vectors must match X.")

    if len(t_of) == 0 or len(t_sinn) == 0:
        raise ValueError("Datasets must not be empty.")

    if not np.all(np.isfinite(t_of)) or not np.all(np.isfinite(t_sinn)):
        raise ValueError("Time vectors contain non-finite values.")

    # Keep OpenFOAM times within the SINN interval.
    valid = (t_of >= t_sinn.min()) & (t_of <= t_sinn.max())
    frame_indices = np.flatnonzero(valid)

    if len(frame_indices) == 0:
        raise ValueError(
            "OpenFOAM and SINN time intervals do not overlap."
        )

    times = t_of[frame_indices]

    # Associate each OpenFOAM time with the closest SINN instant.
    sinn_indices = np.abs(
        times[:, None] - t_sinn[None, :]
    ).argmin(axis=1)

    # --------------------------------------------------------
    # Figure: two fields on top, forces at bottom
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
    # Scalar fields: same colour scale
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

    ax_of.set_title("interFoam")
    ax_sinn.set_title("PINN")

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

    title = fig.suptitle(f"Comparison at t = {times[0]:.4f} s")

    # --------------------------------------------------------
    # Forces Fx
    # --------------------------------------------------------
    if has_forces:
        Fx_of = np.asarray(of["Fx"]).reshape(-1)
        Fx_sinn = np.asarray(sinn["Fx"]).reshape(-1)

        if len(Fx_of) != len(t_of) or len(Fx_sinn) != len(t_sinn):
            raise ValueError("Each Fx series must match its time vector length.")

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
    # Animation update
    # --------------------------------------------------------
    def update(frame):
        i = frame_indices[frame]
        j = sinn_indices[frame]

        scatter_of.set_offsets(X_of[i, :, :2])
        scatter_of.set_array(A_of[i].reshape(-1))

        scatter_sinn.set_offsets(X_sinn[j, :, :2])
        scatter_sinn.set_array(A_sinn[j].reshape(-1))

        title.set_text(
            f"Comparison — t = {t_of[i]:.4f} s "
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
    # Optional export
    # --------------------------------------------------------
    if save_path is not None:
        path = str(save_path)
        if path.lower().endswith(".gif"):
            animation.save(path, writer=PillowWriter(fps=1000 / interval))
        elif path.lower().endswith(".mp4"):
            animation.save(path, writer=FFMpegWriter(fps=1000 / interval))
        else:
            raise ValueError("Output file must be .gif or .mp4")

    if display_animation:
        plt.close(fig)
        display(HTML(animation.to_jshtml()))

    return animation, fig
