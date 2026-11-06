import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import numpy as np
import torch

from IPython.display import HTML, display

from model import SloshingPINN


def plot_history(
        path="outputs/loss_history.json",
        use_same_figure=True,
        loss_names=[
            "continuity",
            "vof",
            "momentum_x",
            "momentum_y",
            "initial_u",
            "initial_v",
            "initial_alpha",
            "wall_u",
            "wall_v",
            "pressure_gauge"],
        title=None,
        ax=None,
        label=None,
        legend=True):
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

    if ax is None:
        if use_same_figure:
            fig_, ax_ = plt.subplots()
            fig, ax = fig_, ax_
        else:
            fig, ax = [], []
    else:
        ax_ = ax
        use_same_figure = True
        fig = plt.gcf()
        fig_ = fig

    if title is None:
        title = "Loss history"

    for name in loss_names:
        values = [
            max(row[name], 1e-16)
            for row in history
        ]

        if not use_same_figure:
            fig_, ax_ = plt.subplots()
            ax_.set_title(f"{title}: {name}")

        ax_.semilogy(epochs, values, label=label if label is not None else name)
        ax_.set_xlabel("Epoch")
        ax_.set_ylabel("Loss (log scale)")
        ax_.grid(True)
        fig_.tight_layout()
        if not use_same_figure:
            fig.append(fig_)
            ax.append(ax_)
    if use_same_figure:
        if legend:
            fig_.legend(loc=7)
            fig.tight_layout()
            fig.subplots_adjust(right=0.75)
        ax_.set_title(title)

    return fig, ax


def plot_histories(paths=[], names=None):
    """_summary_

    Args:
        paths (list, optional): _description_. Defaults to [].
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    if names is None:
        names = [p.parent.name for p in paths]
    for p, n in zip(paths, names):
        plot_history(
            path=p,
            loss_names=["total"],
            label=n,
            ax=ax,
            legend=False
        )
    fig.legend(loc=7)
    fig.subplots_adjust(right=0.75)
    fig.tight_layout()
    ax.set_title("Loss comparison")
    return fig, ax


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
    return model


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
            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.contourf(X, Y, Z, levels=40)
            cbar = fig.colorbar(im, ax=ax, label=field)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title(f"{field} at t = {t:.4f} s")
            ax.set_aspect("equal", adjustable="box")
            return fig
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
            return fig


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
        "u": predictions["u"],
        "v": predictions["v"],
        "p": predictions["p"],
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
    t_max=None
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
    t_max : float or None, optional
        Maximum time to display. If None, uses the maximum time
        from the OpenFOAM dataset.

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
    valid = (t_of >= t_sinn.min()) & (t_of <= t_sinn.max()) & (t_of <= t_max if t_max is not None else True)
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


import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm


# =====================================================================
# Shared domain resolution
# =====================================================================
def _resolve_domain(model, x, y, nx, ny):
    """Build / auto-detect the (x, y) domain shared by plot_* and animate_*.

    Returns a dict with keys:
        structured, Xg, Yg, x_flat, y_flat, nx, ny, ix, iy
    """
    cfg = model.cfg
    if cfg is None:
        raise ValueError("The model has no configuration (model.cfg is None).")

    # ---------- Build a mesh from config bounds ----------
    if x is None or y is None:
        try:
            x_min, x_max = float(cfg.x_min), float(cfg.x_max)
            y_min, y_max = float(cfg.y_min), float(cfg.y_max)
        except AttributeError as e:
            raise ValueError(
                "x and y are None, but cfg is missing one of "
                "x_min/x_max/y_min/y_max."
            ) from e
        ex, ey = x_max - x_min, y_max - y_min
        if ex <= 0 or ey <= 0:
            raise ValueError(f"Invalid domain bounds: ex={ex}, ey={ey}.")
        base = 256
        if nx is None and ny is None:
            if ex >= ey:
                nx, ny = base, max(64, int(round(base * ey / ex)))
            else:
                ny, nx = base, max(64, int(round(base * ex / ey)))
        elif nx is None:
            nx = max(64, int(round(ny * ex / ey)))
        elif ny is None:
            ny = max(64, int(round(nx * ey / ex)))
        xg = np.linspace(x_min, x_max, nx, dtype=np.float32)
        yg = np.linspace(y_min, y_max, ny, dtype=np.float32)
        Xg, Yg = np.meshgrid(xg, yg)
        x_flat, y_flat = Xg.ravel(), Yg.ravel()
        return dict(
            structured=True, Xg=Xg, Yg=Yg,
            x_flat=x_flat, y_flat=y_flat, nx=nx, ny=ny,
            ix=np.searchsorted(xg, x_flat),
            iy=np.searchsorted(yg, y_flat),
        )

    # ---------- User-provided points ----------
    x = np.asarray(x, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of points.")
    n_pts = len(x)

    if nx is not None and ny is not None:
        if nx * ny != n_pts:
            raise ValueError(f"Points ({n_pts}) != nx*ny={nx*ny}.")
        ux, uy = np.unique(x), np.unique(y)
        if len(ux) != nx or len(uy) != ny:
            raise ValueError(
                f"Unique x ({len(ux)}) or y ({len(uy)}) does not match nx/ny."
            )
        Xg, Yg = np.meshgrid(ux, uy)
        return dict(
            structured=True, Xg=Xg, Yg=Yg,
            x_flat=x, y_flat=y, nx=nx, ny=ny,
            ix=np.searchsorted(ux, x),
            iy=np.searchsorted(uy, y),
        )

    ux_c, uy_c = np.unique(x), np.unique(y)
    if len(ux_c) * len(uy_c) == n_pts:
        ix_c = np.searchsorted(ux_c, x)
        iy_c = np.searchsorted(uy_c, y)
        if len(np.unique(ix_c * len(uy_c) + iy_c)) == n_pts:
            nx, ny = len(ux_c), len(uy_c)
            Xg, Yg = np.meshgrid(ux_c, uy_c)
            return dict(
                structured=True, Xg=Xg, Yg=Yg,
                x_flat=x, y_flat=y, nx=nx, ny=ny, ix=ix_c, iy=iy_c,
            )

    return dict(
        structured=False, Xg=None, Yg=None,
        x_flat=x, y_flat=y, nx=None, ny=None, ix=None, iy=None,
    )


def _forward_vof(model, coords):
    """Best-effort extraction of the VOF field from ``model(coords)``."""
    try:
        out = model(coords)
        if isinstance(out, dict):
            return next(
                (out[key] for key in ("alpha", "vof", "a") if key in out),
                None,
            )
        if isinstance(out, (tuple, list)) and len(out) > 0:
            return out[-1]
        return out
    except Exception:
        return None


# =====================================================================
# Static plot
# =====================================================================
@torch.no_grad()
def plot_residuals(
    model,
    x=None,
    y=None,
    t: float = 0.0,
    fields=("continuity", "vof", "momentum_x", "momentum_y"),
    nx=None,
    ny=None,
    cmap="viridis",
    abs_value=True,
    log=False,
    overlay_vof=True,
    figsize=None,
):
    """Plot physical residuals of a PINN as 2D contour maps in (x, y).

    Domain handling
    ---------------
    * If ``x`` and ``y`` are ``None``, a regular Cartesian mesh is built
      from ``model.cfg.x_min/x_max/y_min/y_max`` (resolution ``nx × ny``,
      auto-chosen to respect the aspect ratio unless given).
    * Otherwise the provided points are used, and the plotting mode is
      auto-detected: ``contourf`` on a Cartesian grid, ``tricontourf``
      on an unstructured cloud. No 1D / scatter fallback.

    Parameters
    ----------
    model : SloshingPINN
    x, y : np.ndarray, optional
        Domain points in SI units. If both ``None``, built from cfg bounds.
    t : float
        Time in seconds.
    fields : tuple of str
        Subset of ``("continuity", "vof", "momentum_x", "momentum_y")``.
    nx, ny : int, optional
        Grid resolution (see module docstring).
    cmap : str
        Colormap. Default ``"viridis"``.
    abs_value : bool
        Plot ``|residual|``. Default ``True``.
    log : bool
        Log-scale the magnitude (implies ``|residual|``).
    overlay_vof : bool
        Overlay the ``α = 0.5`` iso-contour of the predicted VOF field.
    figsize : tuple, optional
        Default ``(8, 4.5 * n_fields)``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    cfg = model.cfg
    if cfg is None:
        raise ValueError("The model has no configuration (model.cfg is None).")

    if isinstance(fields, str):
        fields = (fields,)
    valid = {"continuity", "vof", "momentum_x", "momentum_y"}
    for f in fields:
        if f not in valid:
            raise ValueError(f"Invalid field '{f}'. Must be one of {valid}.")

    device = next(model.parameters()).device
    t_val = float(t)

    # ---------- Domain ----------
    dom = _resolve_domain(model, x, y, nx, ny)
    structured = dom["structured"]
    Xg, Yg = dom["Xg"], dom["Yg"]
    x_flat, y_flat = dom["x_flat"], dom["y_flat"]
    NX, NY = dom["nx"], dom["ny"]
    ix, iy = dom["ix"], dom["iy"]

    def to_grid(a):
        G = np.full((NY, NX), np.nan, dtype=np.float64)
        G[iy, ix] = a
        return G

    # ---------- Compute residuals ----------
    coords = torch.tensor(
        np.column_stack(
            [x_flat, y_flat, np.full(len(x_flat), t_val, dtype=np.float32)]
        ),
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    res = model.compute_residuals(coords)
    residual_dict = {
        k: v.detach().cpu().numpy().ravel() for k, v in res.items()
    }

    # ---------- Optional VOF overlay ----------
    vof_field = None
    if overlay_vof:
        v = _forward_vof(model, coords)
        if v is not None:
            vof_field = v.detach().cpu().numpy().ravel()

    # ---------- Figure ----------
    n_fields = len(fields)
    if figsize is None:
        figsize = (8, 4.5 * n_fields)
    fig, axes = plt.subplots(n_fields, 1, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for idx, field in enumerate(fields):
        ax = axes[idx]
        Z = residual_dict[field].astype(np.float64)
        if abs_value or log:
            Z = np.abs(Z)

        # ---- Robust colour scale ----
        if log:
            vmax = float(np.nanpercentile(Z, 99.5))
            vmin = max(float(np.nanpercentile(Z, 1)), vmax * 1e-8, 1e-12)
            if vmax <= 0:
                vmax = 1.0
            scale_kwargs = dict(norm=LogNorm(vmin=vmin, vmax=vmax))
        else:
            p = float(np.nanpercentile(np.abs(Z), 99))
            if not np.isfinite(p) or p == 0.0:
                p = 1.0
            scale_kwargs = (
                dict(vmin=0.0, vmax=p) if abs_value else dict(vmin=-p, vmax=p)
            )

        # ---- Draw: always a contour ----
        if structured:
            Zg = to_grid(Z)
            im = ax.contourf(
                Xg, Yg, Zg, levels=40, cmap=cmap,
                extend="both", **scale_kwargs,
            )
            if vof_field is not None:
                Ag = to_grid(vof_field)
                ax.contour(
                    Xg, Yg, Ag, levels=[0.5],
                    colors="k", linewidths=1.2, linestyles="--",
                )
        else:
            im = ax.tricontourf(
                x_flat, y_flat, Z, levels=40, cmap=cmap,
                extend="both", **scale_kwargs,
            )
            if vof_field is not None:
                ax.tricontour(
                    x_flat, y_flat, vof_field, levels=[0.5],
                    colors="k", linewidths=1.2, linestyles="--",
                )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"{field} residual at t = {t_val:.4f} s")
        ax.grid(alpha=0.15)
        fig.colorbar(
            im, ax=ax,
            label=f"|{field}|" if (abs_value or log) else f"{field}",
        )

    fig.suptitle(f"Physical Residuals at t = {t_val:.4f} s", y=1.01)
    fig.tight_layout()
    return fig


# =====================================================================
# Animation
# =====================================================================
@torch.no_grad()
def animate_residuals(
    model,
    times,
    x=None,
    y=None,
    fields=("continuity", "vof", "momentum_x", "momentum_y"),
    nx=None,
    ny=None,
    cmap="viridis",
    abs_value=True,
    log=False,
    overlay_vof=True,
    fps=10.0,
    interval=None,
    save_path=None,
    dpi=120,
    figsize=None,
    repeat=True,
    display_animation=True,
):
    """Animate PINN residual maps over a sequence of times.

    Same domain logic and same colour defaults as :func:`plot_residuals`.
    Residuals at all time steps are pre-computed once and the colour scale
    is fixed across frames (``p99`` or ``LogNorm`` over all frames), so
    that spatial structure can be compared frame-to-frame.

    Parameters
    ----------
    model : SloshingPINN
    times : array-like of float
        Sequence of time values (seconds).
    x, y : np.ndarray, optional
        Domain points. If both ``None``, built from cfg bounds.
    fields, nx, ny, cmap, abs_value, log, overlay_vof
        See :func:`plot_residuals`.
    fps : float
        Frames per second.
    interval : float, optional
        Delay between frames (ms). Overrides ``fps`` if given.
    save_path : str, optional
        ``.gif`` → PillowWriter, otherwise FFMpegWriter.
    dpi : int
        DPI used when saving.
    figsize : tuple, optional
        Default ``(8, 4.5 * n_fields)``.
    repeat : bool
        Loop the animation in interactive display.
    display_animation : bool
        Whether to display the animation in the notebook.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    fig : matplotlib.figure.Figure
    """
    cfg = model.cfg
    if cfg is None:
        raise ValueError("The model has no configuration (model.cfg is None).")

    if isinstance(fields, str):
        fields = (fields,)
    valid = {"continuity", "vof", "momentum_x", "momentum_y"}
    for f in fields:
        if f not in valid:
            raise ValueError(f"Invalid field '{f}'. Must be one of {valid}.")

    times = np.asarray(times, dtype=np.float64).ravel()
    if len(times) == 0:
        raise ValueError("times must contain at least one value.")
    n_t = len(times)

    device = next(model.parameters()).device

    # ---------- Domain ----------
    dom = _resolve_domain(model, x, y, nx, ny)
    structured = dom["structured"]
    Xg, Yg = dom["Xg"], dom["Yg"]
    x_flat, y_flat = dom["x_flat"], dom["y_flat"]
    NX, NY = dom["nx"], dom["ny"]
    ix, iy = dom["ix"], dom["iy"]
    n_pts = len(x_flat)

    def to_grid(a):
        G = np.full((NY, NX), np.nan, dtype=np.float64)
        G[iy, ix] = a
        return G

    # ---------- Precompute residuals + VOF overlay ----------
    residuals_flat = {f: np.empty((n_t, n_pts), dtype=np.float32) for f in fields}
    vofs = None
    vof_ok = overlay_vof
    if overlay_vof:
        vofs_tmp = np.empty((n_t, n_pts), dtype=np.float32)

    model.eval()
    for k, tk in enumerate(times):
        coords = torch.tensor(
            np.column_stack(
                [x_flat, y_flat, np.full(n_pts, float(tk), dtype=np.float32)]
            ),
            dtype=torch.float32,
            device=device,
        )
        res = model.compute_residuals(coords)
        for f in fields:
            residuals_flat[f][k] = res[f].detach().cpu().numpy().ravel()

        if overlay_vof and vof_ok:
            v = _forward_vof(model, coords)
            if v is None:
                vof_ok = False
            else:
                vofs_tmp[k] = v.detach().cpu().numpy().ravel()
    if overlay_vof and vof_ok:
        vofs = vofs_tmp

    # ---------- Global, fixed colour scales ----------
    scale_kwargs = {}
    for f in fields:
        Z = residuals_flat[f].astype(np.float64)
        if abs_value or log:
            Z = np.abs(Z)
        if log:
            vmax = float(np.nanpercentile(Z, 99.5))
            vmin = max(float(np.nanpercentile(Z, 1)), vmax * 1e-8, 1e-12)
            if vmax <= 0:
                vmax = 1.0
            scale_kwargs[f] = dict(norm=LogNorm(vmin=vmin, vmax=vmax))
        else:
            p = float(np.nanpercentile(np.abs(Z), 99))
            if not np.isfinite(p) or p == 0.0:
                p = 1.0
            scale_kwargs[f] = (
                dict(vmin=0.0, vmax=p) if abs_value else dict(vmin=-p, vmax=p)
            )

    # ---------- Axis limits ----------
    if structured:
        xlim = (float(Xg.min()), float(Xg.max()))
        ylim = (float(Yg.min()), float(Yg.max()))
    else:
        xlim = (float(x_flat.min()), float(x_flat.max()))
        ylim = (float(y_flat.min()), float(y_flat.max()))

    # ---------- Figure ----------
    n_fields = len(fields)
    if figsize is None:
        figsize = (8, 4.5 * n_fields)
    fig, axes = plt.subplots(n_fields, 1, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    def render(k, create_cbar):
        for idx, f in enumerate(fields):
            ax = axes[idx]
            ax.cla()
            Z = residuals_flat[f][k].astype(np.float64)
            if abs_value or log:
                Z = np.abs(Z)

            if structured:
                Zg = to_grid(Z)
                im = ax.contourf(
                    Xg, Yg, Zg, levels=40, cmap=cmap,
                    extend="both", **scale_kwargs[f],
                )
                if vofs is not None:
                    Ag = to_grid(vofs[k])
                    ax.contour(
                        Xg, Yg, Ag, levels=[0.5],
                        colors="k", linewidths=1.2, linestyles="--",
                    )
            else:
                im = ax.tricontourf(
                    x_flat, y_flat, Z, levels=40, cmap=cmap,
                    extend="both", **scale_kwargs[f],
                )
                if vofs is not None:
                    ax.tricontour(
                        x_flat, y_flat, vofs[k], levels=[0.5],
                        colors="k", linewidths=1.2, linestyles="--",
                    )

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title(f"{f} residual at t = {float(times[k]):.4f} s")
            ax.grid(alpha=0.15)

            if create_cbar:
                fig.colorbar(
                    im, ax=ax,
                    label=f"|{f}|" if (abs_value or log) else f"{f}",
                )

        fig.suptitle(
            f"Physical Residuals  (t = {float(times[k]):.4f} s)", y=1.01
        )
        return []

    # First frame + colorbars + frozen layout
    render(0, create_cbar=True)
    fig.tight_layout()
    positions = [ax.get_position() for ax in axes]

    def update(k):
        render(k, create_cbar=False)
        for i, p in enumerate(positions):
            axes[i].set_position(p)
        return []

    if interval is None:
        interval = 1000.0 / max(float(fps), 1e-6)

    anim = FuncAnimation(
        fig, update, frames=n_t, interval=interval, blit=False, repeat=repeat
    )

    if save_path is not None:
        sp = str(save_path)
        if sp.endswith(".gif"):
            from matplotlib.animation import PillowWriter
            anim.save(sp, writer=PillowWriter(fps=fps), dpi=dpi)
        else:
            anim.save(sp, fps=fps, dpi=dpi)
        print(f"Saved animation to {sp}")

    if display_animation:
        plt.close(fig)
        display(HTML(anim.to_jshtml()))

    return anim, fig
