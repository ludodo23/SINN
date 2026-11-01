
import torch

"""Sampling utilities for the Sloshing PINN.

Provides functions for generating collocation points, initial points,
wall points, and adaptive sampling based on physical residuals.
"""

def sample_interior(n, cfg, device=None):
    """Sample the interior domain in SI units.

    Parameters
    ----------
    n : int
        Number of points to sample.
    cfg : object
        Configuration with ``x_min``, ``x_max``, ``y_min``, ``y_max``,
        ``t_min``, ``t_max``, and ``device`` attributes.
    device : str or torch.device, optional
        Device for the generated tensor. Defaults to ``cfg.device``.

    Returns
    -------
    torch.Tensor
        Sampled points, shape ``(n, 3)`` with columns ``(x, y, t)``
        in SI units.
    """
    device = torch.device(device or cfg.device)

    x = cfg.x_min + (cfg.x_max - cfg.x_min) * torch.rand(
        n, 1, device=device
    )
    y = cfg.y_min + (cfg.y_max - cfg.y_min) * torch.rand(
        n, 1, device=device
    )
    t = cfg.t_min + (cfg.t_max - cfg.t_min) * torch.rand(
        n, 1, device=device
    )

    return torch.cat((x, y, t), dim=1)


def sample_initial(n, cfg, device=None):
    """Sample the domain at the initial time, in SI units.

    Parameters
    ----------
    n : int
        Number of points to sample.
    cfg : object
        Configuration with ``x_min``, ``x_max``, ``y_min``, ``y_max``,
        ``t_min``, and ``device`` attributes.
    device : str or torch.device, optional
        Device for the generated tensor. Defaults to ``cfg.device``.

    Returns
    -------
    torch.Tensor
        Sampled points, shape ``(n, 3)`` with columns ``(x, y, t)``
        in SI units, all at ``t = t_min``.
    """
    device = torch.device(device or cfg.device)

    x = cfg.x_min + (cfg.x_max - cfg.x_min) * torch.rand(
        n, 1, device=device
    )
    y = cfg.y_min + (cfg.y_max - cfg.y_min) * torch.rand(
        n, 1, device=device
    )
    t = torch.full_like(x, cfg.t_min)

    return torch.cat((x, y, t), dim=1)


def sample_walls(n, cfg, device=None, include_top=True):
    """Sample the rectangle walls in SI units.

    Parameters
    ----------
    n : int
        Total number of wall points to sample.
    cfg : object
        Configuration with ``x_min``, ``x_max``, ``y_min``, ``y_max``,
        ``t_min``, ``t_max``, and ``device`` attributes.
    device : str or torch.device, optional
        Device for the generated tensor. Defaults to ``cfg.device``.
    include_top : bool, optional
        If False, the top edge is omitted (useful when the top is an
        opening). Default is True.

    Returns
    -------
    torch.Tensor
        Sampled wall points, shape ``(n, 3)`` with columns ``(x, y, t)``
        in SI units. Points are distributed among the sampled walls.
    """
    device = torch.device(device or cfg.device)

    # Three sides if top is open, otherwise four.
    sides = ["left", "right", "bottom"]
    if include_top:
        sides.append("top")

    counts = [n // len(sides)] * len(sides)
    for i in range(n % len(sides)):
        counts[i] += 1

    points = []

    for side, m in zip(sides, counts):
        if m == 0:
            continue

        t = cfg.t_min + (cfg.t_max - cfg.t_min) * torch.rand(
            m, 1, device=device
        )

        if side in ("left", "right"):
            y = cfg.y_min + (cfg.y_max - cfg.y_min) * torch.rand(
                m, 1, device=device
            )
            x_value = cfg.x_min if side == "left" else cfg.x_max
            x = torch.full_like(y, x_value)
            points.append(torch.cat((x, y, t), dim=1))

        else:
            x = cfg.x_min + (cfg.x_max - cfg.x_min) * torch.rand(
                m, 1, device=device
            )
            y_value = cfg.y_min if side == "bottom" else cfg.y_max
            y = torch.full_like(x, y_value)
            points.append(torch.cat((x, y, t), dim=1))

    if not points:
        return torch.empty((0, 3), device=device)

    return torch.cat(points, dim=0)


def robust_normalize(values, eps=1e-8):
    """Normalise a score by its 10% and 90% quantiles.

    Parameters
    ----------
    values : torch.Tensor
        Input values to normalise, will be reshaped to 1D.
    eps : float, optional
        Small value added to avoid division by zero. Default is 1e-8.

    Returns
    -------
    torch.Tensor
        Normalised values in [0, 1].
    """
    values = values.detach().reshape(-1)
    q10 = torch.quantile(values, 0.10)
    q90 = torch.quantile(values, 0.90)
    return ((values - q10) / (q90 - q10 + eps)).clamp(0.0, 1.0)


def adaptive_score(
    continuity_residual,
    vof_residual,
    momentum_residual,
    alpha,
    grad_alpha_norm,
    w_continuity=0.2,
    w_vof=0.4,
    w_momentum=0.2,
    w_interface=0.2,
):
    """Score of difficulty per candidate, without training gradient.

    Parameters
    ----------
    continuity_residual : torch.Tensor
        Continuity residual per candidate point.
    vof_residual : torch.Tensor
        Volume fraction residual per candidate point.
    momentum_residual : torch.Tensor
        Momentum residual per candidate point.
    alpha : torch.Tensor
        Volume fraction per candidate point.
    grad_alpha_norm : torch.Tensor
        Norm of the gradient of alpha per candidate point.
    w_continuity : float, optional
        Weight for continuity residual. Default is 0.2.
    w_vof : float, optional
        Weight for volume fraction residual. Default is 0.4.
    w_momentum : float, optional
        Weight for momentum residual. Default is 0.2.
    w_interface : float, optional
        Weight for interface proximity. Default is 0.2.

    Returns
    -------
    torch.Tensor
        Difficulty score per candidate point.
    """
    rc = robust_normalize(continuity_residual.square())
    rv = robust_normalize(vof_residual.square())
    rm = robust_normalize(momentum_residual)
    ga = robust_normalize(grad_alpha_norm)

    proximity = torch.exp(
        -((alpha.detach().reshape(-1) - 0.5) / 0.15).square()
    )

    interface_score = 0.5 * proximity + 0.5 * ga

    return (
        w_continuity * rc
        + w_vof * rv
        + w_momentum * rm
        + w_interface * interface_score
    )


def sample_mixed_adaptive(candidates, scores, n, adaptive_fraction=0.5,
                          return_indices=False):
    """Select uniform and difficult candidates without replacement.

    Parameters
    ----------
    candidates : torch.Tensor
        Pool of candidate points to select from.
    scores : torch.Tensor
        Difficulty scores corresponding to each candidate.
    n : int
        Number of points to select.
    adaptive_fraction : float, optional
        Fraction of candidates to select based on difficulty.
        The remaining ``1 - adaptive_fraction`` will be selected uniformly.
        Default is 0.5.
    return_indices : bool, optional
        If True, also returns the indices selected from ``candidates``.
        Useful for retrieving the associated scores and residuals.
        Default is False.

    Returns
    -------
    selected : torch.Tensor
        Selected candidate points.
    indices : torch.Tensor, optional
        Indices of the selected candidates in the original pool.
        Only returned if ``return_indices`` is True.
    """
    if n > candidates.shape[0]:
        raise ValueError("Le pool candidat doit contenir au moins n points.")

    n_adaptive = min(n, round(n * adaptive_fraction))
    n_uniform = n - n_adaptive

    uniform_idx = torch.randperm(
        candidates.shape[0], device=candidates.device
    )[:n_uniform]

    mask = torch.ones(
        candidates.shape[0], dtype=torch.bool, device=candidates.device
    )
    mask[uniform_idx] = False
    remaining = torch.where(mask)[0]

    k = min(n_adaptive, remaining.numel())
    if k > 0:
        local_scores = scores[remaining]
        adaptive_idx = remaining[torch.topk(local_scores, k=k).indices]
    else:
        adaptive_idx = remaining[:0]

    indices = torch.cat([uniform_idx, adaptive_idx])
    selected = candidates[indices]
    if return_indices:
        return selected, indices
    return selected
