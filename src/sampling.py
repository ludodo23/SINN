
import torch


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
    """Normalise un score par ses quantiles 10 % et 90 %."""
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
    """Score de difficulté par candidat, sans gradient d'entraînement."""
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


def sample_mixed_adaptive(candidates, scores, n, adaptive_fraction=0.5):
    """Sélectionne des points uniformes et difficiles sans remplacement."""
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
    return candidates[indices]
