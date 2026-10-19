
import torch


def sample_interior(n, cfg, device=None):
    """Échantillonne le domaine intérieur en unités SI."""
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
    """Échantillonne le domaine à l'instant initial, en SI."""
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
    """
    Échantillonne les parois du rectangle en unités SI.

    include_top=False si le bord supérieur est une ouverture.
    Les points sont répartis entre les parois échantillonnées.
    """
    device = torch.device(device or cfg.device)

    # Trois côtés si le sommet est ouvert, sinon quatre.
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
