import torch


def sample_interior(n, cfg, device=None):
    device = device or cfg.device

    x = cfg.x_min + (cfg.x_max - cfg.x_min) * torch.rand(
        n, 1, device=device
    )
    y = cfg.y_min + (cfg.y_max - cfg.y_min) * torch.rand(
        n, 1, device=device
    )
    t = cfg.t_min + (cfg.t_max - cfg.t_min) * torch.rand(
        n, 1, device=device
    )

    return torch.cat([x, y, t], dim=1)


def sample_initial(n, cfg, device=None):
    device = device or cfg.device

    x = cfg.x_min + (cfg.x_max - cfg.x_min) * torch.rand(
        n, 1, device=device
    )
    y = cfg.y_min + (cfg.y_max - cfg.y_min) * torch.rand(
        n, 1, device=device
    )
    t = torch.full_like(x, cfg.t_min)

    return torch.cat([x, y, t], dim=1)


def sample_walls(n, cfg, device=None):
    """
    Échantillonne les quatre côtés du rectangle.

    IMPORTANT : si le bord supérieur est une ouverture, il faut
    retirer ce bord de l'échantillonnage des parois.
    """
    device = device or cfg.device
    m = max(1, n // 4)

    t = cfg.t_min + (cfg.t_max - cfg.t_min) * torch.rand(
        m, 1, device=device
    )
    x = cfg.x_min + (cfg.x_max - cfg.x_min) * torch.rand(
        m, 1, device=device
    )
    y = cfg.y_min + (cfg.y_max - cfg.y_min) * torch.rand(
        m, 1, device=device
    )

    left = torch.cat([
        torch.full_like(y, cfg.x_min), y, t
    ], dim=1)

    right = torch.cat([
        torch.full_like(y, cfg.x_max), y, t
    ], dim=1)

    bottom = torch.cat([
        x, torch.full_like(x, cfg.y_min), t
    ], dim=1)

    top = torch.cat([
        x, torch.full_like(x, cfg.y_max), t
    ], dim=1)

    return torch.cat([left, right, bottom, top], dim=0)