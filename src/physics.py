import torch


def grad(y, x):
    """Gradient automatique de y par rapport à x."""
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def mixture_properties(alpha, cfg):
    """Densité et viscosité dynamique du mélange."""
    rho = (
        alpha * cfg.rho_water
        + (1.0 - alpha) * cfg.rho_air
    )

    mu = (
        alpha * cfg.mu_water
        + (1.0 - alpha) * cfg.mu_air
    )

    return rho, mu


def tank_acceleration(t, cfg):
    """
    Accélération du réservoir si :
        x_tank(t) = A * sin(omega*t + phase)
    """
    return (
        -cfg.tank_amplitude
        * cfg.tank_omega**2
        * torch.sin(cfg.tank_omega * t + cfg.tank_phase)
    )


def effective_viscosity(fields, derivatives, cfg):
    """Viscosité moléculaire et viscosité turbulente."""
    rho, mu = mixture_properties(fields["alpha"], cfg)

    mu_t = (
        rho * cfg.C_mu * fields["k"]**2
        / fields["epsilon"]
    )

    mu_eff = mu + mu_t

    return rho, mu, mu_eff, mu_t