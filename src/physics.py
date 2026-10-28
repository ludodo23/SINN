import torch


def grad(y, x):
    """Automatic gradient of y with respect to x.

    Parameters
    ----------
    y : torch.Tensor
        Output tensor.
    x : torch.Tensor
        Input tensor with respect to which the gradient is computed.

    Returns
    -------
    torch.Tensor
        Gradient of y with respect to x, same shape as x.
    """
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def nondimensionalize_points(points_si: torch.Tensor, cfg) -> torch.Tensor:
    """Non-dimensionalise coordinates and time from SI units.

    Parameters
    ----------
    points_si : torch.Tensor
        Points in SI units, shape ``(N, 3)`` with columns ``(x, y, t)``.
    cfg : object
        Configuration object with ``length_ref`` and ``time_ref`` attributes.

    Returns
    -------
    torch.Tensor
        Non-dimensional points, shape ``(N, 3)`` with columns ``(x*, y*, t*)``.
    """
    L, T = float(cfg.length_ref), float(cfg.time_ref)
    return points_si / points_si.new_tensor([L, L, T])


def dimensionalize_points(points_star: torch.Tensor, cfg) -> torch.Tensor:
    """Dimensionalises coordinates and time from non-dimensional units.

    Parameters
    ----------
    points_star : torch.Tensor
        Non-dimensional points, shape ``(N, 3)`` with columns
        ``(x*, y*, t*)``.
    cfg : object
        Configuration object with ``length_ref`` and ``time_ref`` attributes.

    Returns
    -------
    torch.Tensor
        Points in SI units, shape ``(N, 3)`` with columns ``(x, y, t)``.
    """
    L, T = float(cfg.length_ref), float(cfg.time_ref)
    return points_star * points_star.new_tensor([L, L, T])


def mixture_properties(alpha: torch.Tensor, cfg):
    """Return dimensional density and dynamic viscosity of the mixture.

    Parameters
    ----------
    alpha : torch.Tensor
        Volume fraction of water (1 = water, 0 = air).
    cfg : object
        Configuration object with ``rho_water``, ``rho_air``, ``nu_water``,
        ``nu_air`` attributes.

    Returns
    -------
    rho : torch.Tensor
        Dimensional density of the mixture [kg/m^3].
    mu : torch.Tensor
        Dimensional dynamic viscosity of the mixture [Pa.s].
    """
    rho = alpha * cfg.rho_water + (1.0 - alpha) * cfg.rho_air
    mu_w, mu_a = cfg.rho_water * cfg.nu_water, cfg.rho_air * cfg.nu_air
    mu = alpha * mu_w + (1.0 - alpha) * mu_a
    return rho, mu


def tank_acceleration(t_si: torch.Tensor, cfg) -> torch.Tensor:
    """Prescribed horizontal tank acceleration [m/s^2].

    The tank motion is given by::

        x_tank(t) = A * sin(omega * t + phase)

    where ``A`` is the amplitude, ``omega`` is the angular frequency, and
    ``phase`` is the phase shift.

    Parameters
    ----------
    t_si : torch.Tensor
        Time in seconds.
    cfg : object
        Configuration object with ``tank_amplitude``, ``tank_omega`` and
        ``tank_phase`` attributes.

    Returns
    -------
    torch.Tensor
        Horizontal tank acceleration [m/s^2].
    """
    return -cfg.tank_amplitude * cfg.tank_omega**2 * torch.sin(
        cfg.tank_omega * t_si + cfg.tank_phase
    )
