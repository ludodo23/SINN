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


def nondimensionalize_points(points_si: torch.Tensor, cfg) -> torch.Tensor:
    """Adimensionne les coordonnées et le temps à partir des unités SI.
    points_si : (N, 3), colonnes (x, y, t)
    """
    L, T = float(cfg.length_ref), float(cfg.time_ref)
    return points_si / points_si.new_tensor([L, L, T])

def dimensionalize_points(points_star: torch.Tensor, cfg) -> torch.Tensor:
    """Dimensionne les coordonnées et le temps à partir des unités adimensionnelles.
    points_star : (N, 3), colonnes (x*, y*, t*)
    """
    L, T = float(cfg.length_ref), float(cfg.time_ref)
    return points_star * points_star.new_tensor([L, L, T])

def mixture_properties(alpha: torch.Tensor, cfg):
    """Return dimensional density [kg/m3] and dynamic viscosity [Pa.s]."""
    rho = alpha*cfg.rho_water + (1.0-alpha)*cfg.rho_air
    mu_w, mu_a = cfg.rho_water*cfg.nu_water, cfg.rho_air*cfg.nu_air
    mu = alpha*mu_w + (1.0-alpha)*mu_a
    return rho, mu

def tank_acceleration(t_si: torch.Tensor, cfg) -> torch.Tensor:
    """Prescribed horizontal tank acceleration [m/s2]
    
    t_si is in seconds.
    The tank motion is given by:
        x_tank(t) = A * sin(omega*t + phase)
    where A is the amplitude, omega is the angular frequency, and phase is the phase shift.
    """
    return -cfg.tank_amplitude*cfg.tank_omega**2*torch.sin(
        cfg.tank_omega*t_si + cfg.tank_phase)
