import torch

from physics import (
    grad,
    mixture_properties,
    tank_acceleration,
    effective_viscosity,
)


def mse(value):
    return torch.mean(value**2)


def continuity_loss(model, points, cfg):
    c = points.detach().clone().requires_grad_(True)
    f = model(c)

    du = grad(f["u"], c)
    dv = grad(f["v"], c)

    residual = du[:, 0:1] + dv[:, 1:2]
    return mse(residual)


def vof_loss(model, points, cfg):
    """
    Transport non conservatif de alpha, cohérent si div(u)=0 :
        alpha_t + u*alpha_x + v*alpha_y = 0
    """
    c = points.detach().clone().requires_grad_(True)
    f = model(c)
    da = grad(f["alpha"], c)

    residual = (
        da[:, 2:3]
        + f["u"] * da[:, 0:1]
        + f["v"] * da[:, 1:2]
    )

    return mse(residual)


def momentum_loss(model, points, cfg):
    c = points.detach().clone().requires_grad_(True)
    f = model(c)

    d = {
        name: grad(value, c)
        for name, value in f.items()
    }

    rho, mu, mu_eff, mu_t = effective_viscosity(f, d, cfg)

    u, v, p = f["u"], f["v"], f["p"]

    ux, uy, ut = (
        d["u"][:, 0:1],
        d["u"][:, 1:2],
        d["u"][:, 2:3],
    )
    vx, vy, vt = (
        d["v"][:, 0:1],
        d["v"][:, 1:2],
        d["v"][:, 2:3],
    )

    px = d["p"][:, 0:1]
    py = d["p"][:, 1:2]

    # Contrainte visqueuse effective en 2D.
    tau_xx = 2.0 * mu_eff * ux
    tau_yy = 2.0 * mu_eff * vy
    tau_xy = mu_eff * (uy + vx)

    div_tau_x = (
        grad(tau_xx, c)[:, 0:1]
        + grad(tau_xy, c)[:, 1:2]
    )

    div_tau_y = (
        grad(tau_xy, c)[:, 0:1]
        + grad(tau_yy, c)[:, 1:2]
    )

    # Repère du réservoir : accélération effective g - a_tank.
    a_tank = tank_acceleration(c[:, 2:3], cfg)
    body_x = -a_tank
    body_y = torch.zeros_like(body_x) + cfg.gravity_y

    # Force capillaire CSF : le signe dépend de la convention
    # utilisée pour la normale et la courbure.
    da = grad(f["alpha"], c)
    gx = da[:, 0:1]
    gy = da[:, 1:2]

    norm = torch.sqrt(gx**2 + gy**2 + 1e-12)
    nx = gx / norm
    ny = gy / norm

    curvature = -(
        grad(nx, c)[:, 0:1]
        + grad(ny, c)[:, 1:2]
    )

    fx = cfg.sigma * curvature * gx
    fy = cfg.sigma * curvature * gy

    adv_u = ut + u * ux + v * uy
    adv_v = vt + u * vx + v * vy

    rx = (
        rho * adv_u + px - div_tau_x
        - rho * body_x - fx
    )

    ry = (
        rho * adv_v + py - div_tau_y
        - rho * body_y - fy
    )

    # Mise à l'échelle numérique provisoire.
    scale = cfg.rho_water * max(abs(cfg.gravity_y), 1.0)

    return mse(rx / scale) + mse(ry / scale)


def turbulence_losses(model, points, cfg):
    """
    Fermeture k-epsilon simplifiée.
    Les conditions aux limites de k et epsilon doivent être ajoutées
    conformément au cas réel.
    """
    c = points.detach().clone().requires_grad_(True)
    f = model(c)

    d = {
        name: grad(value, c)
        for name, value in f.items()
    }

    rho, mu, mu_eff, mu_t = effective_viscosity(f, d, cfg)

    u, v = f["u"], f["v"]
    k, eps = f["k"], f["epsilon"]

    ux = d["u"][:, 0:1]
    uy = d["u"][:, 1:2]
    vx = d["v"][:, 0:1]
    vy = d["v"][:, 1:2]

    production = mu_t * (
        2.0 * ux**2
        + 2.0 * vy**2
        + (uy + vx)**2
    )

    dk = d["k"]
    de = d["epsilon"]

    adv_k = (
        dk[:, 2:3]
        + u * dk[:, 0:1]
        + v * dk[:, 1:2]
    )

    adv_eps = (
        de[:, 2:3]
        + u * de[:, 0:1]
        + v * de[:, 1:2]
    )

    # Diffusion avec coefficients variables.
    flux_k_x = mu_eff / cfg.sigma_k * dk[:, 0:1]
    flux_k_y = mu_eff / cfg.sigma_k * dk[:, 1:2]

    flux_eps_x = mu_eff / cfg.sigma_eps * de[:, 0:1]
    flux_eps_y = mu_eff / cfg.sigma_eps * de[:, 1:2]

    div_k = (
        grad(flux_k_x, c)[:, 0:1]
        + grad(flux_k_y, c)[:, 1:2]
    )

    div_eps = (
        grad(flux_eps_x, c)[:, 0:1]
        + grad(flux_eps_y, c)[:, 1:2]
    )

    rk = (
        rho * adv_k
        - production
        + rho * eps
        - div_k
    )

    re = (
        rho * adv_eps
        - cfg.C_eps1 * production * eps / (k + 1e-8)
        + cfg.C_eps2 * rho * eps**2 / (k + 1e-8)
        - div_eps
    )

    scale = cfg.rho_water * max(abs(cfg.gravity_y), 1.0)

    return mse(rk / scale), mse(re / scale)


def initial_loss(model, points, cfg):
    f = model(points)
    y = points[:, 1:2]

    target_alpha = torch.sigmoid(
        (cfg.initial_level - y) / cfg.interface_smoothing
    )

    # Hypothèse provisoire : fluide initialement au repos.
    return (
        mse(f["u"])
        + mse(f["v"])
        + mse(f["alpha"] - target_alpha)
    )


def wall_loss(model, points):
    f = model(points)
    return mse(f["u"]) + mse(f["v"])


def pressure_gauge_loss(model, point):
    """Fixe une jauge de pression pour éliminer la constante arbitraire."""
    return mse(model(point)["p"])