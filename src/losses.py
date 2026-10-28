import torch

from physics import (
    grad,
    mixture_properties,
    tank_acceleration,
)


def mse(value):
    """Compute the mean squared error of a non-dimensional residual.

    Formula::

        MSE(r) = (1/N) * sum(r_i**2)

    Parameters
    ----------
    value : torch.Tensor
        Non-dimensional physical residual.

    Returns
    -------
    torch.Tensor
        Scalar non-dimensional loss.
    """
    return torch.mean(value**2)


def continuity_loss(model, points, cfg):
    """Penalise mass conservation violation for 2D incompressible flow.

    Non-dimensional equation::

        du*/dx* + dv*/dy* = 0

    Residual::

        r_cont = du*/dx* + dv*/dy*

    Loss::

        L_cont = MSE(r_cont)

    Coordinates and velocities must be non-dimensional.

    Parameters
    ----------
    model : torch.nn.Module
        Network predicting u* and v*.
    points : torch.Tensor
        Collocation points, shape ``(N, 3)`` with columns ``(x*, y*, t*)``.
    cfg : object
        Problem configuration.

    Returns
    -------
    torch.Tensor
        Non-dimensional continuity loss.
    """
    c = points.detach().clone().requires_grad_(True)
    f = model(c)

    du = grad(f["u"], c)
    dv = grad(f["v"], c)

    # Divergence of the non-dimensional velocity field.
    residual = du[:, 0:1] + dv[:, 1:2]

    return mse(residual)



def volume_fraction_loss(model, points, cfg=None):
    """Conservative volume fraction transport:

        d(alpha)/dt* + div*(alpha u*) = 0

    Parameters
    ----------
    model : torch.nn.Module
        Network predicting alpha, u, v.
    points : torch.Tensor
        Points, shape ``(N, 3)`` with columns ``(x*, y*, t*)``.
    cfg : object, optional
        Problem configuration (unused in the residual itself).

    Returns
    -------
    torch.Tensor
        Non-dimensional volume fraction loss.
    """
    c = points.detach().clone().requires_grad_(True)
    f = model(c)

    alpha = f["alpha"]
    u = f["u"]
    v = f["v"]

    grad_alpha = grad(alpha, c)
    grad_alpha_u = grad(alpha * u, c)
    grad_alpha_v = grad(alpha * v, c)

    residual = (
        grad_alpha[:, 2:3]
        + grad_alpha_u[:, 0:1]
        + grad_alpha_v[:, 1:2]
    )

    return mse(residual)


def momentum_loss(model, points, cfg):
    """Penalise violation of non-dimensional momentum equations
    for laminar water–air flow.

    Vector equation::

        rho* Du*/Dt* = -grad(p*) + (1/Re) div(tau*)
                       + (rho*/Fr**2) b* + (1/We) kappa* grad(alpha)

    Viscous stress tensor::

        tau* = mu* [grad(u*) + grad(u*)^T]

    where::

        rho* = rho / rho_ref
        mu*  = mu / mu_ref
        Re   = rho_ref U_ref L_ref / mu_ref
        Fr   = U_ref / sqrt(g_ref L_ref)
        We   = rho_ref U_ref**2 L_ref / sigma

    Material acceleration::

        Du*/Dt* = du*/dt* + u* du*/dx* + v* du*/dy*
        Dv*/Dt* = dv*/dt* + u* dv*/dx* + v* dv*/dy*

    Curvature and CSF surface tension force::

        n = grad(alpha) / |grad(alpha)|
        kappa* = -div(n)
        f_sigma* = kappa* grad(alpha)

    The effective acceleration b* is the effective acceleration divided by
    gravity_ref. For an accelerated tank with y-axis pointing upward::

        b_x* = -a_tank / gravity_ref
        b_y* = gravity_y / gravity_ref

    Residuals::

        r_x = rho* Du*/Dt* + dp*/dx* - div(tau*)_x / Re
              - rho* b_x* / Fr**2 - kappa* alpha_x* / We

        r_y = rho* Dv*/Dt* + dp*/dy* - div(tau*)_y / Re
              - rho* b_y* / Fr**2 - kappa* alpha_y* / We

    Loss::

        L_momentum = MSE(r_x) + MSE(r_y)

    Notes
    -----
    Network inputs and outputs must be non-dimensional.
    ``mixture_properties`` must return rho and mu in the physical units
    used for rho_ref and mu_ref.
    The curvature sign convention must remain consistent with the
    surface tension force.

    Parameters
    ----------
    model : torch.nn.Module
        Network predicting u*, v*, p* and alpha.
    points : torch.Tensor
        Collocation points, shape ``(N, 3)`` with columns ``(x*, y*, t*)``.
    cfg : object
        Configuration containing Re, Fr, We, rho_ref, mu_ref,
        gravity_ref and mixture physical properties.

    Returns
    -------
    torch.Tensor
        Non-dimensional momentum loss.
    """
    c = points.detach().clone().requires_grad_(True)
    f = model(c)

    # Derivatives of fields with respect to x*, y* and t*.
    d = {
        name: grad(value, c)
        for name, value in f.items()
    }

    # Mixture properties in physical units, then non-dimensionalised.
    rho, mu = mixture_properties(f["alpha"], cfg)
    rho_star = rho / cfg.rho_ref
    mu_star = mu / cfg.mu_ref

    u, v, p = f["u"], f["v"], f["p"]

    ux = d["u"][:, 0:1]
    uy = d["u"][:, 1:2]
    ut = d["u"][:, 2:3]

    vx = d["v"][:, 0:1]
    vy = d["v"][:, 1:2]
    vt = d["v"][:, 2:3]

    px = d["p"][:, 0:1]
    py = d["p"][:, 1:2]

    # Non-dimensional viscous stresses with variable viscosity.
    # mu* is kept inside the divergence: its spatial gradients must not
    # be neglected at the water–air interface.
    tau_xx = 2.0 * mu_star * ux
    tau_yy = 2.0 * mu_star * vy
    tau_xy = mu_star * (uy + vx)

    div_tau_x = (
        grad(tau_xx, c)[:, 0:1]
        + grad(tau_xy, c)[:, 1:2]
    )

    div_tau_y = (
        grad(tau_xy, c)[:, 0:1]
        + grad(tau_yy, c)[:, 1:2]
    )

    # Effective acceleration in the tank reference frame.
    # gravity_y must be expressed with the same sign as the y-axis.
    t_physical = c[:, 2:3] * cfg.time_ref
    a_tank = tank_acceleration(t_physical, cfg)
    b_x_star = -a_tank / cfg.gravity_ref
    b_y_star = (
        torch.zeros_like(b_x_star)
        + cfg.gravity_y / cfg.gravity_ref
    )

    # CSF surface tension force: normal computed from alpha gradient.
    da = d["alpha"]
    alpha_x = da[:, 0:1]
    alpha_y = da[:, 1:2]

    grad_alpha_norm = torch.sqrt(
        alpha_x**2 + alpha_y**2 + 1e-12
    )

    nx = alpha_x / grad_alpha_norm
    ny = alpha_y / grad_alpha_norm

    # Non-dimensional curvature with convention kappa* = -div(n).
    curvature = -(
        grad(nx, c)[:, 0:1]
        + grad(ny, c)[:, 1:2]
    )

    # Non-dimensional material derivatives.
    adv_u = ut + u * ux + v * uy
    adv_v = vt + u * vx + v * vy

    # Residuals: all terms are now non-dimensional.
    rx = (
        rho_star * adv_u
        + px
        - div_tau_x / cfg.Re
        - rho_star * b_x_star / cfg.Fr**2
        - curvature * alpha_x / cfg.We
    )

    ry = (
        rho_star * adv_v
        + py
        - div_tau_y / cfg.Re
        - rho_star * b_y_star / cfg.Fr**2
        - curvature * alpha_y / cfg.We
    )

    return mse(rx) + mse(ry)


def initial_loss(model, points, cfg):
    """Penalise violation of non-dimensional initial conditions.

    Assumptions::

        u*(x*, y*, 0) = 0
        v*(x*, y*, 0) = 0

    Smoothed initial volume fraction profile::

        alpha_0(y*) = sigmoid((initial_level - y*) / interface_smoothing)

    Loss::

        L_initial = MSE(u*) + MSE(v*) + MSE(alpha - alpha_0)

    ``initial_level`` and ``interface_smoothing`` must be expressed in
    the network's non-dimensional coordinates.

    Parameters
    ----------
    model : torch.nn.Module
        Network predicting u*, v* and alpha.
    points : torch.Tensor
        Points for evaluating initial conditions.
    cfg : object
        Configuration containing the non-dimensional initial level and
        smoothing parameter.

    Returns
    -------
    torch.Tensor
        Non-dimensional initial condition loss.
    """
    f = model(points)
    level_star = cfg.initial_level / cfg.length_ref
    smoothing_star = (
        cfg.interface_smoothing / cfg.length_ref
    )

    target_alpha = torch.sigmoid(
        (level_star - points[:, 1:2]) / smoothing_star
    )

    return (
        mse(f["u"])
        + mse(f["v"])
        + mse(f["alpha"] - target_alpha)
    )


def wall_loss(model, points, cfg):
    """Penalise violation of the no-slip condition on fixed walls.

    Conditions::

        u* = 0
        v* = 0

    Loss::

        L_wall = MSE(u*) + MSE(v*)

    Points must belong to the relevant walls. For a moving wall, the
    non-dimensional wall velocity must be imposed in the chosen frame.

    Parameters
    ----------
    model : torch.nn.Module
        Network predicting u* and v*.
    points : torch.Tensor
        Non-dimensional boundary points, shape ``(N, 3)`` with columns
        ``(x*, y*, t*)``.

    Returns
    -------
    torch.Tensor
        Non-dimensional wall loss.
    """
    f = model(points)
    return mse(f["u"]) + mse(f["v"])


def pressure_gauge_loss(model, point, cfg):
    """Fix the non-dimensional pressure reference at a point.

    Condition::

        p*(x_ref*, y_ref*, t_ref*) = 0

    Loss::

        L_pressure = MSE(p*)

    This condition removes the additive indeterminacy of pressure.
    It does not mean pressure is zero everywhere in the domain.

    Parameters
    ----------
    model : torch.nn.Module
        Network predicting non-dimensional pressure p*.
    point : torch.Tensor
        Non-dimensional reference point.

    Returns
    -------
    torch.Tensor
        Non-dimensional pressure gauge loss.
    """
    return mse(model(point)["p"])