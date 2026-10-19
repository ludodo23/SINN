import torch

from physics import (
    grad,
    mixture_properties,
    tank_acceleration,
)


def mse(value):
    """
    Calcule la moyenne des carrés d'un résidu adimensionné.

    Formule :
        MSE(r) = (1/N) * sum(r_i**2)

    Paramètres
    ----------
    value : torch.Tensor
        Résidu physique adimensionné.

    Retour
    ------
    torch.Tensor
        Perte scalaire sans dimension.
    """
    return torch.mean(value**2)


def continuity_loss(model, points, cfg):
    """
    Pénalise la violation de la conservation de la masse
    pour un écoulement incompressible bidimensionnel.

    Équation adimensionnée :
        du*/dx* + dv*/dy* = 0

    Résidu :
        r_cont = du*/dx* + dv*/dy*

    Perte :
        L_cont = MSE(r_cont)

    Les coordonnées et les vitesses doivent être adimensionnées.

    Paramètres
    ----------
    model : torch.nn.Module
        Réseau prédisant u* et v*.
    points : torch.Tensor
        Points de collocation (N, 3) : (x*, y*, t*).
    cfg
        Configuration du problème.

    Retour
    ------
    torch.Tensor
        Perte adimensionnée de continuité.
    """
    c = points.detach().clone().requires_grad_(True)
    f = model(c)

    du = grad(f["u"], c)
    dv = grad(f["v"], c)

    # Divergence du champ de vitesse adimensionné.
    residual = du[:, 0:1] + dv[:, 1:2]

    return mse(residual)



def vof_loss(model, points, cfg=None):
    """
    Transport conservatif de la fraction volumique :
        d(alpha)/dt* + div*(alpha u*) = 0

    points : (N, 3), colonnes (x*, y*, t*).
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
    """
    Pénalise la violation des équations de quantité de mouvement
    adimensionnées pour un écoulement eau-air laminaire.

    Équation vectorielle :
        rho* Du*/Dt*
        = -grad(p*)
          + (1/Re) div(tau*)
          + (rho*/Fr**2) b*
          + (1/We) kappa* grad(alpha)

    Tenseur des contraintes visqueuses :
        tau* = mu* [grad(u*) + grad(u*)^T]

    où :
        rho* = rho / rho_ref
        mu*  = mu / mu_ref
        Re   = rho_ref U_ref L_ref / mu_ref
        Fr   = U_ref / sqrt(g_ref L_ref)
        We   = rho_ref U_ref**2 L_ref / sigma

    Accélération matérielle :
        Du*/Dt* = du*/dt* + u* du*/dx* + v* du*/dy*
        Dv*/Dt* = dv*/dt* + u* dv*/dx* + v* dv*/dy*

    Courbure et force capillaire CSF :
        n = grad(alpha) / |grad(alpha)|
        kappa* = -div(n)
        f_sigma* = kappa* grad(alpha)

    L'accélération b* est l'accélération effective divisée par
    gravity_ref. Pour un réservoir accéléré et un axe y orienté
    vers le haut :
        b_x* = -a_tank / gravity_ref
        b_y* = gravity_y / gravity_ref

    Résidus :
        r_x = rho* Du*/Dt* + dp*/dx*
              - div(tau*)_x / Re
              - rho* b_x* / Fr**2
              - kappa* alpha_x* / We

        r_y = rho* Dv*/Dt* + dp*/dy*
              - div(tau*)_y / Re
              - rho* b_y* / Fr**2
              - kappa* alpha_y* / We

    Perte :
        L_momentum = MSE(r_x) + MSE(r_y)

    IMPORTANT
    ----------
    Les entrées et sorties du réseau doivent être adimensionnées.
    La fonction mixture_properties doit renvoyer rho et mu dans
    les unités physiques utilisées pour rho_ref et mu_ref.
    La convention de signe de la courbure doit rester cohérente
    avec celle de la force capillaire.

    Paramètres
    ----------
    model : torch.nn.Module
        Réseau prédisant u*, v*, p* et alpha.
    points : torch.Tensor
        Points de collocation (N, 3) : (x*, y*, t*).
    cfg
        Configuration contenant Re, Fr, We, rho_ref, mu_ref,
        gravity_ref et les propriétés physiques du mélange.

    Retour
    ------
    torch.Tensor
        Perte adimensionnée de quantité de mouvement.
    """
    c = points.detach().clone().requires_grad_(True)
    f = model(c)

    # Dérivées des champs par rapport à x*, y* et t*.
    d = {
        name: grad(value, c)
        for name, value in f.items()
    }

    # Propriétés du mélange en unités physiques, puis adimensionnées.
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

    # Contraintes visqueuses adimensionnées avec viscosité variable.
    # On conserve mu* à l'intérieur de la divergence : ses gradients
    # spatiaux ne doivent pas être négligés à l'interface eau-air.
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

    # Accélération effective dans le repère du réservoir.
    # gravity_y doit être exprimé avec le même signe que l'axe y.
    t_physical = c[:, 2:3] * cfg.time_ref
    a_tank = tank_acceleration(t_physical, cfg)
    b_x_star = -a_tank / cfg.gravity_ref
    b_y_star = (
        torch.zeros_like(b_x_star)
        + cfg.gravity_y / cfg.gravity_ref
    )

    # Force capillaire CSF :
    # la normale est calculée à partir du gradient de alpha.
    da = d["alpha"]
    alpha_x = da[:, 0:1]
    alpha_y = da[:, 1:2]

    grad_alpha_norm = torch.sqrt(
        alpha_x**2 + alpha_y**2 + 1e-12
    )

    nx = alpha_x / grad_alpha_norm
    ny = alpha_y / grad_alpha_norm

    # Courbure adimensionnée selon la convention kappa* = -div(n).
    curvature = -(
        grad(nx, c)[:, 0:1]
        + grad(ny, c)[:, 1:2]
    )

    # Dérivées matérielles adimensionnées.
    adv_u = ut + u * ux + v * uy
    adv_v = vt + u * vx + v * vy

    # Résidus : tous les termes sont désormais sans dimension.
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
    """
    Pénalise la violation des conditions initiales adimensionnées.

    Hypothèses :
        u*(x*, y*, 0) = 0
        v*(x*, y*, 0) = 0

    Profil initial lissé de fraction volumique :
        alpha_0(y*) = sigmoid(
            (initial_level - y*) / interface_smoothing
        )

    Perte :
        L_initial = MSE(u*) + MSE(v*)
                    + MSE(alpha - alpha_0)

    initial_level et interface_smoothing doivent être exprimés
    dans les coordonnées adimensionnées du réseau.

    Paramètres
    ----------
    model : torch.nn.Module
        Réseau prédisant u*, v* et alpha.
    points : torch.Tensor
        Points d'évaluation des conditions initiales.
    cfg
        Configuration contenant le niveau initial et le
        paramètre de lissage adimensionnés.

    Retour
    ------
    torch.Tensor
        Perte adimensionnée des conditions initiales.
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
    """
    Pénalise la violation de la condition de non-glissement
    sur des parois fixes.

    Conditions :
        u* = 0
        v* = 0

    Perte :
        L_wall = MSE(u*) + MSE(v*)

    Les points doivent appartenir aux parois concernées.
    Pour une paroi mobile, il faut imposer la vitesse de paroi
    adimensionnée dans le repère choisi.

    Paramètres
    ----------
    model : torch.nn.Module
        Réseau prédisant u* et v*.
    points : torch.Tensor
        Points de bord adimensionnés (x*, y*, t*).

    Retour
    ------
    torch.Tensor
        Perte adimensionnée aux parois.
    """
    f = model(points)
    return mse(f["u"]) + mse(f["v"])


def pressure_gauge_loss(model, point, cfg):
    """
    Fixe la référence de pression adimensionnée en un point.

    Condition :
        p*(x_ref*, y_ref*, t_ref*) = 0

    Perte :
        L_pressure = MSE(p*)

    Cette condition supprime l'indétermination additive de la
    pression. Elle ne signifie pas que la pression est nulle
    dans tout le domaine.

    Paramètres
    ----------
    model : torch.nn.Module
        Réseau prédisant la pression adimensionnée p*.
    point : torch.Tensor
        Point de référence adimensionné.

    Retour
    ------
    torch.Tensor
        Perte adimensionnée de jauge de pression.
    """
    return mse(model(point)["p"])