
import json
import torch

from dataclasses import dataclass, asdict, fields
from pathlib import Path


@dataclass
class Config:
    """Physical and numerical configuration for the Sloshing PINN.

    Attributes
    ----------
    x_min, x_max : float
        Physical domain bounds in x [m].
    y_min, y_max : float
        Physical domain bounds in y [m].
    t_min, t_max : float
        Time interval [s].
    rho_water, rho_air : float
        Phase densities [kg/m^3].
    nu_water, nu_air : float
        Phase kinematic viscosities [m^2/s].
    sigma : float
        Surface tension coefficient [N/m].
    gravity_y : float
        Vertical gravity component [m/s^2].
    length_ref, velocity_ref : float
        Reference length and velocity for non-dimensionalisation.
    rho_ref, mu_ref : float
        Reference density [kg/m^3] and dynamic viscosity [Pa.s].
    gravity_ref : float
        Reference gravity magnitude [m/s^2].
    tank_amplitude : float
        Horizontal tank motion amplitude [m].
    tank_omega : float
        Tank motion angular frequency [rad/s].
    tank_phase : float
        Tank motion phase shift [rad].
    initial_level : float
        Initial free-surface level [m].
    interface_smoothing : float
        Interface smoothing thickness [m].
    hidden_width : int
        Number of hidden units per layer.
    hidden_layers : int
        Number of hidden layers.
    n_collocation : int
        Number of interior collocation points.
    n_initial : int
        Number of initial-condition points.
    n_wall : int
        Number of wall boundary points.
    learning_rate : float
        Adam optimiser learning rate.
    epochs : int
        Number of training epochs.
    print_every : int
        Print loss every N epochs.
    seed : int
        Random seed.
    device : str
        PyTorch device ("cpu" or "cuda").
    model_name: str | None = None
        Nom du modèle / dossier de sauvegarde. Si défini, il remplace
        le nom horodaté par défaut.
    w_continuity, w_vof, w_momentum : float
        Loss weights for physics residuals.
    w_initial : float
        Loss weight for initial conditions.
    w_wall : float
        Loss weight for wall boundary conditions.
    w_pressure_gauge : float
        Loss weight for pressure gauge condition.
    """
    # Domaine physique SI
    x_min: float = 0.0
    x_max: float = 6.0
    y_min: float = -1.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 10.0

    # Propriétés des phases
    rho_water: float = 998.0
    rho_air: float = 1.21
    nu_water: float = 1.0e-6
    nu_air: float = 1.51e-5
    sigma: float = 0.07
    gravity_y: float = -9.81

    # Échelles de référence pour l'adimensionnement
    # À choisir selon la géométrie et le régime étudié.
    length_ref: float = 1.0
    velocity_ref: float = 1.0
    rho_ref: float = 998.0
    mu_ref: float = 0.000998
    gravity_ref: float = 9.81

    # Mouvement horizontal du réservoir
    tank_amplitude: float = 2.0
    tank_omega: float = 0.5
    tank_phase: float = 0.0

    # Interface initiale
    initial_level: float = -0.5
    interface_smoothing: float = 0.02

    # Réseau
    hidden_width: int = 96
    hidden_layers: int = 5

    # Entraînement
    n_collocation: int = 2048
    n_initial: int = 512
    n_wall: int = 512
    learning_rate: float = 1.0e-3
    epochs: int = 1000
    print_every: int = 50
    seed: int = 1234
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Poids des pertes
    w_continuity: float = 1.0
    w_vof: float = 1.0
    w_momentum: float = 1.0
    w_initial: float = 5.0
    w_wall: float = 2.0
    w_pressure_gauge: float = 0.01


    # --- Options expérimentales indépendantes ---
    use_fourier_features: bool = False
    use_adaptive_sampling: bool = False
    use_temporal_segmentation: bool = False
    use_adaptive_loss_weights: bool = False
    use_lbfgs: bool = False

    # Fourier features : fréquences des coordonnées adimensionnelles
    fourier_bands: tuple = (1, 2, 4, 8, 16)

    # Adaptive sampling
    adaptive_fraction: float = 0.5
    n_candidate_points: int = 8192
    sampling_update_interval: int = 100

    # Segmentation temporelle, en secondes physiques
    temporal_window_size: float = 1.0
    temporal_overlap: float = 0.0
    temporal_replay_fraction: float = 0.25

    # Pondération adaptative des pertes
    weight_update_interval = 1
    min_adaptive_weight: float = 0.1
    max_adaptive_weight: float = 10.0
    gradnorm_alpha: float = 0.5
    gradnorm_lr: float = 1e-3

    # Optimisation
    lbfgs_epochs: int = 1000
    lbfgs_max_iter: int = 200
    lbfgs_history_size: int = 50
    lbfgs_lr: float = 0.5

    @property
    def mu_water(self):
        """Dynamic viscosity of water [Pa.s]."""
        return self.rho_water * self.nu_water

    @property
    def mu_air(self):
        """Dynamic viscosity of air [Pa.s]."""
        return self.rho_air * self.nu_air

    @property
    def time_ref(self):
        """Reference time scale [s]."""
        return self.length_ref / self.velocity_ref

    @property
    def Re(self):
        """Reynolds number based on reference scales."""
        return (
            self.rho_ref * self.velocity_ref * self.length_ref
            / self.mu_ref
        )

    @property
    def Fr(self):
        """Froude number based on reference scales."""
        return self.velocity_ref / (
            self.gravity_ref * self.length_ref
        ) ** 0.5

    @property
    def We(self):
        """Weber number based on reference scales."""
        return (
            self.rho_ref * self.velocity_ref**2 * self.length_ref
            / self.sigma
        )

    def validate(self):
        """Validate configuration parameters.

        Returns
        -------
        Config
            The validated configuration instance (self).
        """
        assert self.x_max > self.x_min
        assert self.y_max > self.y_min
        assert self.t_max > self.t_min

        assert self.rho_water > 0 and self.rho_air > 0
        assert self.nu_water > 0 and self.nu_air > 0
        assert self.sigma > 0

        assert self.length_ref > 0
        assert self.velocity_ref > 0
        assert self.rho_ref > 0
        assert self.mu_ref > 0
        assert self.gravity_ref > 0

        assert self.interface_smoothing > 0
        assert self.print_every > 0

        assert 0.0 <= self.adaptive_fraction <= 1.0
        assert self.n_candidate_points >= self.n_collocation
        assert self.sampling_update_interval > 0
        assert self.temporal_window_size > 0
        assert 0.0 <= self.temporal_overlap < self.temporal_window_size
        assert 0.0 <= self.temporal_replay_fraction <= 1.0
        assert 0.0 < self.min_adaptive_weight <= self.max_adaptive_weight
        assert self.lbfgs_epochs > 0
        assert self.lbfgs_max_iter > 0

        return self

    def save_json(self, filepath="config.json"):
        """Save configuration to a JSON file.

        Parameters
        ----------
        filepath : str or pathlib.Path, optional
            Path to the output JSON file. Default is "config.json".
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4, ensure_ascii=False)

    @classmethod
    def load_json(cls, filepath="config.json", validate=True):
        """Load configuration from a JSON file.

        Parameters
        ----------
        filepath : str or pathlib.Path, optional
            Path to the JSON configuration file. Default is "config.json".
        validate : bool, optional
            If True, validate the loaded configuration. Default is True.

        Returns
        -------
        Config
            Configuration instance loaded from the file.
        """
        filepath = Path(filepath)

        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                "The JSON file must contain a configuration object."
            )

        valid_keys = {f.name for f in fields(cls)}
        filtered_data = {
            key: value
            for key, value in data.items()
            if key in valid_keys
        }

        config = cls(**filtered_data)

        if validate:
            config.validate()

        return config
