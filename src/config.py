
import json
import torch

from dataclasses import dataclass, asdict, fields
from pathlib import Path


@dataclass
class Config:
    # Domaine spatial et temporel — SI
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

    # Mouvement horizontal du réservoir
    tank_amplitude: float = 2.0
    tank_omega: float = 0.5
    tank_phase: float = 0.0

    # Constantes k-epsilon
    C_mu: float = 0.09
    C_eps1: float = 1.44
    C_eps2: float = 1.92
    sigma_k: float = 1.0
    sigma_eps: float = 1.3

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
    w_k: float = 0.1
    w_epsilon: float = 0.1
    w_initial: float = 5.0
    w_wall: float = 2.0
    w_pressure_gauge: float = 0.01

    @property
    def mu_water(self):
        return self.rho_water * self.nu_water

    @property
    def mu_air(self):
        return self.rho_air * self.nu_air

    def validate(self):
        assert self.x_max > self.x_min
        assert self.y_max > self.y_min
        assert self.t_max > self.t_min
        assert self.rho_water > 0 and self.rho_air > 0
        assert self.nu_water > 0 and self.nu_air > 0
        assert self.interface_smoothing > 0
        return self

    def save_json(self, filepath="config.json"):
        """Enregistre les paramètres dans un fichier JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(
                asdict(self),
                f,
                indent=4,
                ensure_ascii=False
            )

    @classmethod
    def load_json(cls, filepath="config.json", validate=True):
        """Charge les paramètres JSON en conservant les valeurs par défaut
        des paramètres absents."""
        filepath = Path(filepath)

        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                "Le fichier JSON doit contenir un objet de configuration."
            )

        # Ignore les clés inconnues et conserve les valeurs par défaut
        # pour les paramètres absents du fichier.
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
