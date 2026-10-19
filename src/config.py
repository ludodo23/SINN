
import json
import torch

from dataclasses import dataclass, asdict, fields
from pathlib import Path


@dataclass
class Config:
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

    @property
    def mu_water(self):
        return self.rho_water * self.nu_water

    @property
    def mu_air(self):
        return self.rho_air * self.nu_air

    @property
    def time_ref(self):
        return self.length_ref / self.velocity_ref

    @property
    def Re(self):
        return (
            self.rho_ref * self.velocity_ref * self.length_ref
            / self.mu_ref
        )

    @property
    def Fr(self):
        return self.velocity_ref / (
            self.gravity_ref * self.length_ref
        ) ** 0.5

    @property
    def We(self):
        return (
            self.rho_ref * self.velocity_ref**2 * self.length_ref
            / self.sigma
        )

    def validate(self):
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

        return self

    def save_json(self, filepath="config.json"):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4, ensure_ascii=False)

    @classmethod
    def load_json(cls, filepath="config.json", validate=True):
        filepath = Path(filepath)

        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                "Le fichier JSON doit contenir un objet de configuration."
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
