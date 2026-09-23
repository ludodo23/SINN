import os
from pathlib import Path
import numpy as np
import pyvista as pv

DIR = Path(__file__).parent

CASE = Path(
    DIR,
    r"../sloshing/sloshing"
)

DATA = Path(
    DIR,
    r"../data"
)

FOAM_FILE = CASE / "sloshing.foam"
OUTPUT = DATA / "sloshing_data.npz"


reader = pv.OpenFOAMReader(str(FOAM_FILE))
reader.enable_all_cell_arrays()

times = np.asarray(reader.time_values, dtype=float)

print(f"Nombre de temps : {len(times)}")
print(f"t min = {times[0]}")
print(f"t max = {times[-1]}")

X_all = []
alpha_all = []
U_all = []
p_all = []
k_all = []
epsilon_all = []
t_all = []


for i, t in enumerate(times):

    print(f"[{i+1}/{len(times)}] t = {t:.6g}")

    reader.set_active_time_value(float(t))

    data = reader.read()
    mesh = data["internalMesh"]

    centers = mesh.cell_centers().points

    alpha = np.asarray(mesh["alpha.eau"])
    U = np.asarray(mesh["U"])
    p = np.asarray(mesh["p_rgh"])
    k = np.asarray(mesh["k"])
    epsilon = np.asarray(mesh["epsilon"])

    X_all.append(centers)
    alpha_all.append(alpha)
    U_all.append(U)
    p_all.append(p)
    k_all.append(k)
    epsilon_all.append(epsilon)

    t_all.append(
        np.full(alpha.shape, t, dtype=float)
    )


X_all = np.asarray(X_all)
alpha_all = np.asarray(alpha_all)
U_all = np.asarray(U_all)
p_all = np.asarray(p_all)
k_all = np.asarray(k_all)
epsilon_all = np.asarray(epsilon_all)
t_all = np.asarray(t_all)

os.makedirs(DATA, exist_ok=True)


np.savez_compressed(
    OUTPUT,
    time=times,
    X=X_all,
    alpha_eau=alpha_all,
    U=U_all,
    p_rgh=p_all,
    k=k_all,
    epsilon=epsilon_all,
    T=t_all,
)

print()
print("Fichier écrit :", OUTPUT)
print("X :", X_all.shape)
print("U :", U_all.shape)
print("alpha :", alpha_all.shape)