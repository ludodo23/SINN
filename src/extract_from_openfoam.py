from pathlib import Path
import numpy as np
import pyvista as pv


DIR = Path(__file__).parent

CASE = DIR / "../sloshing/sloshing"
DATA = DIR / "../data"

FOAM_FILE = CASE / "sloshing.foam"

FORCE_FILE = (
    CASE
    / "postProcessing"
    / "forceCompositeMonitorOnRightInDefault"
    / "0"
    / "force.dat"
)

MOMENT_FILE = (
    CASE
    / "postProcessing"
    / "forceCompositeMonitorOnRightInDefault"
    / "0"
    / "moment.dat"
)

OUTPUT = DATA / "sloshing_data.npz"


# ============================================================
# OpenFOAM
# ============================================================

reader = pv.OpenFOAMReader(str(FOAM_FILE))
reader.enable_all_cell_arrays()

times = np.asarray(reader.time_values, dtype=float)

print(f"Nombre de temps CFD : {len(times)}")
print(f"t min = {times[0]}")
print(f"t max = {times[-1]}")


# ============================================================
# Champs CFD
# ============================================================

X_all = []
alpha_all = []
U_all = []
p_all = []
k_all = []
epsilon_all = []


for i, t in enumerate(times):

    print(f"[CFD {i+1}/{len(times)}] t = {t:.6g}")

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


X_all = np.asarray(X_all)
alpha_all = np.asarray(alpha_all)
U_all = np.asarray(U_all)
p_all = np.asarray(p_all)
k_all = np.asarray(k_all)
epsilon_all = np.asarray(epsilon_all)


# ============================================================
# Lecture des forces
# ============================================================

print()
print("Lecture des forces...")

force_data = np.loadtxt(FORCE_FILE)

force_time = force_data[:, 0]

# Colonnes 1,2,3 = Fx, Fy, Fz
Fx = force_data[:, 1]
Fy = force_data[:, 2]

print(f"Nombre de points force : {len(force_time)}")


# ============================================================
# Lecture des moments
# ============================================================

print("Lecture des moments...")

moment_data = np.loadtxt(MOMENT_FILE)

moment_time = moment_data[:, 0]

# Colonnes 1,2,3 = Mx, My, Mz
Mz = moment_data[:, 3]

print(f"Nombre de points moment : {len(moment_time)}")


# ============================================================
# Synchronisation temporelle
# ============================================================

print()
print("Synchronisation temporelle...")

# Domaine temporel commun
t_min = max(
    times[0],
    force_time[0],
    moment_time[0]
)

t_max = min(
    times[-1],
    force_time[-1],
    moment_time[-1]
)

print(f"Intervalle CFD      : [{times[0]}, {times[-1]}]")
print(f"Intervalle forces   : [{force_time[0]}, {force_time[-1]}]")
print(f"Intervalle moments  : [{moment_time[0]}, {moment_time[-1]}]")
print(f"Intervalle commun   : [{t_min}, {t_max}]")


# Indices CFD appartenant à l'intervalle commun
mask = (
    (times >= t_min) &
    (times <= t_max)
)

times_sync = times[mask]

X_sync = X_all[mask]
alpha_sync = alpha_all[mask]
U_sync = U_all[mask]
p_sync = p_all[mask]
k_sync = k_all[mask]
epsilon_sync = epsilon_all[mask]


# ============================================================
# Interpolation des forces
# ============================================================

Fx_sync = np.interp(
    times_sync,
    force_time,
    Fx
)

Fy_sync = np.interp(
    times_sync,
    force_time,
    Fy
)

Mz_sync = np.interp(
    times_sync,
    moment_time,
    Mz
)


# ============================================================
# Vérification
# ============================================================

print()
print("Dataset synchronisé :")

print(f"Nombre de temps avant : {len(times)}")
print(f"Nombre de temps après : {len(times_sync)}")

print(f"t début : {times_sync[0]}")
print(f"t fin   : {times_sync[-1]}")

print()
print("Shapes :")
print("time     :", times_sync.shape)
print("X        :", X_sync.shape)
print("U        :", U_sync.shape)
print("alpha    :", alpha_sync.shape)
print("p_rgh    :", p_sync.shape)
print("k        :", k_sync.shape)
print("epsilon  :", epsilon_sync.shape)
print("Fx       :", Fx_sync.shape)
print("Fy       :", Fy_sync.shape)
print("Mz       :", Mz_sync.shape)

# ============================================================
# Sauvegarde
# ============================================================

DATA.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUTPUT,

    time=times_sync,

    X=X_sync,
    U=U_sync,
    alpha_eau=alpha_sync,
    p_rgh=p_sync,
    k=k_sync,
    epsilon=epsilon_sync,

    Fx=Fx_sync,
    Fy=Fy_sync,
    Mz=Mz_sync,
)


# ============================================================
# Résumé
# ============================================================

print()
print("=" * 60)
print("DATASET TERMINÉ")
print("=" * 60)

print("Fichier :", OUTPUT)

print()
print("Shapes :")

print("time      :", times_sync.shape)
print("X         :", X_sync.shape)
print("U         :", U_sync.shape)
print("alpha     :", alpha_sync.shape)
print("p_rgh     :", p_sync.shape)
print("k         :", k_sync.shape)
print("epsilon   :", epsilon_all.shape)

print()
print("Forces :")

print("Fx        :", Fx_sync.shape)
print("Fy        :", Fy_sync.shape)
print("Mz        :", Mz_sync.shape)