from pathlib import Path
import re
import numpy as np
import pyvista as pv


def read_gravity(case: Path) -> np.ndarray:
    """Read the gravity vector from an OpenFOAM case (constant/g)."""
    g_file = case / "constant" / "g"
    text = g_file.read_text()
    m = re.search(r"value\s*\(\s*([^)]+)\)", text)
    if not m:
        raise ValueError(f"Could not parse gravity from {g_file}")
    return np.asarray([float(v) for v in m.group(1).split()], dtype=float)


def read_gravity(case: Path) -> np.ndarray:
    """Read the gravity vector from an OpenFOAM case (constant/g)."""
    g_file = case / "constant" / "g"
    text = g_file.read_text()
    m = re.search(r"value\s*\(\s*([^)]+)\)", text)
    if not m:
        raise ValueError(f"Could not parse gravity from {g_file}")
    return np.asarray([float(v) for v in m.group(1).split()], dtype=float)


DIR = Path(__file__).parent
CASE = DIR / "../sloshing/sloshing"
OUTPUT = DIR / "../data/sloshing_data.npz"


def build_dataset(case: Path = CASE, output: Path = OUTPUT):
    """Extract CFD data from an OpenFOAM case and save it to a compressed file.

    Parameters
    ----------
    case : pathlib.Path, optional
        Path to the OpenFOAM case directory. Default is
        ``../sloshing/sloshing`` relative to this file.
    output : pathlib.Path, optional
        Path to the output compressed ``.npz`` file. Default is
        ``../data/sloshing_data.npz`` relative to this file.

    Returns
    -------
    None
        The dataset is saved to disk; nothing is returned.

    Notes
    -----
    Reads cell-centered fields (alpha.eau, U, p_rgh, k, epsilon) at all
    available time steps, synchronises them with force and moment data,
    and saves the synchronised dataset as a compressed NPZ file.
    """

    # ============================================================
    # OpenFOAM
    # ============================================================

    foam_file = case / "sloshing.foam"

    reader = pv.OpenFOAMReader(str(foam_file))
    reader.enable_all_cell_arrays()

    times = np.asarray(reader.time_values, dtype=float)

    print(f"Nombre de temps CFD : {len(times)}")
    print(f"t min = {times[0]}")
    print(f"t max = {times[-1]}")


    # ============================================================
    # CFD fields
    # ============================================================

    X_all = []
    alpha_all = []
    u_all = []
    v_all = []
    w_all = []
    p_all = []
    p_rgh_all = []
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
        u = np.asarray(U[:, 0])
        v = np.asarray(U[:, 1])
        w = np.asarray(U[:, 2])
        p_rgh = np.asarray(mesh["p_rgh"])
        # Mixture density (water/air) consistent with src/physics.py
        rho_water = 998.0
        rho_air = 1.21
        rho = alpha * rho_water + (1.0 - alpha) * rho_air
        g_vec = read_gravity(case)
        p = p_rgh + rho * (centers @ g_vec)
        k = np.asarray(mesh["k"])
        epsilon = np.asarray(mesh["epsilon"])

        X_all.append(centers)
        alpha_all.append(alpha)
        u_all.append(u)
        v_all.append(v)
        w_all.append(w)
        p_all.append(p)
        p_rgh_all.append(p_rgh)
        k_all.append(k)
        epsilon_all.append(epsilon)


    X_all = np.asarray(X_all)
    alpha_all = np.asarray(alpha_all)
    u_all = np.asarray(u_all)
    v_all = np.asarray(v_all)
    w_all = np.asarray(w_all)
    p_all = np.asarray(p_all)
    p_rgh_all = np.asarray(p_rgh_all)
    k_all = np.asarray(k_all)
    epsilon_all = np.asarray(epsilon_all)


    # ============================================================
    # Lecture des forces
    # ============================================================

    force_file = (
        case
        / "postProcessing"
        / "forceCompositeMonitorOnRightInDefault"
        / "0"
        / "force.dat"
    )

    print()
    print("Lecture des forces...")

    force_data = np.loadtxt(force_file)

    force_time = force_data[:, 0]

    # Colonnes 1,2,3 = Fx, Fy, Fz
    Fx = force_data[:, 1]
    Fy = force_data[:, 2]

    print(f"Nombre de points force : {len(force_time)}")


    # ============================================================
    # Lecture des moments
    # ============================================================

    moment_file = (
        case
        / "postProcessing"
        / "forceCompositeMonitorOnRightInDefault"
        / "0"
        / "moment.dat"
    )

    print("Lecture des moments...")

    moment_data = np.loadtxt(moment_file)

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
    u_sync = u_all[mask]
    v_sync = v_all[mask]
    w_sync = w_all[mask]
    p_sync = p_all[mask]
    p_rgh_sync = p_rgh_all[mask]
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
    print("u        :", u_sync.shape)
    print("alpha    :", alpha_sync.shape)
    print("p        :", p_sync.shape)
    print("k        :", k_sync.shape)
    print("epsilon  :", epsilon_sync.shape)
    print("Fx       :", Fx_sync.shape)
    print("Fy       :", Fy_sync.shape)
    print("Mz       :", Mz_sync.shape)

    # ============================================================
    # Sauvegarde
    # ============================================================

    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,

        time=times_sync,

        X=X_sync,
        u=u_sync,
        v=v_sync,
        w=w_sync,
        alpha=alpha_sync,
        p=p_sync,
        p_rgh=p_rgh_sync,
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

    print("Fichier :", output)

    print()
    print("Shapes :")

    print("time      :", times_sync.shape)
    print("X         :", X_sync.shape)
    print("u         :", u_sync.shape)
    print("alpha     :", alpha_sync.shape)
    print("p         :", p_sync.shape)
    print("k         :", k_sync.shape)
    print("epsilon   :", epsilon_sync.shape)

    print()
    print("Forces :")

    print("Fx        :", Fx_sync.shape)
    print("Fy        :", Fy_sync.shape)
    print("Mz        :", Mz_sync.shape)