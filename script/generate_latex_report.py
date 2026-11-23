#!/usr/bin/env python3
"""
Script Python qui génère un fichier LaTeX (report.tex) à la structure du notebook
'notebooks/Etude_comparative_16_configurations_SloshingPINN.ipynb', en y incluant :
  - les figures de reports/figures/ (openfoam_vs_*, residuals_*, loss_*, scores, etc.)
  - les tableaux de reports/tables/ (configuration.csv, metrics.csv, results.csv, factorial_analysis.csv)
Le script ne dépend pas du contenu exact des checkpoints ni des exports OpenFOAM.
"""

from pathlib import Path
import sys
import csv
import os

# --- Chemins du projet ---
ROOT = Path(__file__).resolve().parents[1]  # SINN/
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
OUTPUT_TEX = ROOT / "reports" / "etude_comparative_16_configurations.tex"

# --- Utilitaires ---

def escape_latex(text: str) -> str:
    """Échappe les caractères spéciaux LaTeX."""
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("^", "\\textasciicircum{}")
        .replace("~", "\\textasciitilde{}")
    )


def csv_to_latex_table(csv_path, caption, label, columns_beautify=None):
    """
    Convertit un CSV simple en tableau LaTeX booktabs.
    Si le fichier n'existe pas, retourne un message indiquant qu'il est introuvable.
    """
    if not csv_path.is_file():
        sub_sec = "\\subsection{" + escape_latex(caption) + "}"
        para = "\\paragraph{Fichier introuvable: " + csv_path.name + "}"
        return sub_sec + "\n" + para + "\n"

    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return f"\\subsection{{{escape_latex(caption)}}}\n\\paragraph{{Tableau vide.}}\n"

        if columns_beautify is None:
            fieldnames = reader.fieldnames or []
        else:
            fieldnames = list(columns_beautify.keys())

        headers = [columns_beautify.get(col, escape_latex(col)) for col in fieldnames]
        ncols = len(fieldnames)
        sep = "c" * ncols

        lines = []
        lines.append(f"\\subsection{{{escape_latex(caption)}}}")
        lines.append(f"\\begin{{table}}[h!]")
        lines.append(f"\\centering")
        lines.append(f"\\begin{{tabular}}{{{sep}}}\\toprule")
        lines.append(" & ".join(headers) + r" \\\\\\midrule")

        for row in rows:
            vals = []
            for col in fieldnames:
                val = row.get(col, "")
                if val is None or str(val).strip() == "":
                    val = "-"
                vals.append(escape_latex(str(val)))
            lines.append(" & ".join(vals) + r" \\\\")

        lines.append(r"\\bottomrule")
        lines.append(f"\\end{{tabular}}")
        lines.append(f"\\end{{table}}")
        lines.append(f"\\label{{{escape_latex(label)}}}")
        lines.append("")
        return "\n".join(lines)
    except Exception as e:
        sub_sec = "\\subsection{" + escape_latex(caption) + "}"
        para = "\\paragraph{Erreur: " + str(e) + "}"
        return sub_sec + "\n" + para + "\n"


def collect_figures_categories():
    """Regroupe les figures par catégories reconnues."""
    categories = {}
    seen_names = set()
    prefix_to_key = {
        "openfoam_vs": "openfoam_vs",
        "residulas": "residulas",
        "residuals": "residulas",
        "loss_": "loss_",
        "alpha_score": "score_alpha",
        "u_score": "score_u",
        "v_score": "score_v",
        "p_score": "score_p",
        "rmse_score": "score_rmse",
        "relative_l2_score": "score_relative_l2",
        "mae_score": "score_mae",
        "p95_score": "score_p95",
    }

    if not FIGURES.exists():
        return categories

    all_png = sorted(FIGURES.glob("*.png"))

    for fig_path in all_png:
        name_stem = fig_path.stem
        assigned = False
        for prefix, cat_key in prefix_to_key.items():
            if name_stem.startswith(prefix):
                if cat_key not in categories:
                    categories[cat_key] = []
                if fig_path.name not in seen_names:
                    categories[cat_key].append(fig_path)
                    seen_names.add(fig_path.name)
                assigned = True
                break
        if not assigned:
            if "autres" not in categories:
                categories["autres"] = []
            if fig_path.name not in seen_names:
                categories["autres"].append(fig_path)
                seen_names.add(fig_path.name)

    return categories

def generate_tex() -> None:
    """Génère le fichier LaTeX complet."""
    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    categories = collect_figures_categories()

    lines = []

    # Preamble
    lines.append(r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage[hidelinks]{hyperref}
\usepackage{array}
\usepackage{float}
\geometry{a4paper, left=2cm, right=2cm, top=2cm, bottom=2cm}
\title{Étude comparative des 16 configurations de SloshingPINN}
\author{Script Python automatique}
\date{\today}
\begin{document}
\maketitle
""")

    # --- Section : Plan factoriel ---
    lines.append(r"\section{Plan factoriel des 16 configurations}")
    lines.append(
        "Protocole et d'analyse pour les quatre options : Fourier Features, Adaptive Sampling, Temporal Segmentation, Adaptive Weighting (GradNorm)."
    )
    lines.append("")
    lines.append(csv_to_latex_table(
        TABLES / "configuration.csv",
        "Tableau récapitulatif des 16 configurations",
        "tab:configurations",
        columns_beautify={
            "configuration": "Configuration",
            "Fourier Features": "Fourier Features",
            "Adaptive Sampling": "Adaptive Sampling",
            "Temporal Segmentation": "Temporal Segmentation",
            "Adaptive Weighting": "Adaptive Weighting",
        },
    ))
    lines.append("")

    # --- Section : Courbes de perte ---
    lines.append(r"\section{Courbes de perte (loss vs épochs)}")
    lines.append(
        "Représentation de l'évolution des fonctions de perte au cours de l'entraînement pour chaque configuration."
    )
    lines.append("")
    loss_figs = categories.get("loss_", [])
    if loss_figs:
        lines.append("\\begin{center}")
        for fig_path in loss_figs[:12]:
            fig_name = fig_path.name
            lines.append(
                f"\\includegraphics[width=0.45\\linewidth]{{figures/{fig_name}}}"
                + f"\\caption{{{escape_latex(fig_path.stem)}}}"
                + "\\hfill"
            )
        lines.append("\\end{center}")
        lines.append(f"({len(loss_figs)} figures de perte trouvées dans {FIGURES})")
    else:
        lines.append(
            "\\paragraph{Aucune figure de perte trouvée dans }"
            + f"{FIGURES}."
        )
    lines.append("")

    # --- Section : Chargement des données ---
    lines.append(r"\section{Chargement des données et des modèles}")
    lines.append(
        "OpenFOAM data : TODO."
    )
    lines.append(
        "Modèles entraînés : issu des 16 configurations factorielle."
    )
    lines.append("")
    config_csv = TABLES / "configuration.csv"
    if config_csv.exists():
        with open(config_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        n_configs = len(rows)
        lines.append(f"Nombre total de configurations : {n_configs}.")
    else:
        lines.append("Les 16 configurations sont définies par les combinaisons des quatre options.")
    lines.append("")

    # --- Section : Jeu d'évaluation commun ---
    lines.append(r"\section{Jeu d'évaluation commun}")
    lines.append(
        "Un jeu de validation commun, indépendant des données d'entraînement, est utilisé pour comparer les 16 configurations."
    )
    lines.append(
        "Les cellules dépendant du format exact des checkpoints et des exports OpenFOAM sont laissées à adapter."
    )
    lines.append("")

    # --- Section : Comparaison visuelle avec OpenFOAM ---
    lines.append(r"\section{Comparaison visuelle avec OpenFOAM}")
    lines.append(
        "Figure(s) montrant la comparaison entre les prédictions du modèle PINN et les données OpenFOAM pour les champs u, v, alpha sur des instants sélectionnés."
    )
    lines.append("")
    openfoam_figs = categories.get("openfoam_vs", [])
    if openfoam_figs:
        lines.append("\\begin{center}")
        gsize = 4
        for i in range(0, min(len(openfoam_figs), 60), gsize):
            group = openfoam_figs[i:i+gsize]
            lines.append("\\begin{tabular}{cccc}")
            for fig_path in group:
                fig_name = fig_path.name
                lines.append(
                    f"\\includegraphics[width=0.22\\linewidth]{{figures/{fig_name}}} & "
                )
            lines[-1] = lines[-1].rstrip(" & ")
            lines.append(r"\\")
            lines.append("\\end{tabular}")
            lines.append("\\vspace{0.5cm}")
        lines.append("\\end{center}")
        lines.append(f"({min(len(openfoam_figs), 60)} figures sur {len(openfoam_figs)} affichées)")
    else:
        lines.append(
            "\\paragraph{Aucune figure openfoam_vs_*.png trouvée dans }"
            + f"{FIGURES}."
        )
    lines.append("")

    # --- Section : Cartographie des résidus physiques ---
    lines.append(r"\section{Cartographie des résidus physiques}")
    lines.append(
        "Figure(s) montrant la carte des résidus physiques (PDE residuals) pour chaque modèle sur plusieurs instants."
    )
    lines.append("")
    residuals_figs = categories.get("residulas", [])
    if residuals_figs:
        lines.append("\\begin{center}")
        gsize = 4
        for i in range(0, min(len(residuals_figs), 60), gsize):
            group = residuals_figs[i:i+gsize]
            lines.append("\\begin{tabular}{cccc}")
            for fig_path in group:
                fig_name = fig_path.name
                lines.append(
                    f"\\includegraphics[width=0.22\\linewidth]{{figures/{fig_name}}} & "
                )
            lines[-1] = lines[-1].rstrip(" & ")
            lines.append(r"\\")
            lines.append("\\end{tabular}")
            lines.append("\\vspace{0.5cm}")
        lines.append("\\end{center}")
        lines.append(f"({min(len(residuals_figs), 60)} figures sur {len(residuals_figs)} affichées)")
    else:
        lines.append(
            "\\paragraph{Aucune figure residulas_*.png ni residuals_*.png trouvée dans }"
            + f"{FIGURES}."
        )
    lines.append("")

    # --- Section : Évaluation quantitative ---
    lines.append(r"\section{Évaluation quantitative}")
    lines.append(
        "Indicateurs : MAE, RMSE, erreur relative L2, percentile 95 de l'erreur absolue."
    )
    lines.append("")
    metrics_csv = TABLES / "metrics.csv"
    lines.append(csv_to_latex_table(
        metrics_csv,
        "Tableau des métriques d'erreur par configuration et champ",
        "tab:metrics",
        columns_beautify={
            "configuration": "Config",
            "field": "Champ",
            "MAE": "MAE",
            "RMSE": "RMSE",
            "Relative_L2": "Rel. L2",
            "P95_abs_error": "P95 abs",
        },
    ))
    lines.append("")

    # --- Section : Comparaison et classement ---
    lines.append(r"\section{Comparaison et classement}")
    lines.append("")
    score_keys = ["score_rmse", "score_relative_l2", "score_mae", "score_p95"]
    for sk in score_keys:
        score_figs = categories.get(sk, [])
        if score_figs:
            lines.append(f"\\subsection{{Comparaison {sk}}}")
            lines.append("\\begin{center}")
            for fig_path in score_figs[:6]:
                fig_name = fig_path.name
                lines.append(
                    f"\\includegraphics[width=0.3\\linewidth]{{figures/{fig_name}}}"
                    + f"\\caption{{{escape_latex(fig_path.stem)}}}"
                )
            lines.append("\\end{center}")
            lines.append("")

    results_csv = TABLES / "results.csv"
    lines.append(csv_to_latex_table(
        results_csv,
        "Tableau des résultats complets (incluant entraînement et métriques)",
        "tab:results",
        columns_beautify=None,
    ))
    lines.append("")

    # --- Section : Effets des options ---
    lines.append(r"\section{Effets des options (analyse factorielle descriptive)}")
    lines.append(
        "Un effet négatif sur RMSE signifie que l'option activée réduit le score moyen. "
        "Pour interpréter solidement les effets et interactions, entraîner si possible plusieurs graines par configuration."
    )
    lines.append("")
    factorial_csv = TABLES / "factorial_analysis.csv"
    lines.append(csv_to_latex_table(
        factorial_csv,
        "Analyse factorielle : effet de chaque option sur chaque champ (RMSE)",
        "tab:factorial",
        columns_beautify={
            "field": "Champ",
            "factor": "Option",
            "mean_when_on": "Moyenne (option ON)",
            "mean_when_off": "Moyenne (option OFF)",
            "effect_on_minus_off": "Effet (ON - OFF)",
        },
    ))
    lines.append("")

    # --- Fermeture ---
    lines.append(r"\end{document}")

    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Report LaTeX généré : {OUTPUT_TEX}")
    print(f"   Catégories de figures trouvées : {list(categories.keys())}")


if __name__ == "__main__":
    generate_tex()
