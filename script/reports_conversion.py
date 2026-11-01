"""
Conversion .tex -> .docx et .odt via Pandoc.
La gestion de la bibliographie (--citeproc + --bibliography) est activée
automatiquement si un fichier .bib est présent à côté du .tex.
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

try:
    import pypandoc
except ImportError:
    sys.exit("pypandoc manquant : pip install pypandoc")

logger = logging.getLogger("reports_conversion")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def verifier_pandoc() -> str:
    """Vérifie que le binaire pandoc est disponible et retourne sa version."""
    if shutil.which("pandoc") is None:
        raise RuntimeError(
            "Pandoc introuvable. Installez-le :\n"
            "  Linux  : sudo apt install pandoc (ou .deb officiel)\n"
            "  macOS  : brew install pandoc\n"
            "  Windows: https://pandoc.org/installing.html"
        )
    version = pypandoc.get_pandoc_version()
    logger.info(f"Pandoc détecté : version {version}")
    return version


def convertir_tex(chemin_tex: str | Path, chemin_sortie_base: str | Path) -> dict[str, Path | None]:
    """
    Convertit un .tex en .docx et .odt.

    La bibliographie est traitée uniquement si un fichier .bib est trouvé
    (par défaut : même nom que le .tex, ou 'references.bib' dans le même dossier).
    """
    chemin_tex = Path(chemin_tex).resolve()
    chemin_sortie_base = Path(chemin_sortie_base).resolve()

    if not chemin_tex.is_file():
        raise FileNotFoundError(f"Fichier .tex introuvable : {chemin_tex}")

    verifier_pandoc()

    # --- Détection de la bibliographie ---
    dossier = chemin_tex.parent
    candidats_bib = [
        dossier / f"{chemin_tex.stem}.bib",   # même nom que le .tex
        dossier / "references.bib",           # nom classique
        dossier / "bibliography.bib",
    ]
    bib_path = next((p for p in candidats_bib if p.is_file()), None)

    # --- Options Pandoc ---
    options_communes = [
        "--standalone",
        "--number-sections",
        f"--resource-path={dossier}",  # important : permet à Pandoc de trouver images/.bib
    ]

    if bib_path is not None:
        logger.info(f"Bibliographie détectée : {bib_path}")
        options_communes += [
            "--citeproc",
            f"--bibliography={bib_path}",
        ]
    else:
        logger.info("Aucun fichier .bib trouvé : conversion sans bibliographie.")

    sorties: dict[str, Path | None] = {}

    for ext, fmt in (("docx", "docx"), ("odt", "odt")):
        sortie = chemin_sortie_base.with_suffix(f".{ext}")
        try:
            logger.info(f"Conversion -> {sortie}")
            pypandoc.convert_file(
                str(chemin_tex),
                fmt,
                outputfile=str(sortie),
                extra_args=options_communes,
            )
            sorties[ext] = sortie
            logger.info(f"  OK ({sortie.stat().st_size / 1024:.1f} Ko)")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"  Échec {ext} : {exc}")
            sorties[ext] = None

    return sorties


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("mon_document.tex")
    base = src.with_name(src.stem + "_converti")
    resultats = convertir_tex(src, base)
    for fmt, path in resultats.items():
        print(f"{fmt}: {path}" if path else f"{fmt}: ÉCHEC")