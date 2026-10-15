# Sloshing PINN — projet modulaire

## Objectif

Prototype de réseau de neurones informé par la physique (PINN)
pour un problème de ballottement eau-air en deux dimensions.

Les équations continues sont utilisées comme résidus d'entraînement.
Aucun snapshot OpenFOAM n'est utilisé comme donnée d'entraînement.
OpenFOAM est réservé à la validation indépendante.

## Structure

- config.py : paramètres physiques et numériques.
- model.py : architecture du réseau.
- physics.py : propriétés du mélange et dérivées.
- losses.py : résidus physiques et pertes.
- sampling.py : échantillonnage des points.
- train.py : entraînement.
- visualization.py : visualisation des pertes et des champs.
- notebooks/01_train.ipynb : entraînement dans Jupyter.
- notebooks/02_postprocess.ipynb : post-traitement.

## Installation

Python 3.10 ou supérieur est recommandé.

Créer un environnement virtuel :

    python -m venv .venv

Activer l'environnement sous Linux/macOS :

    source .venv/bin/activate

Sous Windows PowerShell :

    .venv\Scripts\Activate.ps1

Installer les dépendances :

    pip install -r requirements.txt

## Lancement

Depuis le répertoire du projet :

    python train.py

Les résultats sont écrits dans outputs/ :
- sloshing_pinn.pt
- loss_history.json

## Avertissements scientifiques

Ce projet est un prototype, pas un modèle validé quantitativement.

Les paramètres de géométrie, propriétés des fluides, mouvement
du réservoir et niveau initial doivent être vérifiés dans les
dictionnaires OpenFOAM réels.

Le mouvement supposé est :
x_tank(t) = A sin(omega*t + phase).

Le réseau prédit la vitesse relative dans le repère du réservoir.
La pression prédite est la pression mécanique p, pas directement
le champ OpenFOAM p_rgh.

Le signe de la force de tension superficielle doit être validé
sur un cas test. La fermeture k-epsilon est simplifiée et ne
reproduit pas toutes les corrections ni fonctions de paroi
d'une implémentation OpenFOAM.

Attention : le code impose actuellement u=v=0 sur les quatre
bords du rectangle, y compris le bord supérieur. Si le haut
du domaine est une ouverture, il faut modifier l'échantillonnage
et appliquer les conditions aux limites adaptées.

Les conditions initiales de k et epsilon ne sont pas encore
imposées explicitement. Les poids des pertes et leur mise à
l'échelle devront également être étudiés.

Avant une simulation complète, vérifier :
1. la géométrie et les patchs ;
2. l'amplitude, la fréquence et la phase du mouvement ;
3. les propriétés physiques ;
4. les conditions initiales et aux limites ;
5. le signe de la tension superficielle ;
6. la cohérence du repère et de la pression lors de la validation.