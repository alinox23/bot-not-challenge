# bot-not-challenge
Ce projet utilise l'apprentissage automatique (**Random Forest**) pour classifier les comptes utilisateur en tant qu'humains ou bots, en se basant sur des statistiques publiques (followers, activité, âge du compte).
## Installation

1. Clonez le dépôt :
   ```bash
   git clone [https://github.com/alinox23/bot-not-challenge.git](https://github.com/alinox23/bot-not-challenge.git)
   
2. Installez les dépendances:
    ```bash
   pip install -r requirements.txt
   

## Utilisation

1. Entraînement:
Lancez ```python main.py``` pour générer les données et entraîner le modèle.
2. Prédiction :
Utilisez ```python predict.py``` pour tester un compte spécifique.

## Modèle
Le modèle actuel est une Forêt Aléatoire atteignant une précision de 100% sur les données synthétiques.
    