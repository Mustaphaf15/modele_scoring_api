# API de Scoring Crédit

---

## Description

Cette API a été développée dans le cadre de la formation OpenClassrooms pour le projet "Implémentez un modèle de scoring".

L'objectif est de fournir une solution de "scoring crédit" pour la société financière "Prêt à dépenser", qui propose des crédits à la consommation pour des personnes ayant peu ou pas d'historique de prêt. L'outil calcule la probabilité qu'un client rembourse son crédit, puis classe la demande en crédit accordé ou refusé.

Cette API s'appuie sur un modèle de classification développé à partir des données issues de la compétition Kaggle Home Credit Default Risk.

## Prérequis

Python 3.11 ou version supérieure.
Dépendances spécifiées dans requirements.txt.

## Installation

Clonez ce dépôt : `bash  git clone <url_du_dépôt> cd <nom_du_dossier> `

Installez les dépendances : `bash  pip install -r requirements.txt `

Assurez-vous que les fichiers nécessaires sont présents :

application_test.csv : Données clients pour la prédiction.
model.pkl : Modèle pré-entraîné.

## Utilisation

Lancer l'API
Démarrez le serveur FastAPI : `bash  uvicorn app:app --reload `

Accédez à la documentation interactive Swagger à l'adresse : `bash  http://localhost:8000/docs `

## Endpoints

1. Accueil
   URL : GET /
   Description : Retourne un message de bienvenue.
2. Liste des IDs Clients
   URL : GET /list_id
   Description : Retourne une liste des IDs disponibles dans les données.
3. Prédiction pour un client
   URL : GET /predict/{client_id}
   Description : Effectue une prédiction pour un client spécifique.
   Paramètre requis :
   client_id : Identifiant unique du client (SK_ID_CURR).

## Déploiement

Cette API peut être déployée sur des plateformes comme Deta Space, en suivant ces étapes :

Installez l'outil Deta CLI : `bash  curl -fsSL https://get.deta.dev/cli.sh | sh `

Connectez-vous avec votre compte Deta : `bash  deta login `

Déployez le projet : `bash  deta deploy `

## Structure du Projet

app.py : Script principal contenant l'implémentation de l'API.
config.yaml : Fichier de configuration avec les chemins vers les fichiers nécessaires.
requirements.txt : Liste des dépendances Python.
