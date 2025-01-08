# Utiliser l'image Python officielle
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt pour installer les dépendances
COPY requirements.txt /app/requirements.txt

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier les fichiers nécessaires pour l'API
COPY app.py /app/app.py
COPY config.yaml /app/config.yaml

# Charger les fichiers spécifiés dans config.yaml
COPY data/application_test.csv /app/data/application_test.csv
COPY models/artifacts/best_model/model.pkl /app/models/artifacts/best_model/model.pkl

# Exposer le port par défaut de l'API
EXPOSE 8000

# Lancer l'application FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
