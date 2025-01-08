from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import yaml
from joblib import load
import os

# Charger la configuration à partir du fichier config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

CSV_FILE = config["CSV_FILE"]
MODEL_FILE = config["MODEL_FILE"]
# Valeur par défaut 0.5 si non définie
OPTIMAL_THRESHOLD = config.get("OPTIMAL_THRESHOLD", 0.5)

# Charger le modèle avec joblib
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Le fichier modèle spécifié ({MODEL_FILE}) est introuvable.")

model = load(MODEL_FILE)


def create_new_features(df):
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    return df


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction ! créée dans le cadre de la formation openclassrooms, Implémentez un modèle de scoring"}


@app.get("/get_list_id")
def list_id():
    if not os.path.exists(CSV_FILE):
        raise HTTPException(
            status_code=404, detail="Le fichier CSV spécifié est introuvable.")

    df = pd.read_csv(CSV_FILE)
    if 'SK_ID_CURR' not in df.columns:
        raise HTTPException(
            status_code=400, detail="La colonne 'SK_ID_CURR' est absente du fichier CSV.")

    return {"SK_ID_CURR": df['SK_ID_CURR'].tolist()}


@app.get("/get_predict/{client_id}")
def predict(client_id: int):
    if not os.path.exists(CSV_FILE):
        raise HTTPException(
            status_code=404, detail="Le fichier CSV spécifié est introuvable.")

    df = pd.read_csv(CSV_FILE)
    df = create_new_features(df)
    if 'SK_ID_CURR' not in df.columns:
        raise HTTPException(
            status_code=400, detail="La colonne 'SK_ID_CURR' est absente du fichier CSV.")

    client_data = df[df['SK_ID_CURR'] == client_id]
    if client_data.empty:
        raise HTTPException(
            status_code=404, detail=f"Client avec ID {client_id} non trouvé.")

    # Préparer les données pour la prédiction en convertissant en DataFrame
    client_features = client_data.drop(columns=['SK_ID_CURR'])

    # Vérifier que toutes les features nécessaires sont présentes
    if not all(feature in client_features.columns for feature in model.feature_names_in_):
        missing_features = [
            feature for feature in model.feature_names_in_ if feature not in client_features.columns]
        raise HTTPException(
            status_code=400, detail=f"Les features suivantes sont manquantes: {', '.join(missing_features)}")

    # Effectuer la prédiction en utilisant les probabilités pour la classe positive
    # Probabilité de la classe positive
    pred_proba = model.predict_proba(client_features)[:, 1]
    predict = model.predict_proba(client_features)
    # Appliquer le seuil optimal pour déterminer la classe finale
    prediction = (pred_proba >= OPTIMAL_THRESHOLD).astype(int)
    print(predict)
    # Retourner la prédiction
    return {"SK_ID_CURR": client_id, "prediction": int(prediction[0]), "probability": pred_proba[0]}


@app.get("/get_client_info/{client_id}")
async def get_client_info(client_id: int):
    try:
        data = pd.read_csv(CSV_FILE)
        df = create_new_features(df)
        client_data = data[data["SK_ID_CURR"] == client_id]
        if client_data.empty:
            raise ValueError(
                f"Aucune donnée trouvée pour le client ID {client_id}.")
        return client_data.to_dict(orient="records")[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
