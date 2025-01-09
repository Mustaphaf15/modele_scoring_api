import streamlit as st
import requests
import pandas as pd
import numpy as np

# API Base URL
API_BASE_URL = "https://modele-scoring-api.onrender.com"

# Streamlit App
st.set_page_config(page_title="Crédit Scoring", layout="wide")

st.title("Application de Scoring Crédit")

st.write(
    "Bienvenue dans l'application de scoring crédit ! Cette application calcule la probabilité qu'un client rembourse son crédit et décide si le crédit est accordé ou refusé."
)
# Fetch client IDs
st.sidebar.header("Menu")
client_ids_response = requests.get(f"{API_BASE_URL}/get_list_id")

if client_ids_response.status_code == 200:
    client_ids = client_ids_response.json().get("SK_ID_CURR", [])
    if client_ids:
        selected_client_id = \
            st.sidebar.selectbox(
                "Sélectionnez un client",
                client_ids,
                index=None
                )

        # Fetch client info
        if selected_client_id:
            st.subheader("Informations du client")
            client_info_response = requests.get(
                f"{API_BASE_URL}/get_client_info/{selected_client_id}"
            )

            if client_info_response.status_code == 200:
                client_info = client_info_response.json()
                client_info_df = pd.DataFrame([client_info])
                client_info_df = client_info_df.T.reset_index()
                client_info_df.columns = client_info_df.iloc[0]
                client_info_df = client_info_df[1:].reset_index(drop=True)

                st.dataframe(client_info_df,
                             use_container_width=True,
                             hide_index=True)

                # Fetch prediction
                st.subheader("Prédiction de remboursement")
                prediction_response = requests.get(
                    f"{API_BASE_URL}/get_predict/{selected_client_id}"
                )

                if prediction_response.status_code == 200:
                    prediction_data = prediction_response.json()

                    prediction = prediction_data.get("prediction", "N/A")
                    probability = prediction_data.get("probability", "N/A")

                    if prediction == 1:
                        st.success(
                            f"Crédit ACCORDÉ avec une probabilité de remboursement de {probability:.2%}"
                        )
                    else:
                        st.error(
                            f"Crédit REFUSÉ avec une probabilité de remboursement de {probability:.2%}"
                        )

                else:
                    st.error(
                        f"Erreur lors de la récupération de la prédiction : {prediction_response.text}"
                    )
            else:
                st.error(
                    f"Erreur lors de la récupération des informations du client : {client_info_response.text}"
                )
    else:
        st.error("Aucun ID client trouvé.")
else:
    st.error(f"Erreur lors de la récupération des IDs client : {client_ids_response.text}")
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.info("l'instance s'arrêtera en cas d'inactivité, ce qui peut retarder les requêtes de 50 secondes ou plus!")