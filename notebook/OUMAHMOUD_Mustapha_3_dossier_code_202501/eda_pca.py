import pandas as pd
import numpy as np
import seaborn as sns
import time
import warnings
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, make_scorer, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.pipeline import Pipeline

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_column', 122) # Afficher l'ensemble des colonnes
pd.set_option('display.max_row', 67) # Afficher l'ensemble des lignes

def display_pca_lasso_feature(df_train):
    var_cat = df_train.drop(columns=['TARGET'], axis=1).select_dtypes(include=['object']).columns.to_list()
    var_num = df_train.drop(columns=['TARGET'], axis=1).select_dtypes(exclude=['object']).columns.to_list()
    # Gestion des valeurs manquantes et encodage
    imputed_num = SimpleImputer(strategy='mean')
    X_num_imputed = imputed_num.fit_transform(df_train[var_num])
    
    # Standardisation des données numériques 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num_imputed)
    
    imputed_cat = SimpleImputer(strategy='most_frequent')
    X_cat_imputed = imputed_cat.fit_transform(df_train[var_cat])
    
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_cat_encoded = encoder.fit_transform(X_cat_imputed)
    
    # Concaténer les données numériques et catégorielles encodées
    X = np.concatenate([X_scaled, X_cat_encoded.toarray()], axis=1)
    
    # Calculer le nombre optimal de composantes PCA
    pca = PCA()
    pca.fit(X)
    
    # Visualiser la variance expliquée cumulée
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Nombre de composantes')
    plt.ylabel('Variance expliquée cumulée')
    plt.title('Choix du nombre de composantes PCA')
    plt.grid(True)
    plt.show()
    
    # Définir un seuil de variance expliquée (par exemple, 95%)
    explained_variance_threshold = 0.95
    
    # Déterminer le nombre de composantes nécessaire pour atteindre le seuil
    n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= explained_variance_threshold) + 1
    
    # Réduire la dimensionnalité à l'aide du nombre de composantes optimal
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    print("Nombre de composantes PCA retenues :", n_components)
    print("Variance expliquée totale :", np.sum(pca.explained_variance_ratio_))
    
    # LassoCV pour sélectionner les variables
    lasso = LassoCV(cv=5)
    y = df_train['TARGET']
    lasso.fit(X, y)  
    
    # Sélection des variables
    selector = SelectFromModel(lasso)
    selector.fit(X, y)
    X_lasso = selector.transform(X)
    
    print("Nombre de variables sélectionnées par Lasso :", X_lasso.shape[1])
    
    # Conciliation ACP et Lasso : différentes approches
    # Sélection des variables
    lasso_pca = LassoCV(cv=5)
    lasso_pca.fit(X_pca, y)
    
    # 1. Utiliser les composantes PCA comme entrées pour Lasso
    lasso_pca = LassoCV(cv=5)
    lasso_pca.fit(X_pca, y)
    
    # Sélection des variables
    selector_pca = SelectFromModel(lasso_pca)
    selector_pca.fit(X_pca, y)
    X_lasso_pca = selector_pca.transform(X_pca)
    
    print("Nombre de variables sélectionnées par Lasso (PCA):", X_lasso_pca.shape[1])

    # Étape 1: Obtenir les composantes principales et leur contribution
    # 'pca.components_' contient les poids des variables originales dans chaque composante principale
    # Chaque ligne représente une composante principale
    pca_components = pca.components_ 
    
    # Étape 2: Obtenir les variables originales contribuant fortement aux composantes retenues
    # Seuil pour considérer une variable comme importante dans une composante principale
    threshold = 0.1  # seuil
    
    # Variables importantes pour chaque composante principale retenue
    important_vars = []
    for i in range(pca.n_components):  # Pour chaque composante retenue
        component_weights = pca_components[i]
        important_indices = np.where(np.abs(component_weights) >= threshold)[0]  # Variables significatives
        important_vars.extend(important_indices)  # Ajouter les indices des variables originales
    
    # Supprimer les doublons pour obtenir les variables uniques
    important_vars = list(set(important_vars))
    
    # Étape 3: Mapper les indices des variables aux noms des colonnes d'origine
    original_features = var_num + encoder.get_feature_names_out(var_cat).tolist()
    important_features = [original_features[i] for i in important_vars]

    print("Variables importantes identifiées par PCA-Lasso :", important_features)
    print(len(important_features), 'variables des ', len(original_features), ' variables après encodage sont identifiées par PCA-Lasso comme variable importante')

    # Obtenir les indices des variables sélectionnées
    selected_features_indices = selector_pca.get_support(indices=True)
    # Obtenir les noms des variables sélectionnées
    selected_features = np.array(original_features)[selected_features_indices]
    # Créer un DataFrame pour visualiser les importances
    importance_df = pd.DataFrame({'feature': selected_features, 'importance': lasso_pca.coef_[selected_features_indices]})
    
    # Trier par ordre d'importance décroissante
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    # Visualiser les importances
    plt.figure(figsize=(12, 12))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Importance des variables sélectionnées par Lasso-PCA')
    plt.xlabel('Importance')
    plt.ylabel('Variable')
    plt.show()
    plt.close()