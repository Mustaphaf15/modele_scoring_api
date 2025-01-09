from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer
import numpy as np
import pandas as pd


def create_preprocessor(numerical_features, categorical_features):
    """
    Create a preprocessing pipeline for numerical and categorical features.

    Parameters:
    numerical_features: list of str
        List of numerical feature names.
    categorical_features: list of str
        List of categorical feature names.

    Returns:
    preprocessor: ColumnTransformer
        Preprocessing pipeline.
    """
    # Numerical transformer
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor



def build_pipeline(model, var_num, var_cat, param_grid=None, with_smote=True, custom_scorer=None):
    """
    Construire un pipeline de classification avec validation croisée, incluant le prétraitement.

    Paramètres :
    model : estimateur
        Le modèle de classification à utiliser.
    var_num : list de str
        Liste des variables numériques.
    var_cat : list de str
        Liste des variables catégoriques.
    param_grid : dict, par défaut=None
        Grille d'hyperparamètres pour GridSearchCV.
    with_smote : bool, par défaut=True
        Indique si SMOTE doit être utilisé pour le suréchantillonnage.
    custom_scorer : callable, par défaut=None
        Fonction personnalisée pour calculer un score de validation.

    Retourne :
    pipeline : Pipeline ou GridSearchCV
        Le pipeline final avec ou sans GridSearchCV.
    """
    # Create preprocessor
    preprocessor = create_preprocessor(var_num, var_cat)

    # Choose pipeline type based on SMOTE
    if with_smote:
        pipeline = make_imb_pipeline(
            SMOTE(random_state=42),
            preprocessor,
            model
        )
    else:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Return pipeline or GridSearchCV
    if param_grid:
        if custom_scorer:
            business_scorer = make_scorer(
                                    custom_scorer,
                                    greater_is_better=False,
                                    needs_proba=True
                                )
            pipeline = GridSearchCV(
                pipeline, 
                param_grid=param_grid, 
                scoring=business_scorer, 
                cv=cv, 
                n_jobs=-1
               ) 
        else:
            pipeline = GridSearchCV(
                pipeline, 
                param_grid=param_grid, 
                scoring='roc_auc', 
                cv=cv, 
                n_jobs=-1
               )

    return pipeline