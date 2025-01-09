import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import networkx as nx
import warnings
import gc
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_column', 122) # Afficher l'ensemble des colonnes
pd.set_option('display.max_row', 67) # Afficher l'ensemble des lignes



path_train = '../data/application_train.csv'
path_test = '../data/application_test.csv'
path_data_description = '../data/HomeCredit_columns_description.csv'
path_png_data = '../data/home_credit.png'
df_train = pd.read_csv(path_train)
df_test =  pd.read_csv(path_test)
df_description = pd.read_csv(path_data_description, encoding='latin')
df_description = df_description[df_description['Table'] == 'application_{train|test}.csv'][['Row', 'Description', 'Special']].set_index('Row')


def display_head_train(nb_lignes):
    return df_train.head(nb_lignes)

    
def display_head_test(nb_lignes):
    return df_test.head(nb_lignes)

    
def show_table_schemas():
    # Charger l'image
    img = mpimg.imread(path_png_data) 
    # Afficher l'image
    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    plt.close()


def print_shape_tarin():
    print(f'Nombre de lignes et colonnes dans application_train.csv : {df_train.shape}')

def print_shape_test():
    df =  pd.read_csv(path_train)
    print(f'Nombre de lignes et colonnes dans application_test.csv : {df_test.shape}')

def column_comparison():
    if(set(df_train.columns) -  set(df_test.columns)):
        print(f'Les variables suivantes sont dans application_train.csv mais ne se trouvent pas dans application_test.csv:{set(df_train.columns) -  set(df_test.columns)}')
    else:
        print("L'ensemble des variables dans application_train.csv sont aussi dans application_test.csv")
        
    if(set(df_test.columns) - set(df_train.columns)):
        print(f'Les variables suivantes sont dans application_test.csv mais ne se trouvent pas dans application_train.csv:{set(df_train.columns) -  set(df_test.columns)}')
    else:
        print("L'ensemble des variables dans application_test.csv sont aussi dans application_train.csv ")


def variable_types_train():
    return df_train.dtypes.value_counts()


def variable_types_test():
    return df_test.dtypes.value_counts()

def display_data_duplicated():
    global df_train
    global df_test
    df = pd.concat([df_train.drop(columns=['TARGET'], axis=1), df_test], )
    print("Nombre de lignes dupliquées dans les deux eux de données:", 
          df.duplicated(keep=False).sum())
    del df
    gc.collect
    
def delete_SK_ID_CURR():
    global df_train
    df_train = df_train.drop(columns=['SK_ID_CURR'], axis= 1)

    
def missing_values_table():
    global df_train
    global df_description
    # Total des valeurs manquantes
    mis_val = df_train.isnull().sum()
    # Pourcentage de valeurs manquantes
    mis_val_percent = 100 * df_train.isnull().sum() / len(df_train)
    # Faire un tableau avec les résultats
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    # Renommer les colonnes
    mis_val_table = mis_val_table.rename(
    columns = {0 : 'Valeurs manquantes', 1 : '% des valeurs totales'})

    # Trier le tableau par pourcentage décroissant
    mis_val_table = mis_val_table[
    mis_val_table.iloc[:,1] != 0].sort_values(
    '% des valeurs totales', ascending=False).round(1)

    # Afficher les informations récapitulatives
    print ("Le jeu de données contient " + str(df_train.shape[1]) + " colonnes.\n"
            + str(mis_val_table.shape[0]) +
              " colonnes contiennent des valeurs manquantes.")
    mis_val_table = pd.merge(mis_val_table, df_description, left_index=True, right_index=True, how='left')
    # Renvoyer le dataframe avec les informations manquantes
    return mis_val_table


def diplay_target():
    global df_train
    print(df_train['TARGET'].value_counts(normalize=True))
    plt.figure(figsize=(12, 6))
    sns.countplot(x='TARGET', data=df_train, hue="TARGET")
    plt.title('Distribution de la variable TARGET')
    plt.xlabel('Target (0: Pas de difficultés de payement , 1: Difficultés de payement)')
    plt.ylabel('Nombre')
    plt.show()
    plt.close()


def display_qual_var(missing_values):
    global df_train
    global df_test
    global df_description
    var_cat = df_train.drop(columns=['TARGET'], axis=1).select_dtypes(include=['object']).columns.to_list()
    print(f'''Notre jeu de données train contient {len(var_cat)} variables qualitatives:''')
    for var in var_cat:
        # Préparer les données pour l'affichage
        data = df_train[var].value_counts().reset_index()
        data.columns = [var, 'Nombre']  # Renommer les colonnes pour clarté
        # Calcul des pourcentages
        data['Pourcentage'] = (data['Nombre'] / data['Nombre'].sum()) * 100
        plt.figure(figsize=(12, 6))  # Définir la taille de la figure
        # Créer un graphique en barres horizontales avec Seaborn
        barplot = sns.barplot(
            data=data, 
            y='Nombre', 
            x=var, 
            hue=var  # Choix d'une palette de couleurs
        )
        # Ajout des étiquettes de pourcentage sur les barres
        for index, row in data.iterrows():
            barplot.text(index, row['Nombre'], f'{row["Pourcentage"]:.1f}%', color='black', ha="center")

        # Ajouter un titre et des labels
        plt.title(f'Distribution de la variable {var}')
        plt.xlabel(var)
        plt.ylabel('Nombre')


        if len(df_train[var].unique()) > 7:
            plt.xticks(rotation = 90)
        
        # Afficher le graphique
        plt.show()
        plt.close()
        
        print(f'{var :-<55} {len(df_train[var].unique())} valeurs unqiues', ' et ' + \
              str(missing_values.loc[var, '% des valeurs totales']) + ' % de valeur manquantes' \
              if var in missing_values.index else '', f' :\n {df_train[var].unique()}')
        not_in_test = set(df_train[var].unique()) - set(df_test[var].unique())
        if not_in_test:
            print(f'les valeurs suivantes ne sont pas dans le test: {not_in_test}', '\n')
        else:
            print('\n')
        not_in_train = set(df_test[var].unique()) - set(df_train[var].unique())
        if not_in_train:
            print(f'les valeurs suivantes sont dans le test mais pas dans le train: {not_in_test}', '\n')
        else:
            print('\n')
        print(f"Description de la variable {var}: {df_description.loc[var, 'Description']}")
        print(data)
        print('\n'*2)


def show_outlier_qual_var():
    global df_train
    print("Les valeurs suivantes 'XNA' dans la variable CODE_GENDER, 'Maternity leave' dans NAME_INCOME_TYPE, 'Unknown' dans NAME_FAMILY_STATUS remontent dans le jeu de données train et ne sont pas dans le test, Nombre d'individues concernés :")
    print(df_train[(df_train['CODE_GENDER'] == 'XNA') | (df_train['NAME_INCOME_TYPE'] == 'Maternity leave') | \
          (df_train['NAME_FAMILY_STATUS'] == 'Unknown')].shape[0], ' pour un total de ', df_train.shape[0])
    print(f"Cela représente uniquement {df_train[(df_train['CODE_GENDER'] == 'XNA') | (df_train['NAME_INCOME_TYPE'] == 'Maternity leave') | (df_train['NAME_FAMILY_STATUS'] == 'Unknown')].shape[0]/ df_train.shape[0] * 100:.3f}%")
    print("Nous pouvons supprimer les 11 individues")
    df_train = df_train[(df_train['CODE_GENDER'] != 'XNA') & (df_train['NAME_INCOME_TYPE'] != 'Maternity leave') & (df_train['NAME_FAMILY_STATUS'] != 'Unknown')]
    print_shape_tarin()


def show_describe_var_num(var=None):
    global df_train
    var_num = df_train.drop(columns=['TARGET'], axis=1).select_dtypes(exclude=['object']).columns.to_list()
    if var:
        return df_train[var].describe()
    else:
        return df_train[var_num].describe()


def show_binary_cols(missing_values):
    global df_train
    global df_description
    var_num = df_train.drop(columns=['TARGET'], axis=1).select_dtypes(exclude=['object']).columns.to_list()
    binary_cols = [
        col for col in var_num 
        if set(df_train[col].unique()) <= {0, 1}
    ]
    print(len(binary_cols), 'des ', len(var_num), ' variables numériques.')
    for var in binary_cols:
        print(var, ': ', df_description.loc[var, 'Description'])
        if var in missing_values.index:
            print(str(missing_values.loc[var, '% des valeurs totales']) + ' % de valeur manquantes')


def show_num_cols_with_neg(missing_values):
    global df_train
    global df_description
    var_num = df_train.drop(columns=['TARGET'], axis=1).select_dtypes(exclude=['object']).columns.to_list()
    # Identifier les colonnes numériques avec des valeurs négatives
    num_cols_with_neg = df_train[var_num].columns[(df_train[var_num] < 0).any()].tolist()
    print('Les variables suivantes du jeu de données contienent des valeurs négatives :', num_cols_with_neg)
    print(len(num_cols_with_neg), ' variables des ', len(var_num), 'variabes numériques')
    for var in num_cols_with_neg:
        plt.figure(figsize=(12, 6))  # Configuration de la taille de la figure
        # Afficher la distribution
        sns.histplot(df_train[var], kde=True, bins=30)
        # Ajouter des titres et labels
        plt.title(f'Distribution de la variable {var} (avec des valeurs négatives)')
        plt.xlabel(var)
        plt.ylabel('Fréquence')
        # Afficher le graphique
        plt.show()
        plt.close()
        print(var, ': ', df_description.loc[var, 'Description'])
        if var in missing_values.index:
            print(str(missing_values.loc[var, '% des valeurs totales']) + ' % de valeur manquantes')
        print('\n'*2)

    
def transform_days_employed():
    global df_train
    var = 'DAYS_EMPLOYED'
    df_train[var] = df_train[var].where(df_train[var] <= 0, np.nan)
    plt.figure(figsize=(12, 6))  # Configuration de la taille de la figure
    # Afficher la distribution
    sns.histplot(df_train[var], kde=True, bins=30)
    # Ajouter des titres et labels
    plt.title(f'Distribution de la variable {var} (avec des valeurs négatives)')
    plt.xlabel(var)
    plt.ylabel('Fréquence')
    # Afficher le graphique
    plt.show()
    plt.close()
    plt.close()



def transform_negative_values_global():
    """
    Modifie directement le DataFrame global `df_train` :
    1. Sélectionne les colonnes contenant des valeurs négatives.
    2. Remplace les valeurs positives par NaN.
    3. Divise les valeurs négatives (non nulles) par -365.
    """
    global df_train 
    var_num = df_train.drop(columns=['TARGET'], axis=1).select_dtypes(exclude=['object']).columns.to_list()
    # Sélectionner les colonnes contenant des valeurs négatives
    cols_with_negatives = df_train[var_num].columns[(df_train[var_num] < 0).any()].tolist()
    
    # Appliquer les transformations sur les colonnes sélectionnées
    for col in cols_with_negatives:
        # Remplacer les valeurs positives par NaN
        df_train[col] = np.where(df_train[col] >= 0, np.nan, df_train[col])
        
        # Diviser les valeurs négatives (et non nulles) par -365
        df_train[col] = np.where(df_train[col].notna(), df_train[col] / -365, df_train[col])
        


def find_outliers(df, col):
    Q1 = df[col].quantile(0.25)  # Premier quartile
    Q3 = df[col].quantile(0.75)  # Troisième quartile
    IQR = Q3 - Q1               # Intervalle interquartile
    lower_bound = Q1 - 1.5 * IQR  # Limite inférieure
    upper_bound = Q3 + 1.5 * IQR  # Limite supérieure
    
    # Retourne True si des outliers sont présents
    return (df[col] < lower_bound) | (df[col] > upper_bound)


def list_cols_with_outliers():
    global df_train
    var_num = df_train.drop(columns=['TARGET'], axis=1).select_dtypes(exclude=['object']).columns.to_list()
    cols_with_outliers = [col for col in var_num if find_outliers(df_train, col).any()]
    binary_cols = [
        col for col in cols_with_outliers 
        if set(df_train[col].unique()) <= {0, 1}
    ]
    # Exclure les colonnes binaires des colonnes avec outliers
    cols_with_outliers = [col for col in cols_with_outliers if col not in binary_cols]
    print(len(cols_with_outliers), ' Varriables numériques (sans les variables binaires) ont des valeurs aberrantes:')
    for col in cols_with_outliers:
        plt.figure(figsize=(10, 6))
        
        # Distribution avec Seaborn
        sns.boxplot(data=df_train, x=col)
        
        # Ajouter des titres et labels
        plt.title(f'Boxplot de la variable {col} (avec des valeurs aberrantes)')
        plt.xlabel(col)
        plt.show()
        plt.close()
    
        # Calculer le nombre de valeurs aberrantes
        outliers = find_outliers(df_train, col)
        num_outliers = outliers.sum()
        percent_outliers = (num_outliers / len(df_train)) * 100
        
        
        # Afficher le nombre de valeurs aberrantes
        print(f"Variable : {col}, Nombre de valeurs aberrantes : {num_outliers}, soit : {percent_outliers:.2f}%")
        print('\n'*2)
        
    return cols_with_outliers


def remplacer_valeurs_aberrantes_par_nan(cols_with_outliers, seuil_iqr=1.5):
    """
    Remplace les valeurs aberrantes dans les colonnes numériques d'un DataFrame par NaN.

    Paramètres:
    -----------
    df : pandas.DataFrame
        Le DataFrame contenant les données.
    seuil_iqr : float, optional (par défaut 1.5)
        Le seuil pour l'écart interquartile (IQR). Les valeurs en dehors de
        [Q1 - seuil_iqr * IQR, Q3 + seuil_iqr * IQR] sont considérées comme aberrantes.

    Retour:
    -------
    pandas.DataFrame
        Le DataFrame avec les valeurs aberrantes remplacées par NaN.
    """
    global df_train
    # Sélectionner uniquement les colonnes numériques
    var_num = df_train.drop(columns=['TARGET'], axis=1).select_dtypes(exclude=['object']).columns.to_list()
    binary_cols = [
        col for col in var_num 
        if set(df_train[col].unique()) <= {0, 1}
    ]

    # Exclure les colonnes binaires des colonnes numériques
    var_num = [col for col in var_num if col not in binary_cols]
    
    # Parcourir chaque colonne numérique
    for col in cols_with_outliers:
        # Calculer les quartiles et l'IQR
        Q1 = df_train[col].quantile(0.25)
        Q3 = df_train[col].quantile(0.75)
        IQR = Q3 - Q1

        # Définir les bornes pour les valeurs aberrantes
        lower_bound = Q1 - seuil_iqr * IQR
        upper_bound = Q3 + seuil_iqr * IQR

        # Remplacer les valeurs aberrantes par NaN
        df_train[col] = np.where(
                (
                    df_train[col] < lower_bound
                ) | (
                    df_train[col] > upper_bound
                ), 
                np.nan, df_train[col])
        
        plt.figure(figsize=(10, 6))
        # Distribution avec Seaborn
        sns.boxplot(data=df_train, x=col)
        # Ajouter des titres et labels
        plt.title(f'Boxplot de la variable {col} (avec des valeurs aberrantes)')
        plt.xlabel(col)
        plt.show()
        plt.close()



def display_corr():
    global df_train
    # Sélectionner uniquement les colonnes numériques
    var_num = df_train.drop(columns=['TARGET'], axis=1).select_dtypes(exclude=['object']).columns.to_list()
    # Filtrer les colonnes numériques
    num_cols = df_train[var_num + ['TARGET']].columns
    
    # Calculer la matrice de corrélation
    correlation_matrix = df_train[num_cols].corr()
    
    # Configurer la taille de la heatmap
    plt.figure(figsize=(12, 10))
    
    # Tracer la heatmap
    sns.heatmap(
        correlation_matrix, 
        annot=False,      # Vous pouvez passer à True pour afficher les valeurs
        cmap="coolwarm",  # Palette de couleurs
        linewidths=0.5,   # Taille des séparateurs entre les cellules
        cbar=True,        # Afficher la barre de couleur
        fmt=".2f"         # Format des valeurs si annot=True
    )
    
    # Ajouter un titre
    plt.title("Heatmap des corrélations entre les variables numériques")
    plt.show()
    plt.close()


def display_corr_target():
    global df_train
    # Sélectionner uniquement les colonnes numériques
    var_num = df_train.drop(columns=['TARGET'], axis=1).select_dtypes(exclude=['object']).columns.to_list()
    
    # Calculer les corrélations avec la variable TARGET
    correlations = df_train[var_num + ['TARGET']].corr()['TARGET'].sort_values(ascending=False)
    
    # Filtrer uniquement les variables numériques
    num_corr = correlations[df_train[var_num].columns]
    
    # Afficher les variables corrélées avec TARGET (positives et négatives)
    print("Corrélations des variables numériques avec TARGET :")
    print(num_corr)
    
    # Visualisation des corrélations significatives
    threshold = 0.1  # Par exemple, pour un seuil de corrélation absolue > 0.1
    significant_corr = num_corr[abs(num_corr) > threshold]
    
    print("\nVariables ayant une corrélation significative avec TARGET (>|0.1|) :")
    print(significant_corr)
    
    # Tracer un barplot des corrélations significatives
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=significant_corr.index, 
        y=significant_corr.values, 
        hue=significant_corr.index
    )
    plt.xticks(rotation=90)
    plt.title("Corrélation des variables numériques significatives avec TARGET")
    plt.xlabel("Variables")
    plt.ylabel("Coefficient de corrélation")
    plt.show()
    plt.close()
    
def display_strong_corr():
    global df_train
    
    # Sélectionner uniquement les colonnes numériques
    var_num = df_train.drop(columns=['TARGET'], axis=1).select_dtypes(exclude=['object']).columns.to_list()
    # Calculer les corrélations avec la variable TARGET
    corr_matrix = df_train[var_num].corr()
    
    # Filtrer les corrélations fortes (par exemple, > 0.8 ou < -0.8)
    strong_corr = corr_matrix[abs(corr_matrix) > 0.8]
    
    # Masquer la moitié supérieure de la heatmap pour éviter la redondance
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Créer une heatmap personnalisée
    f, ax = plt.subplots(figsize=(12, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    #sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matrice de corrélation (moitié inférieure)')
    plt.show()
    plt.close()


def create_new_features():
    global df_train
    df_train['DAYS_EMPLOYED_PERC'] = df_train['DAYS_EMPLOYED'] / df_train['DAYS_BIRTH']
    df_train['INCOME_CREDIT_PERC'] = df_train['AMT_INCOME_TOTAL'] / df_train['AMT_CREDIT']
    df_train['INCOME_PER_PERSON'] = df_train['AMT_INCOME_TOTAL'] / df_train['CNT_FAM_MEMBERS']
    df_train['ANNUITY_INCOME_PERC'] = df_train['AMT_ANNUITY'] / df_train['AMT_INCOME_TOTAL']
    df_train['PAYMENT_RATE'] = df_train['AMT_ANNUITY'] / df_train['AMT_CREDIT']
    plt.figure(figsize = (12, 20))
    print('Visualiser les nouvelles variables:')
    for i, feature in enumerate([
        'DAYS_EMPLOYED_PERC',
        'INCOME_CREDIT_PERC',
        'INCOME_PER_PERSON',
        'ANNUITY_INCOME_PERC',
        'PAYMENT_RATE'
    ]):
        
        # create a new subplot for each source
        plt.subplot(5, 1, i + 1)
        # plot repaid loans
        sns.kdeplot(df_train.loc[df_train['TARGET'] == 0, feature], label = 'target == 0')
        # plot loans that were not repaid
        sns.kdeplot(df_train.loc[df_train['TARGET'] == 1, feature], label = 'target == 1')
        
        # Label the plots
        plt.title('Distribution of %s by Target Value' % feature)
        plt.xlabel('%s' % feature); plt.ylabel('Density');
        
    plt.tight_layout(h_pad = 2.5)
    return df_train