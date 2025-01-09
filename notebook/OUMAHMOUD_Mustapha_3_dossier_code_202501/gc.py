import sys
import gc
import pandas as pd

# Fonction pour afficher les variables et leur taille mémoire dans un DataFrame trié
def get_memory_usage():
    # Créer une liste de tuples (nom de la variable, taille mémoire)
    memory_usage = [(name, sys.getsizeof(obj)) for name, obj in globals().items() if not name.startswith('_')]
    
    # Créer un DataFrame à partir de la liste
    df = pd.DataFrame(memory_usage, columns=['Variable', 'Taille Mémoire (bytes)'])
    
    # Trier le DataFrame par taille mémoire (du plus grand au plus petit)
    df = df.sort_values(by='Taille Mémoire (bytes)', ascending=False)
    
    return df

# Fonction pour supprimer des variables spécifiques et libérer la mémoire
def delete_variables_and_clear_memory(variables_to_delete):
    for var in variables_to_delete:
        if var in globals():
            print(f"Suppression de la variable : {var}")
            del globals()[var]
        else:
            print(f"Variable '{var}' non trouvée.")
    
    # Forcer le garbage collector à libérer la mémoire
    gc.collect()
    print("Mémoire libérée.")

# Fonction pour vider la RAM
def clear_ram(df, top_n=None):
    """
    Supprime les variables les plus gourmandes en mémoire à partir du DataFrame.
    
    :param df: DataFrame retourné par get_memory_usage().
    :param top_n: Nombre de variables à supprimer (par défaut, toutes les variables du DataFrame).
    """
    if top_n is not None:
        # Sélectionner les N variables les plus gourmandes
        variables_to_delete = df['Variable'].head(top_n).tolist()
    else:
        # Sélectionner toutes les variables du DataFrame
        variables_to_delete = df['Variable'].tolist()
    
    # Supprimer les variables et libérer la mémoire
    delete_variables_and_clear_memory(variables_to_delete)