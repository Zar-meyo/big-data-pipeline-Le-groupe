Nom du groupe: Le Groupe

Membre du groupe:
- Juliette Jacquot
- Lylian Cazale
- Matis Braun
- Virgile Hermant

# Travail effectué

## Data Ingestion

Problème au niveau de la lecture du `.csv`, il y a un problème de droit entre spark et docker que nous n'avons pas réussi à fixer. 
Nous passons donc par `pandas` pour lire le fichier avant de l'envoyer à spark.

# Utilisation

1. Mettre le fichier `ecommerce_data_with_trends.csv` dans le dossier `app/`
2. Lancer `docker compose up`
