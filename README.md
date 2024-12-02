Nom du groupe: Le Groupe

Membre du groupe:
- Juliette Jacquot
- Lylian Cazale
- Matis Braun
- Virgile Hermant

# Travail restant

## Data Ingestion

Un `HDFS` a été ajouté au docker-compose pour lire le fichier `.csv`, il reste à s'assurer qu'après création du dataframe par spark, celui-ci soit accessible pour les autres parties.

## Data PreProcessing and Cleaning

La partie preprocessing a été effectuée dans le script pour l'analyse avec Spark, nous prévoyons de le remettre dans la bonne partie pour la semaine prochaine.

Aussi, la partie écriture du dataset propre dans HDFS pose encore problème, nous n'avons pas encore de moyen de le tester sans la partie intégration NOSQL

## DEA

Finir l'analyse en se penchant sur des différences dans des catégories spéciales.
Correctement finir la description de toutes les figures du notebook.

## SPARK DEA

Il reste à trouver un moyen d'enregistrer les clients ayant le plus dépensé.

Correctement finir la description de toutes les figures.

## ML ANALYSIS

Améliorer le LSTM

Correctement finir la description de toutes les figures (cluster et prédictions) obtenues dans un fichier markdown.

## OPTIONS

### CLOUD

Nous avons essayé de tout dockeriser, il reste encore à lier toutes les parties dans le container `app` afin que tout se lance le mieux possible

### NoSQL

Nous avons ajouté un `mongodb` qui tourne dans le docker compose, il manque à lier les scripts d'intégration des données dans l'application

# Utilisation

1. Mettre le fichier `ecommerce_data_with_trends.csv` dans le dossier `data/`
2. Lancer `docker compose up`

Pour lancer le partie ML, en attendant le raccord à l'application principale, vous pouvez lancer:
`python ML-Analysis/MLClustering_Analysis.py` ou `python ML-Analysis/Predicting_modeling.py` après avoir `uv sync` pour les dépendances python.
