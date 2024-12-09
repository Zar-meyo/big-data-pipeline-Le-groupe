Nom du groupe: Le Groupe  
  
Membre du groupe:  
- Juliette Jacquot  
- Lylian Cazale  
- Matis Braun  
- Virgile Hermant  
  
# Travail effectué  
  
## Data Ingestion  
  
L'application lit le fichier `.csv` afin de créer le dataframe spark.
  
## Data PreProcessing and Cleaning  

Pour le pré traitement du dataset, nous retirons les valeurs `null` et les dupliquées.
On extrait les catégories principales et les sous-catégories des produits.
Et enfin on extrait les jours de semaine et les mois des achats.
  
## DEA  
Pour lancer le notebook, il faut que le dataset `ecommerce_data_with_trends.csv` soit présent dans le soddier app en dehors du dossier de cette partie.  
  
Plusieurs analyses ont été faite pour cette partie, une analyse par type de client sur plusieurs variables, des analyses temporelles en fonction des jours et des mois, ainsi qu'une analyse par temps et par catégories.   
  
## SPARK DEA  
  
Le rapport se situe dans le fichier `analysis/spark_data_analysis.ipynb`. Il explique tous les graphiques générés par la partie `spark analysis` de l'app.

En résumé, on analyse les achats par type de clients (B2B vs B2C) et ensuite les types de produits achetés selon le mois et le jour de la semaine.
On repasse par `pandas` pour enregistrer le `.csv` détaillant les clients ayant le plus dépensés.

## ML ANALYSIS  
  
 Cette partie se découpe en deux étapes:
 - Le cluster analysis: on fait trois clusters qui utilise différentes features pour réaliser des études dessus. 
 - Prédiction: on réalise deux modèles de prédictions différents. Le premier, un arbre de décision, prédit si un customer va faire un achat dans la prochaine heure. Le second, un LSTM, prédit le montant total dépensé par chaque client dans le cas d'un prochain achat.
  Un rapport expliquant les graphiques générés se trouve dans `ML-analysis/ML_Report.ipynb`.
  
# Utilisation  
  
1. Mettre le fichier `ecommerce_data_with_trends.csv` dans le dossier `/app`  
2. Lancer `docker compose up` 
