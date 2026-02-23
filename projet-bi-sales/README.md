# BI & Sales Performance Dashboard
# Business Intelligence & Sales Performance Dashboard

Ce répertoire documente la phase d'ingénierie des données (Data Engineering / ETL) et la construction d'un tableau de bord décisionnel basé sur le dataset public "Superstore Sales".

## Objectif
Transformer un ensemble de données brutes (fichiers plats) en un modèle de données relationnel structuré, afin d'alimenter un tableau de bord analytique centralisant les KPIs (CA, Marge, Panier Moyen, Croissance).

## Modélisation des Données (Star Schema)
Plutôt que d'interroger un fichier plat lourd, les données ont été normalisées en un **Modèle en Étoile** via Python (Pandas) :

* **`fact_sales`** : Table de faits contenant les identifiants uniques et les mesures quantitatives (Montant, Quantité, Remise, Profit).
* **`dim_customer`** : Dimension des clients (794 entrées uniques).
* **`dim_product`** : Dimension des produits (1863 entrées uniques).
* **`dim_location`** : Dimension géographique normalisée grâce à la création d'une clé primaire artificielle `location_id`.

## KPIs Définis
* **Chiffre d'Affaires Global (Revenue)**
* **Profit Total**
* **Panier Moyen (Average Order Value - AOV)**
* **Volume de Commandes**

## Structure du projet
* `01_Data_Model_Superstore.ipynb` : Notebook Python contenant le processus ETL (Extraction, Transformation, Modélisation).
* `dashboard_bi_d3.html` : L'interface visuelle du tableau de bord codée en D3.js.
* `projet-2.html` : La présentation métier détaillée de l'étude de cas (accessible depuis le portfolio).
