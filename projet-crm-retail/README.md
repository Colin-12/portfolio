# CRM Analytics & Performance Commerciale
# Pilotage CRM Multicanal & Analyse d'Incrémentalité

Ce répertoire contient l'ensemble du code et des analyses d'un projet de Data Analytics appliqué au Marketing (CRM). 

* **Note de confidentialité :** Afin de respecter les accords de confidentialité (NDA) liés à mes expériences professionnelles, ce projet utilise un dataset public (UCI Online Retail). Cependant, l'architecture des données, la simulation des campagnes et les calculs d'incrémentalité reflètent exactement les méthodologies que j'applique en entreprise.*

## Objectif du Projet
Dépasser l'analyse classique des KPIs d'engagement (taux d'ouverture/clic) pour mesurer le **véritable revenu incrémental** généré par une campagne multicanale (Email vs SMS), en tenant compte de la pression commerciale et de la segmentation client.

## Méthodologie & Architecture des Données
Pour simuler un environnement Data Warehouse réaliste, le dataset brut a été transformé en un modèle relationnel robuste :

1. **`transactions` (Faits) :** Nettoyage des données réelles (exclusion des retours).
2. **`customers` (Dimension) :** Référentiel client unique.
3. **`rfm_features` (Segmentation) :** Calcul des scores de Récence, Fréquence et Montant calculés *strictement avant* le lancement de la campagne pour éviter toute fuite de données (Data Leakage). Classification en 3 segments : VIP, Réguliers, En Risque.
4. **`campaigns` & `campaign_exposure` (Mapping) :** Simulation d'un ciblage avec répartition stricte : 45% Email, 45% SMS, et **10% Groupe Témoin (Control Group)**.

## Insights Métier & Recommandations
L'analyse croisée des conversions réelles post-ciblage a révélé un insight majeur :
* **Effet de Cannibalisation :** Le groupe témoin du segment "VIP" a organiquement mieux converti (42.4%) que les VIP ayant reçu un Email (33.6%) ou un SMS (37.6%). Solliciter ce segment a donc généré un ROI négatif et une potentielle fatigue marketing.
* **Uplift positif :** À l'inverse, les campagnes ont généré un revenu incrémental net sur le segment "Régulier".
* **Recommandation :** Stopper la pression promotionnelle sur les VIP et réallouer le budget CRM (prioritairement sur l'Email, moins coûteux) vers le segment des acheteurs réguliers.

## Stack Technique
* **Préparation des données & Analyse :** Python (Pandas, Numpy) via Jupyter Notebook / Google Colab.
* **Data Visualization (Exploratoire) :** Matplotlib, Seaborn.
* **Dashboarding Interactif :** D3.js (HTML/CSS/JS) avec filtres croisés dynamiques.

## Structure des fichiers
* `01_Data_Preparation_CRM.ipynb` : Le notebook contenant l'ETL, la logique de ciblage, l'A/B test et l'analyse ROI.
* `dashboard_d3.html` : Le tableau de bord interactif codé en D3.js (accessible via la page projet).
* `projet-1.html` : La page web de présentation de l'étude de cas (intégrée au portfolio principal).
* `chart-crm.png` : Export statique de la visualisation principale.
