# Projet CRM — Mesure de l'Incrémentalité (A/B Test)

## Structure du dossier

```
projet-crm-retail/
├── projet-1.html              # Page projet
├── dashboard_d3.html          # Dashboard interactif D3.js
├── data_crm_incremental.json  # Données réelles du Z-Test
├── 01_Data_Preparation_CRM.ipynb  # Notebook de préparation
├── 02_Power_Analysis.py       # Analyse de puissance statistique
├── 03_Charts_Portfolio.py     # Visuels portfolio
├── chart-crm.png              # Graphique statique
├── power_analysis.png         # Graphique power analysis
└── README.md                  
```


### 1. Dashboard D3.js 
- Données réelles intégrées
- ROI illisibles remplacés par le **CA Incrémental Net** comme métrique principale
- Ajout d'un **waterfall chart** (décomposition CA base / uplift / coût)
- Barres hachurées pour signaler visuellement les résultats non significatifs
- KPI "Taille requise / cellule" issu de la power analysis
- Design modernisé (DM Sans, JetBrains Mono, palette slate)

### 2. Power Analysis 
- Graphique double panel : courbes N vs MDE + barres comparatives
- Intégré dans la page projet avec image et chiffres clés

### 3. Page projet réécrite
- Conclusions alignées avec la rigueur statistique
- "Non concluant" au lieu de "échec" — l'absence de preuve ≠ preuve d'absence
- Section Power Analysis intégrée (section 4)
- Recommandations concrètes (re-test Réguliers avec N ≥ 1 800)
- Limites enrichies (ratio témoin, attribution, coûts)

### 4. Visuels statiques améliorés
- Double panel : conversion brute + uplift avec p-values
- Barres hachurées + badge "Aucun résultat significatif"
- Palette professionnelle, export haute résolution (180 dpi)

## Données

- **Source** : [UCI Online Retail](https://archive.ics.uci.edu/ml/datasets/online+retail)
- **Assignation campagne** : simulée aléatoirement (seed 42)
- **Période campagne** : 01/05/2011 – 15/05/2011
