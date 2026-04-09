# 🏦 Projet 5 — Système RAG de Production sur Corpus Financier Hétérogène

## Architecture

Système RAG multi-agents pour l'analyse de rapports annuels bancaires et de documentation réglementaire.

```
Question utilisateur
        │
        ▼
┌─────────────────┐
│   GUARDS V2     │── Hors-sujet / Injection → Rejet
│  (regex + LLM)  │
└────────┬────────┘
         ▼
┌─────────────────┐
│  ORCHESTRATEUR   │
│  (routing LLM)   │
└──┬─────┬─────┬──┘
   │     │     │
   ▼     ▼     ▼
┌──────┐┌──────┐┌──────┐
│Agent ││Agent ││Agent │
│Rapp. ││Régl. ││Dir.  │
└──┬───┘└──┬───┘└──────┘
   │       │
   ▼       ▼
┌─────────────────┐
│  HYBRID SEARCH   │
│ BM25 + Embeddings│
│ + Fusion RRF     │
└────────┬────────┘
         ▼
┌─────────────────┐
│   ChromaDB       │
│ 2 collections    │
└─────────────────┘
```

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| LLM principal | Groq / `llama-3.3-70b-versatile` |
| LLM rapide (router, guards) | Groq / `llama-3.1-8b-instant` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB (in-memory) |
| Recherche hybride | BM25 + Embeddings + Reciprocal Rank Fusion |
| Extraction PDF | `pymupdf4llm` (PDF → Markdown structuré) |
| Framework | LangChain |
| UI | Streamlit |

## Déploiement sur Streamlit Cloud

### 1. Préparer les PDFs sur Google Drive

Uploadez vos PDFs sur Google Drive et récupérez l'ID de chaque fichier :
- Clic droit → "Obtenir le lien" → l'ID est la partie entre `/d/` et `/view`
- Ex : `https://drive.google.com/file/d/ABC123/view` → ID = `ABC123`
- **Important** : rendez chaque fichier accessible ("Tout le monde avec le lien")

### 2. Configurer les secrets Streamlit

Dans Streamlit Cloud → Settings → Secrets, ajoutez :

```toml
GROQ_API_KEY = "gsk_..."
GDRIVE_ID_ACPR = "id_du_fichier_acpr"
GDRIVE_ID_COMMUNIQUE = "id_du_communique_presse"
GDRIVE_ID_URD = "id_du_document_urd"
GDRIVE_ID_SFH = "id_du_fichier_sfh"
```

### 3. Déployer

1. Créez un repo GitHub avec ces fichiers
2. Allez sur [share.streamlit.io](https://share.streamlit.io)
3. Connectez le repo → sélectionnez `app.py`
4. Ajoutez les secrets (étape 2)
5. Deploy

## Corpus documentaire

| Document | Type | Collection |
|----------|------|------------|
| Rapport annuel ACPR 2024 | Réglementation | `reglementation` |
| Communiqué résultats annuels 2025 | Rapports | `rapports_annuels` |
| URD Crédit Mutuel Arkéa 2024 | Rapports | `rapports_annuels` |
| SFH RFA 2025 | Rapports | `rapports_annuels` |

## Benchmark

| Catégorie | Score routing | Latence moyenne |
|-----------|--------------|-----------------|
| Rapports financiers | 4/4 (100%) | ~1 345 ms |
| Réglementation | 2/3 (67%) | ~2 083 ms |
| Agent direct | 2/2 (100%) | ~1 672 ms |
| Sécurité (guards) | 3/3 (100%) | ~30 ms |
| **Global** | **11/12 (92%)** | **~1 255 ms** |
