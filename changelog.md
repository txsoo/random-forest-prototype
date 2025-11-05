# Hit Identificator - Journal de Développement

## 2025-09-19 - Fiabilisation entrainement et preparation

### Modifications
- Ajout d'un hash dataset ordonne et des listes `inchikeys`/`scaffold_labels` dans `chembl_dataset_preparation.py`; enrichissement des exports de splits avec `dataset_hash` et verifications renforcees (`verify_outputs`).
- `random_forest_training.py`: controle d'integrite dataset/split, seeds globaux, calibration isotonic via hold-out, `oob_score`, optimisation de seuil (MCC) et nouvelles metriques (balanced accuracy, MCC, class balance, snapshots).
- Sauvegarde dans `models/metrics.json` de l'instantane dataset, des versions, du contexte calibration/split avec lissage EMA des metriques numeriques.

### Impact
- Detection immediate des divergences entre donnees et splits et meilleure reproductibilite.
- Probabilites calibrees et seuil optimal -> meilleure articulation rappel/specifite selon les besoins produit.
- Historique des performances enrichi et contextualise pour suivre la stabilite du modele.

## 📅 2025-09-13 - Étape 1 : Côté Scientifique (TERMINÉE)

### ✅ Réalisations

#### 1. Configuration du Projet
- **`requirements.txt`** créé avec toutes les dépendances nécessaires :
  - `rdkit` pour la chémoinformatique
  - `chembl-webresource-client` pour l'accès aux données ChEMBL
  - `scikit-learn` pour le machine learning
  - `pandas`, `numpy` pour la manipulation de données
  - `matplotlib`, `seaborn` pour les visualisations
  - Support ONNX pour l'export de modèles

#### 2. Préparation du Dataset ChEMBL
- **`chembl_dataset_preparation.py`** développé avec classe `ChEMBLDatasetPreparator`
- **Cible sélectionnée** : CHEMBL279 (Cyclooxygenase-2 / COX-2)
- **Pipeline de données** :
  - Téléchargement automatique depuis ChEMBL API
  - Filtrage des données IC50 (0.1 nM - 100 μM)
  - Conversion en pIC50 et classification binaire (seuil = 6)
  - Récupération des structures SMILES
  - Calcul des descripteurs moléculaires

#### 3. Descripteurs Moléculaires
- **Morgan Fingerprints** (ECFP4) : 2048 bits, rayon=2
- **Descripteurs RDKit** :
  - Poids moléculaire
  - LogP (lipophilie)
  - Donneurs/accepteurs de liaisons H
  - Liaisons rotables
  - Surface polaire topologique (TPSA)

#### 4. Modèle Random Forest
- **`random_forest_training.py`** avec classe `RandomForestTrainer`
- **Fonctionnalités** :
  - Préprocessing automatique (split stratifié, normalisation)
  - Optimisation hyperparamètres (Grid Search optionnel)
  - Entraînement Random Forest optimisé pour chémoinformatique
  - Évaluation complète (ROC AUC, Precision-Recall, Feature Importance)
  - Visualisations automatiques des performances

#### 5. Export et Sauvegarde
- **Formats multiples** :
  - Pickle (`.pkl`) - format natif Python
  - Joblib (`.joblib`) - optimisé scikit-learn
  - ONNX (`.onnx`) - format portable inter-plateformes
- **Métadonnées** complètes sauvegardées
- **Scaler** sauvegardé pour normalisation cohérente

#### 6. Notebook de Démonstration
- **`hit_identification_demo.ipynb`** créé
- **Contenu** :
  - Pipeline complet interactif
  - Exploration et visualisation des données
  - Évaluation détaillée du modèle
  - Test de prédiction sur molécules connues (Aspirine, Ibuprofène, etc.)
  - Analyse des features importantes

### 📊 Résultats Attendus
- **Dataset** : ~1500-2000 composés après filtrage
- **Features** : ~2054 (2048 Morgan bits + 6 descripteurs)
- **Performance attendue** : ROC AUC > 0.8 pour COX-2
- **Classes** : Actifs (~20-30%) vs Inactifs

### 🔧 Architecture Technique
```
chembl_dataset_preparation.py
├── ChEMBLDatasetPreparator
│   ├── download_bioactivity_data()
│   ├── get_compound_data()
│   ├── clean_bioactivity_data()
│   ├── calculate_molecular_descriptors()
│   └── prepare_dataset()

random_forest_training.py
├── RandomForestTrainer
│   ├── load_dataset()
│   ├── preprocess_data()
│   ├── hyperparameter_tuning()
│   ├── train_model()
│   ├── evaluate_model()
│   └── save_model()
```

### 📁 Structure des Fichiers
```
Hit identificator/
├── chembl_dataset_preparation.py
├── random_forest_training.py
├── hit_identification_demo.ipynb
├── requirements.txt
├── data/                          # Généré après exécution
│   ├── X_features.npy
│   ├── y_labels.npy
│   ├── dataset_info.pkl
│   └── chembl_dataset_full.csv
└── models/                        # Généré après entraînement
    ├── random_forest_model.pkl
    ├── random_forest_model.joblib
    ├── random_forest_model.onnx
    ├── scaler.pkl
    ├── model_metadata.pkl
    └── plots/
        ├── evaluation_metrics.png
        └── feature_importance.png
```

### 🎯 Prochaines Étapes (Étape 2-4)
- [ ] **Infrastructure Compute** : Client Docker pour compute provider
- [ ] **Smart Contract** : Gestion décentralisée des jobs ML
- [ ] **Frontend** : Interface web React/Next.js
- [ ] **Intégration** : Pipeline complet bout-en-bout

---

## 📝 Notes Techniques

### Choix de COX-2 comme Cible
- Cible bien documentée dans ChEMBL
- Données IC50 abondantes et de qualité
- Pertinence pharmaceutique (anti-inflammatoires)
- Bon équilibre actifs/inactifs pour classification

### Optimisations Implémentées
- **Traitement par batch** pour éviter timeouts API
- **Gestion d'erreurs** robuste pour SMILES invalides
- **Normalisation** des features pour améliorer performances
- **Validation croisée** pour sélection hyperparamètres
- **Export multi-format** pour compatibilité maximale

### Métriques de Performance
- **ROC AUC** : Mesure globale de discrimination
- **Precision-Recall** : Important pour classes déséquilibrées
- **Feature Importance** : Interprétabilité du modèle
- **Confusion Matrix** : Analyse détaillée des erreurs

---

## 🛠️ 2025-09-14 - Correctif : Nettoyage robuste des bioactivités

### Problème
- TypeError dans `chembl_dataset_preparation.py` > `clean_bioactivity_data()` lors du filtrage de `standard_value` (comparaison str/float).

### Cause racine
- `standard_value` contenait des chaînes et des unités hétérogènes (`standard_units`) non normalisées, entraînant des comparaisons directes str vs float.

### Correctif
- Conversion de `standard_value` en numérique via `pd.to_numeric(..., errors='coerce')`.
- Normalisation des `standard_units` (pm, nm, µm/um, mm, m).
- Calcul `pic50` en privilégiant `pchembl_value` si disponible; sinon conversion en M via les unités puis `-log10(M)`.
- Filtrage plausible par `pic50` dans [4, 10] (≈ 0.1 nM à 100 µM).
- Définition de `active` par `pic50 >= 6`.

### Fichier impacté
- `chembl_dataset_preparation.py` (méthode `clean_bioactivity_data`).

### Résultat
- Pipeline robuste aux types/units hétérogènes; plus d'erreur de type et cohérence accrue du dataset.

---

## 🚀 2025-09-14 - Mise à niveau du pipeline ML (scaffold split, nested CV, régression pIC50, calibration, AD, chembl_dataset_preparation.logs)

### Contexte
- L’exécution du pipeline se fait dans l’environnement conda `Aragorn` et doit produire un audit complet.

### Changements clés
- Split train/test par scaffolds (Bemis–Murcko) et nested cross-validation (GroupKFold).
- Objectif principal en régression pIC50; classification binaire optionnelle au cutoff 6.5 et étiquettes 3 classes (≤5.5, 5.5–6.5, ≥6.5).
- Consolidation par `canonical_smiles` (médiane pIC50), retrait des contradictions (std > 1.0), homogénéisation de l’espèce (priorité Homo sapiens).
- Calibration des probabilités (isotonic/sigmoid) et estimation d’incertitude (écart-type des arbres RF + conformal prediction).
- Domaine d’applicabilité via similarité Tanimoto au plus proche voisin du train.
- Métriques de tri: PR-AUC, EF@1%, EF@5%, Top-k precision, BEDROC.
- Audit complet dans `chembl_dataset_preparation.logs` (versions, date d’extraction, filtres appliqués, cutoff, hyperparamètres, résultats de validation, AD).

### Fichiers impactés
- `chembl_dataset_preparation.py` (refonte majeure du pipeline).
- `rules.md` (rappel environnement conda et mise à jour des logs).

### Sorties générées
- `data/X_features.npy`, `data/y_labels.npy` (binaire 6.5), `data/y_reg.npy`, `data/y_labels_3class.npy`, `data/chembl_dataset_full.csv`, `data/dataset_info.pkl`.
- `chembl_dataset_preparation.logs` pour l’audit horodaté et détaillé.

---

## 🧾 2025-09-14 - Journal des modifications (CHANGELOG)

### Ajouts
- Curation ChEMBL robuste dans `chembl_dataset_preparation.py` :
  - Filtres d’activité stricts : `standard_type=IC50`, `standard_relation='='`, `assay_type='B'`, `assay_confidence_score>=8`, `data_validity_comment is null`.
  - Journalisation des versions et capture de la release ChEMBL.
  - Standardisation chimique (rdMolStandardize), calcul d’InChIKey et déduplication.
  - Filtres de qualité chimique (PAINS, Brenk, NIH, non organiques, mélanges) avec comptages par raison.
  - Agrégation par InChIKey (médiane pIC50), seuil d’écart-type des réplicats configurable (par défaut 0.5).
  - Curation des bits : suppression des colonnes à variance nulle uniquement (bits constants 0/1) ; normalisation z uniquement des physico-chimiques.
  - Splits scaffold/cluster (Bemis–Murcko, Butina@0.6/0.7) avec exports JSON et empreintes (hash).
  - Statistiques de domaine d’applicabilité et export de la distribution (NN Tanimoto).
  - Export Parquet du dataset complet ; enrichissement des champs de `dataset_info.pkl`.
- Documentation : `DATA_CARD.md`, `rules_compliance.md`.
- Tests : `tests/test_dataset_integrity.py` couvrant les invariants du dataset.

### Modifications
- Fichier de log renommé en `chembl_dataset_preparation.logs` et référencé dans `logs.md`.
- `rules.md` conservé comme référence ; instructions d’exécution et journalisation alignées.

### Corrections
- Seuil binaire pIC50 ≥ 6.5 appliqué de manière cohérente dans le pipeline et les logs.

---

*Dernière mise à jour : 2025-09-14 13:21*

---

## 🧩 2025-09-14 - Correctifs pipeline dataset (alignement, curation bits, traçabilité)

### Problèmes résolus
- Splits JSON mal alignés avec X: recalcul des splits sur l’index de `final_df` et assertions de couverture/overlap.
- 211 colonnes Morgan constantes conservées: purge via `VarianceThreshold(0.0)` et curation bits constants uniquement.
- Réplicats bruyants: exclusion `pic50_std > 0.5` (paramétrable) en agrégation InChIKey, avec journalisation des rejets.
- Traçabilité: ajout `inchikey` et `standard_smiles` dans CSV/Parquet.
- Métadonnées: enrichies (`chembl_release`, `quality_rejections`, `bit_curation`, `versions`, `hashes` des splits, stats AD).
- Densité Morgan: maintenue dans [0.5%, 5%] après VarianceThreshold(0.0); fallback à 4096 bits si densité > 5%.

---

## 🔒 2025-09-15 - Durcissement final du pipeline dataset (conformité règles)

### Changements clés
- Standardisation chimique stricte: `FragmentParent → Normalize → Reionize → Canonicalize`; InChIKey via `from rdkit.Chem import inchi; inchi.MolToInchiKey(...)` uniquement.
- Agrégation par InChIKey avec exclusion des réplicats `pic50_std > 0.5` avant descripteurs et sauvegardes; logs des rejets.
- Curation des features: retrait des seules colonnes constantes; Z-score des 6 physico-chimiques; `VarianceThreshold(0.0)` juste avant sauvegarde de X; densité Morgan contrôlée (fallback 4096 bits).
- Splits recalculés après `final_df` avec assertions anti-fuite (overlap InChIKey/SMILES=0, couverture indices, max=n-1); export JSON + hash synchronisé dans `dataset_info.pkl`.
- `dataset_info.pkl` enrichi: distributions `y_reg`, paramètres d'empreinte, hashes des splits, stats AD.

### Exécution
- Environnement: `conda activate Aragorn`.
- Calibration scikit-learn: compat 1.3 (`CalibratedClassifierCV(estimator=..., cv='prefit')`) avec garde-fous de classes.

### Sorties
- Exports cohérents: `X_features.npy`, `y_labels.npy`, `y_reg.npy`, `y_labels_3class.npy`, `chembl_dataset_full.parquet/csv`, `splits/*.json`, `ad_stats.json`, `ad_nn_similarity.npy`.

*Dernière mise à jour : 2025-09-15 21:24*

## 📅 2025-09-16 - Alignement dataset (B=1073) — Décision et exécution (EN COURS)

### 🔎 Problème
Mélange d’artefacts **A (1634)** et **B (1073)** dans `data/`, entraînant des incohérences entre X/Y/info/CSV et les splits/AD/duplicates.

### 📌 Constats (état actuel de `data/`)
**Jeu A (1634) — aligné en interne**
- `X_features.npy` : `(1634, 2054)`
- `y_labels.npy` : `1634` *(taux d’actifs 64.81 %)*
- `y_labels_3class.npy` : `{inactive=171, intermediate=404, active=1059}`
- `y_reg.npy` : `1634` *(moy=6.76, std=0.73, q05=5.57, q50=6.75, q95=8.00)*
- `dataset_info.pkl` : `n_samples=1634`, `n_features=2054` *(métadonnées minimales)*
- `chembl_dataset_full.csv` : `(1634, 2059)` → **sans** `inchikey`/`standard_smiles`

**Jeu B (1073) — aligné en interne**
- `scaffold_split.json` : `train=859`, `test=214`, `index_max=1072`
- `cluster_split_t06.json` : `train=858`, `test=215`
- `cluster_split_t07.json` : `train=858`, `test=215`
- `ad_nn_similarity.npy` : `215` valeurs *(moy=0.686, std=0.135)*
- `ad_stats.json` : `q05=0.411`, `q50=0.693`, `q95=0.870`, seuil recommandé `0.30`
- `duplicates_report.csv` : `1073` lignes, `7` groupes `xhash` dupliqués, taille max `3`

**Conclusion d’état**
- `splits/`, AD et `duplicates` correspondent à **B=1073**.
- `X/Y/dataset_info/CSV` correspondent à **A=1634**.

---

### ✅ Décision
- **Option retenue** : **B (1073)** pour lot verrouillé, traçable, avec splits/AD/duplicates prêts.
- **Garde-fou** : `LOCK_B_1073=1` pour assert de taille.

---

### 🛠️ Plan d’action (B=1073)
1. Exécuter `prepare_dataset()` avec `LOCK_B_1073=1` sur le lot B.
2. **Régénérer et écraser** sur 1073 :  
   - `X_features.npy`, `y_labels*.npy`, `y_reg.npy`, `dataset_info.pkl`, `chembl_dataset_full.*`
3. **Conserver** (déjà cohérents B) :  
   - `splits/`, `ad_stats.json`, `ad_nn_similarity.npy`, `duplicates_report.csv`
4. **Exporter** CSV/Parquet **avec** `inchikey` et `standard_smiles` (script corrigé).
5. **Optionnel** : archiver les versions non alignées en `*_raw_*` (géré par le code).

---

### ↩️ Alternative (si A=1634 était retenu)
- Recalculer `scaffold_split.json`, `cluster_split_t06.json`, `cluster_split_t07.json` sur 1634.
- Recalculer `ad_nn_similarity.npy` et `ad_stats.json` sur 1634 (split scaffold).
- Refaire `duplicates_report.csv` et vérifier **0 fuite xhash** train/test.
- Réécrire `dataset_info.pkl` (hash splits, AD, feature_names, versions…).
- Réécrire `chembl_dataset_full.csv/parquet` avec `inchikey`/`standard_smiles`.

---

### ✍️ Changements de code (déjà intégrés)
- `LOCK_B_1073` pour verrouillage de taille sur B.
- Export `inchikey`/`standard_smiles` dans CSV/Parquet.
- Anti-fuite `xhash` dans les splits + rapport de doublons.
- AD (NN Tanimoto) + exports `ad_nn_similarity.npy` / `ad_stats.json`.
- `dataset_info.pkl` enrichi (hash splits/AD, densité Morgan, versions…).

---

### 🧪 Check d’intégrité (rapide)
- **A (1634)** : X/Y/info/CSV **cohérents**, mais CSV **sans identifiants** et **pas** de splits/AD/duplicates correspondants.
- **B (1073)** : splits/AD/duplicates **OK, sans fuite** ; **manquent** X/Y/info/CSV recalculés sur 1073.

---

### 📁 Structure attendue après alignement B

```
data/
├── X_features.npy # (1073, 2054)
├── y_labels.npy # (1073,)
├── y_labels_3class.npy # (1073,)
├── y_reg.npy # (1073,)
├── dataset_info.pkl # meta enrichies (hash, AD, versions…)
├── chembl_dataset_full.csv # avec inchikey/standard_smiles
├── chembl_dataset_full.parquet # idem
├── splits/
│ ├── scaffold_split.json
│ ├── cluster_split_t06.json
│ └── cluster_split_t07.json
├── ad_stats.json
├── ad_nn_similarity.npy
└── duplicates_report.csv
```

---

### 🚀 Prochaines étapes
- Lancer `prepare_dataset()` avec `LOCK_B_1073=1`.
- Committer les artefacts et **tagger** : `dataset_COX2_B_1073_r1`.

---

*Dernière mise à jour : 2025-09-16*