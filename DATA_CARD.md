# DATA CARD — Hit Identificator Dataset (COX-2)

## Source et Extraction
- Source: ChEMBL (via chembl-webresource-client)
- Cible: CHEMBL279 (Cyclooxygenase-2 / COX-2)
- Release ChEMBL: voir `dataset_info.pkl` → `chembl_release`
- Date de requête: voir `dataset_info.pkl` → `request_date`

## Filtres et Curation
- Activités: `standard_type = IC50`, `standard_relation = '='`, `assay_type = 'B'`
- Qualité: `assay_confidence_score ≥ 8`, `data_validity_comment is null`
- Unités: standardisation (pm, nm, um, mm, m)
- pIC50: priorité `pchembl_value`, sinon conversion de `standard_value` en M puis `-log10(M)`
- Plage scientifique: pIC50 ∈ [4, 10]
- Espèce: homogénéisation (priorité `Homo sapiens`)

## Standardisation chimique
- rdMolStandardize: `FragmentParent` → `Normalize` → `Reionize` → `Tautomer Canonicalization`
- Identifiant: `InChIKey`
- Déduplication: par `InChIKey`

## Qualité chimique (exclusions)
- Mixtures, non organiques (sans Carbone)
- Filtres RDKit `FilterCatalog`: PAINS, Brenk, NIH
- Compte et raisons: voir `dataset_info.pkl` → `quality_rejections`

## Agrégation des réplicats
- Agrégation par `InChIKey`: médiane pIC50, `n_measures`
- Exclusion réplicats incohérents: `std > 0.5` (paramétrable)

## Étiquetage
- Binaire: actif si pIC50 ≥ 6.5 (seule vérité)
- 3 classes: ≤5.5 / (5.5, 6.5) / ≥6.5

## Descripteurs
- Morgan ECFP4: `radius = 2`, `nBits = 2048` (bits `uint8`). Si densité moyenne > 5%, fallback `nBits = 4096`.
- Descripteurs physico-chimiques: `molecular_weight`, `logp`, `hbd`, `hba`, `rotatable_bonds`, `tpsa`
- Curation des bits: suppression des colonnes à variance nulle uniquement (bits constants 0/1); les bits rares sont conservés
- Normalisation: uniquement les 6 physico-chimiques (z-score)
- Densité moyenne ECFP: voir `dataset_info.pkl` → `morgan_density_mean`

## Splits d’évaluation
- Scaffold split (Bemis–Murcko) — indices exportés: `data/splits/scaffold_split.json`
- Cluster split (Butina Tanimoto 0.6, 0.7) — exporté: `cluster_split_t06.json`, `cluster_split_t07.json`
- Invariants: 0 overlap `InChIKey` et SMILES entre train/test

## Domaine d’applicabilité (AD)
- Similarité Tanimoto NN (train→test sur split scaffold)
- Distribution sauvegardée: `data/ad_nn_similarity.npy`, stats `data/ad_stats.json`
- Seuil recommandé: 0.3

## Distribution des classes et statistiques
- Voir `dataset_info.pkl`:
  - `n_samples`, `n_features`
  - `n_active_6p5`, `n_inactive_6p5`, `activity_ratio_6p5`
  - `class3_counts`
  - `bit_curation`, `morgan_active_bits_mean`
  - `splits.{scaffold,cluster_t06,cluster_t07}.hash`

## Limitations et Risques de biais
- Biais de publication (ChEMBL), redondances d’analogues (atténuées par splits chimie)
- Qualité d’annotation assays; variabilité expérimentale
- AD: performances dégradées hors domaine (NN < 0.3)

## Sorties
- `data/X_features.npy`
- `data/y_labels.npy`, `data/y_reg.npy`, `data/y_labels_3class.npy`
- `data/chembl_dataset_full.parquet`, `data/dataset_info.pkl`
- `data/splits/*.json`, `data/ad_nn_similarity.npy`, `data/ad_stats.json`

## RUNBOOK — Comment régénérer
1) Activer l’environnement (règle repo 1):
   - `conda activate Aragorn`
2) Installer dépendances (si nécessaire):
   - `pip install -r requirements.txt`
3) Générer le dataset:
   - `$env:LOCK_B_1073="1"; python .\chembl_dataset_preparation.py`
4) Vérifier intégrité (pytest):
   - `pytest -q`

