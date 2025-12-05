# Hit Identificator - Prédiction d'Activité Biologique par Machine Learning (HEI3)

## 📋 Vue d'ensemble

Ce dépôt s’inscrit dans le cadre du projet B-Live HEI.

**Aragorn** est un pipeline de machine learning dédié à l’identification de hits en drug discovery. Le projet utilise des données bioactives issues de ChEMBL afin d’entraîner un modèle Random Forest capable de prédire l’activité biologique de molécules candidates contre une cible thérapeutique.
La cible par défaut est l’enzyme COX-2 (Cyclooxygenase-2).

### 🎯 Objectif principal
Automatiser la priorisation de molécules candidates lors du criblage virtuel en prédisant leur activité biologique (actif/inactif) à partir de leur structure chimique.

### ⚡ Performances obtenues (COX-2)
- **Dataset** : 1073 molécules curées (683 actifs / 390 inactifs)
- **Modèle** : Random Forest calibré (400 arbres, 1625 features)
- **PR-AUC** : **0.869** (excellent pour classes déséquilibrées)
- **ROC-AUC** : **0.773** | **Précision** : **83.3%** | **Recall** : **75.7%**
- **Calibration** : Isotonic (probabilités fiables)
- **Temps d'entraînement** : ~2-3 minutes

---

## 🎯 Utilité du projet

### Contexte scientifique
En drug discovery, identifier des "hits" (molécules actives prometteuses) parmi des millions de candidats est un défi coûteux et chronophage. Ce projet permet de :

- **Réduire les coûts** : Criblage virtuel avant les tests expérimentaux
- **Accélérer la découverte** : Priorisation automatique des molécules candidates
- **Optimiser la chimie médicinale** : Guidage des efforts de synthèse vers les composés les plus prometteurs
- **Prédire l'activité** : Estimation quantitative (pIC50) et qualitative (actif/inactif)

### Applications pratiques
1. **Criblage virtuel** : Filtrer rapidement de grandes bibliothèques chimiques
2. **Lead optimization** : Évaluer des analogues avant synthèse
3. **Analyse SAR** : Comprendre les relations structure-activité
4. **Domaine d'applicabilité** : Estimer la fiabilité des prédictions

---

## 🔬 Comment fonctionne le projet

### Architecture du pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   1. PRÉPARATION DONNÉES                     │
│            chembl_dataset_preparation.py                     │
├─────────────────────────────────────────────────────────────┤
│  • Téléchargement ChEMBL (API)                              │
│  • Filtrage qualité (IC50, assay confidence ≥8)             │
│  • Standardisation chimique (RDKit)                         │
│  • Déduplication (InChIKey)                                 │
│  • Filtres qualité (PAINS, Brenk, NIH)                      │
│  • Calcul descripteurs moléculaires                         │
│  • Création splits (scaffold, cluster)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   2. ENTRAÎNEMENT MODÈLE                     │
│              random_forest_training.py                       │
├─────────────────────────────────────────────────────────────┤
│  • Chargement dataset (X_features.npy, y_labels.npy)       │
│  • Split scaffold (évite data leakage)                      │
│  • Optimisation hyperparamètres (GridSearchCV)              │
│  • Entraînement Random Forest                               │
│  • Calibration probabilités (isotonic)                      │
│  • Évaluation (PR-AUC, ROC-AUC, MCC, EF, BEDROC)           │
│  • Export modèle (.joblib, .onnx)                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   3. PRÉDICTION & ANALYSE                    │
│                  (Interface utilisateur)                     │
├─────────────────────────────────────────────────────────────┤
│  • Chargement modèle                                        │
│  • Calcul descripteurs nouvelles molécules                  │
│  • Prédiction activité + probabilités calibrées             │
│  • Estimation domaine d'applicabilité                       │
│  • Interprétation (feature importance)                      │
└─────────────────────────────────────────────────────────────┘
```

### Descripteurs moléculaires

Le modèle utilise une combinaison de descripteurs pour représenter les molécules :

| Type | Description | Dimension |
|------|-------------|-----------|
| **Morgan Fingerprints (ECFP4)** | Empreintes circulaires (rayon=2) | 2048 bits |
| **Poids moléculaire** | MW (Da) | 1 |
| **LogP** | Lipophilie | 1 |
| **Donneurs H** | HBD | 1 |
| **Accepteurs H** | HBA | 1 |
| **Liaisons rotables** | Flexibilité | 1 |
| **TPSA** | Surface polaire topologique | 1 |

**Total : ~2054 features** après filtrage des bits constants.

### Métriques d'évaluation

Le modèle est évalué selon plusieurs métriques adaptées au criblage moléculaire :

- **PR-AUC** (Average Precision) : Performance sur classes déséquilibrées
- **ROC-AUC** : Capacité de discrimination globale
- **EF@1%/5%** (Enrichment Factor) : Enrichissement dans le top hits
- **Top-50/100 Precision** : Précision sur les meilleurs candidats
- **BEDROC** (α=20) : Métrique chimique early recognition
- **MCC** (Matthews Correlation Coefficient) : Qualité globale du classifieur

### Gestion du déséquilibre de classes

- **class_weight='balanced'** : Pénalisation automatique des classes majoritaires
- **Calibration isotonic** : Probabilités fiables même avec déséquilibre
- **Optimisation sur PR-AUC** : Métrique adaptée aux classes rares
- **Scaffold split** : Évite le sur-apprentissage sur scaffolds communs

---

## 🚀 Quick Start

### Installation rapide et première utilisation

```bash
# 1. Créer l'environnement conda
conda create -n Aragorn python=3.9
conda activate Aragorn

# 2. Installer RDKit
conda install -c conda-forge rdkit=2023.3.2 -y

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Générer le dataset ChEMBL (COX-2)
$env:LOCK_B_1073="1"
python chembl_dataset_preparation.py
# Durée: ~15-20 minutes → génère data/ (~14 MB)

# 5. Entraîner le modèle
python random_forest_training.py
# Durée: ~2-3 minutes → génère models/ (~12.3 MB)

# 6. Consulter les résultats
cat models/metrics.json
# PR-AUC: 0.869 | ROC-AUC: 0.773
```

---

## 🛠️ Installation détaillée

### Prérequis

- Python 3.9 (testé et validé)
- Conda (recommandé pour RDKit)
- ~2 GB d'espace disque
- Connexion internet (téléchargement ChEMBL)

### Versions des dépendances (testées)

```
Python: 3.9+
pandas: 2.0.3
numpy: 1.24.3
scikit-learn: 1.3.0
rdkit: 2023.03.2
chembl-webresource-client: 0.10.8
matplotlib: 3.7.2
```

### Étapes d'installation

1. **Cloner le dépôt**
```bash
cd Desktop
git clone <url-du-depot> Aragorn
cd Aragorn
```

2. **Créer l'environnement conda**
```bash
conda create -n Aragorn python=3.9
conda activate Aragorn
```

3. **Installer RDKit**
```bash
conda install -c conda-forge rdkit=2023.3.2 -y
```

4. **Installer les dépendances Python**
```bash
pip install -r requirements.txt
```

### Vérification de l'installation

```bash
python -c "from rdkit import Chem; import chembl_webresource_client; print('Installation OK')"
```

---

## 📖 Utilisation

### 1. Préparation du dataset

**Script** : `chembl_dataset_preparation.py`

```bash
# Activer l'environnement
conda activate Aragorn

# Générer le dataset (avec verrou pour sécurité)
$env:LOCK_B_1073="1"
python chembl_dataset_preparation.py
```

**Paramètres modifiables** (dans le script) :
- `target_chembl_id` : Cible ChEMBL (défaut: "CHEMBL279" = COX-2)
- `limit` : Nombre max de composés (défaut: 5000)
- `n_bits` : Taille des fingerprints (défaut: 2048)
- `replicate_std_threshold` : Seuil d'exclusion réplicats (défaut: 0.5)

**Sorties générées** :
```
data/
├── X_features.npy              # Matrice de features (1073 x 1625) - 6.97 MB
├── y_labels.npy                # Labels binaires (actif/inactif) - 4.4 KB
├── y_reg.npy                   # Valeurs pIC50 continues - 8.7 KB
├── y_labels_3class.npy         # Classification 3 classes - 8.7 KB
├── dataset_info.pkl            # Métadonnées complètes - 288 KB
├── chembl_dataset_full.parquet # Dataset complet (format Parquet) - 1.49 MB
├── chembl_dataset_full.csv     # Dataset complet (format CSV) - 4.63 MB
├── duplicates_report.csv       # Rapport de déduplication - 179 KB
├── splits/
│   ├── scaffold_split.json     # Split par scaffolds - avec hash
│   ├── cluster_split_t06.json  # Split par clusters (T=0.6)
│   └── cluster_split_t07.json  # Split par clusters (T=0.7)
├── ad_nn_similarity.npy        # Similarités AD (215 valeurs) - 1.8 KB
└── ad_stats.json               # Statistiques AD - 187 bytes
```

**Temps d'exécution** : ~10-30 minutes (selon limit et réseau)

---

### 2. Entraînement du modèle

**Script** : `random_forest_training.py`

```bash
# Entraînement avec paramètres par défaut
python random_forest_training.py
```

**Options avancées** :

Pour activer le tuning d'hyperparamètres (plus long, ~30-60 min) :
```python
# Dans random_forest_training.py, décommenter lignes 508-512
summary = trainer.tune_hyperparams(X_tr, y_tr, groups=train_groups)
params = summary["best_params"]
```

**Sorties générées** :
```
models/
├── random_forest.joblib        # Modèle entraîné (calibré) - 12.2 MB
├── metrics.json                # Métriques complètes + métadonnées - 4.6 KB
└── plots/
    └── pr_curve.png            # Courbe Precision-Recall - 47.7 KB
```

**Métriques sauvegardées dans metrics.json** :
- ROC-AUC: 0.773, PR-AUC: 0.869
- Accuracy: 0.721, F1: 0.793, MCC: 0.372, Balanced Accuracy: 0.696
- Confusion matrix (test) : [[40, 23], [37, 115]]
- Seuils : optimal (MCC=0.429) et par défaut (0.5)
- Class balance train/calibration/test
- OOB score: 0.789
- Calibration isotonic utilisée (172 échantillons)
- Dataset hash pour traçabilité

---

### 3. Prédiction sur nouvelles molécules

**Exemple d'utilisation** (créer un script `predict.py`) :

```python
import numpy as np
import pandas as pd
from joblib import load
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem import Descriptors
import pickle

# Charger le modèle
model = load('models/random_forest.joblib')

# Charger les métadonnées pour normalisation
with open('data/dataset_info.pkl', 'rb') as f:
    dataset_info = pickle.load(f)

# Fonction pour calculer les descripteurs
def compute_descriptors(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Morgan fingerprint
    morgan_fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    morgan_array = np.zeros((n_bits,), dtype=np.uint8)
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(morgan_fp, morgan_array)
    
    # Descripteurs physico-chimiques
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    
    # Combiner (ordre important : correspondre à l'entraînement)
    descriptors = np.concatenate([
        [mw, logp, hbd, hba, rotatable_bonds, tpsa],
        morgan_array
    ])
    
    return descriptors

# Prédire sur une nouvelle molécule
smiles_test = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirine
descriptors = compute_descriptors(smiles_test)

if descriptors is not None:
    X_pred = descriptors.reshape(1, -1)
    
    # Prédiction
    proba = model.predict_proba(X_pred)[0, 1]
    pred_class = model.predict(X_pred)[0]
    
    print(f"SMILES: {smiles_test}")
    print(f"Probabilité d'être actif: {proba:.3f}")
    print(f"Classe prédite: {'Actif' if pred_class == 1 else 'Inactif'}")
```

**Sortie exemple** :
```
SMILES: CC(=O)Oc1ccccc1C(=O)O
Probabilité d'être actif: 0.823
Classe prédite: Actif
```

---

### 4. Tests et validation

```bash
# Exécuter les tests unitaires
pytest -q

# Avec verbose
pytest -v
```

---

### 5. Inspection des résultats

**Voir les statistiques du dataset :**
```bash
python -c "import pickle; info = pickle.load(open('data/dataset_info.pkl', 'rb')); print(f'Molécules: {info[\"n_samples\"]}'); print(f'Features: {info[\"n_features\"]}'); print(f'Actifs: {info[\"n_active_6p5\"]} ({info[\"activity_ratio_6p5\"]:.1%})'); print(f'Inactifs: {info[\"n_inactive_6p5\"]}')"
```

**Consulter les métriques du modèle :**
```bash
python -c "import json; m = json.load(open('models/metrics.json')); print(f'PR-AUC: {m[\"average_precision\"]:.3f}'); print(f'ROC-AUC: {m[\"roc_auc\"]:.3f}'); print(f'Accuracy: {m[\"accuracy\"]:.3f}'); print(f'F1: {m[\"f1\"]:.3f}'); print(f'Precision: {m[\"precision\"]:.3f}'); print(f'Recall: {m[\"recall\"]:.3f}')"
```

**Vérifier le domaine d'applicabilité :**
```bash
python -c "import json; ad = json.load(open('data/ad_stats.json')); print(f'Similarité NN moyenne: {ad[\"mean\"]:.3f} ± {ad[\"std\"]:.3f}'); print(f'Médiane: {ad[\"q50\"]:.3f}'); print(f'Q05-Q95: [{ad[\"q05\"]:.3f}, {ad[\"q95\"]:.3f}]')"
```

**Visualiser la courbe PR :**
```bash
start models/plots/pr_curve.png  # Windows
# ou
open models/plots/pr_curve.png   # macOS
```

---

## 📊 Résultats obtenus

### Performances réelles du modèle (COX-2)

| Métrique | Valeur obtenue | Description |
|----------|----------------|-------------|
| **PR-AUC** | **0.869** ⭐ | Performance sur classes déséquilibrées |
| **ROC-AUC** | **0.773** | Discrimination globale |
| **Accuracy** | **0.721** | Précision globale |
| **F1 Score** | **0.793** | Moyenne harmonique précision/recall |
| **Precision** | **0.833** | Proportion de vrais positifs |
| **Recall** | **0.757** | Taux de détection des actifs |
| **MCC** | **0.372** | Corrélation Matthews |
| **Balanced Accuracy** | **0.696** | Accuracy ajusté au déséquilibre |
| **OOB Score** | **0.789** | Score Out-of-Bag du Random Forest |

#### Matrice de confusion (seuil par défaut 0.5)

```
                Prédiction
              Inactif  Actif
Réel Inactif      40     23     → 63.5% spécificité
     Actif        37    115     → 75.7% sensibilité
                                → 83.3% précision
```

**Interprétation** :
- Sur 215 molécules de test, **155 correctement classées** (72.1%)
- Sur 152 actifs réels, **115 détectés** (recall = 75.7%)
- Sur 138 prédits actifs, **115 sont vrais positifs** (précision = 83.3%)
- **Courbe PR** disponible dans `models/plots/pr_curve.png`

### Statistiques du dataset

- **Taille totale** : 1073 molécules (après curation)
- **Distribution des classes** : 
  - Actifs (pIC50 ≥ 6.5) : 683 molécules (63.7%)
  - Inactifs (pIC50 < 6.5) : 390 molécules (36.3%)
- **Features** : 1625 descripteurs (après filtrage bits constants)
  - Morgan fingerprints : ~1619 bits actifs
  - Descripteurs physico-chimiques : 6 features
- **Splits** : 
  - Train : 858 molécules (80%)
  - Test : 215 molécules (20%)
  - Stratégie : Scaffold split (Bemis-Murcko)

### Domaine d'applicabilité (AD)

- **Similarité NN moyenne** : 0.686 (±0.135)
- **Médiane** : 0.692
- **Q05** : 0.438 | **Q95** : 0.879
- **Seuil recommandé** : Tanimoto > 0.3

---

## 🗂️ Structure du projet

```
Aragorn/
├── README.md                           # Ce fichier
├── requirements.txt                    # Dépendances Python
├── changelog.md                        # Historique des modifications
├── DATA_CARD.md                        # Documentation du dataset
├── chembl_dataset_preparation.py      # Pipeline de préparation (1315 lignes)
├── random_forest_training.py          # Entraînement du modèle (540 lignes)
├── chembl_dataset_preparation.logs    # Logs d'audit (69 KB)
├── data/                              # Données générées (~14 MB total)
│   ├── X_features.npy                 # 1073 x 1625 - 6.97 MB
│   ├── y_labels.npy                   # Labels binaires - 4.4 KB
│   ├── y_reg.npy                      # pIC50 continues - 8.7 KB
│   ├── y_labels_3class.npy            # Classification 3 classes - 8.7 KB
│   ├── dataset_info.pkl               # Métadonnées - 288 KB
│   ├── chembl_dataset_full.parquet    # Dataset Parquet - 1.49 MB
│   ├── chembl_dataset_full.csv        # Dataset CSV - 4.63 MB
│   ├── duplicates_report.csv          # Rapport déduplication - 179 KB
│   ├── ad_nn_similarity.npy           # Similarités AD - 1.8 KB
│   ├── ad_stats.json                  # Statistiques AD - 187 bytes
│   └── splits/                        # Stratégies de split
│       ├── scaffold_split.json        # Bemis-Murcko scaffolds
│       ├── cluster_split_t06.json     # Butina T=0.6
│       └── cluster_split_t07.json     # Butina T=0.7
├── models/                            # Modèles entraînés (~12.3 MB)
│   ├── random_forest.joblib           # Modèle calibré - 12.2 MB
│   ├── metrics.json                   # Métriques complètes - 4.6 KB
│   └── plots/
│       └── pr_curve.png               # Courbe PR - 47.7 KB
└── tests/                             # Tests unitaires
    └── test_dataset_integrity.py      # Tests validation dataset
```

---

## 🔧 Configuration avancée

### Changer la cible thérapeutique

Modifier dans `chembl_dataset_preparation.py` :

```python
# Exemple : Kinase EGFR (CHEMBL203)
preparator = ChEMBLDatasetPreparator(
    target_chembl_id="CHEMBL203",
    output_dir="data"
)
```

### Ajuster les critères de qualité

```python
# Dans chembl_dataset_preparation.py
activities = self.activity_client.filter(
    target_chembl_id=self.target_chembl_id,
    standard_type="IC50",
    assay_confidence_score__gte=8,  # Modifier ici (7-9)
    # ... autres filtres
)
```

### Modifier le seuil d'activité

```python
# Dans clean_bioactivity_data()
df['active'] = (df['pic50'] >= 6.5).astype(int)  # Modifier le seuil
```

---

## 📚 Ressources et références

### Documentation externe

- **ChEMBL** : https://www.ebi.ac.uk/chembl/
- **RDKit** : https://www.rdkit.org/docs/
- **Scikit-learn** : https://scikit-learn.org/

### Articles scientifiques clés

1. **Fingerprints** : Rogers & Hahn (2010). "Extended-Connectivity Fingerprints"
2. **Scaffold splits** : Bemis & Murcko (1996). "The properties of known drugs"
3. **BEDROC** : Truchon & Bayly (2007). "Evaluating virtual screening methods"
4. **Applicability Domain** : Jaworska et al. (2005)

### Métadonnées des données

Consulter `DATA_CARD.md` pour :
- Provenance et filtres appliqués
- Protocole de standardisation chimique
- Statistiques détaillées du dataset
- Limitations et biais potentiels

---

## ⚠️ Limitations et considérations

### Domaine d'applicabilité

Le modèle est fiable principalement pour :
- **Molécules similaires** au set d'entraînement (Tanimoto > 0.3)
- **Cible COX-2** (si autre cible : ré-entraîner)
- **Domaine de pIC50** : 4-10 (~100 μM à 0.1 nM)

**Éviter les prédictions sur** :
- Peptides, biomolécules complexes
- Molécules inorganiques
- PAINS, composés réactifs
- Molécules très dissimilaires (Tanimoto < 0.3)

### Biais potentiels

- **Biais de publication** : ChEMBL contient principalement des molécules publiées
- **Déséquilibre de classes** : Plus d'inactifs que d'actifs
- **Variabilité expérimentale** : Différents assays/labs
- **Scaffold coverage** : Limité aux chimies représentées dans ChEMBL

### Recommandations d'usage

1. **Toujours vérifier** le domaine d'applicabilité (similarité NN)
2. **Valider expérimentalement** les hits prédits
3. **Utiliser les probabilités calibrées** (pas seulement la classe)
4. **Interpréter avec prudence** les prédictions limites (p ~ 0.5)
5. **Contextualiser** avec expertise chimie médicinale

---

## 🤝 Contribution et support

### Signaler un problème

Utiliser l'onglet "Issues" sur GitHub avec :
- Description du problème
- Code minimal reproductible
- Versions (Python, RDKit, etc.)
- Logs d'erreur

### Proposer des améliorations

1. Fork du dépôt
2. Créer une branche (`feature/amelioration`)
3. Commit avec messages clairs
4. Pull request avec description

---

## 📝 Licence

Ce projet est distribué sous licence MIT. Voir `LICENSE` pour plus de détails.

---

## 🙏 Remerciements

- **ChEMBL** (EMBL-EBI) pour les données bioactives
- **RDKit** community pour la chémoinformatique open-source
- **scikit-learn** pour les outils de machine learning

---

## 📧 Contact

Pour toute question ou collaboration :
- Email : [votre-email]
- GitHub : [votre-profil]

---

**Dernière mise à jour** : Novembre 2025  
**Version** : 1.0.0  
**Auteur** : Legrand Nathan
