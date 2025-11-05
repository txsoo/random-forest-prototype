"""
ChEMBL Dataset Preparation for Hit Identification
================================================

Ce script télécharge et prépare un dataset ChEMBL pour l'entraînement d'un modèle Random Forest
destiné à l'identification de hits en drug discovery.

Étapes:
1. Téléchargement des données bioactives depuis ChEMBL
2. Filtrage et nettoyage des données
3. Calcul des descripteurs moléculaires (Morgan fingerprints)
4. Préparation du dataset pour l'entraînement ML
"""

import pandas as pd
import numpy as np
import chembl_webresource_client
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import FilterCatalog
from rdkit.Chem import inchi
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    r2_score, mean_squared_error
)
from sklearn.base import clone
from sklearn.feature_selection import VarianceThreshold
import pickle
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AuditLogger:
    """Audit logger simple écrivant dans chembl_dataset_preparation.logs (texte brut)."""
    def __init__(self, log_path: str = "chembl_dataset_preparation.logs"):
        self.log_path = log_path
        self._append("\n" + "="*80 + "\n")
        self._append(f"Session start: {datetime.now().isoformat()}\n")

    def _append(self, txt: str):
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(txt)

    def log(self, msg: str):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
        print(msg)
        self._append(line)


class ChEMBLDatasetPreparator:
    def __init__(self, target_chembl_id="CHEMBL279", output_dir="data"):
        """
        Initialise le préparateur de dataset ChEMBL
        
        Args:
            target_chembl_id (str): ID ChEMBL de la cible (par défaut: CHEMBL279 - Cyclooxygenase-2)
            output_dir (str): Répertoire de sortie pour les fichiers
        """
        self.target_chembl_id = target_chembl_id
        self.output_dir = output_dir
        self.activity_client = new_client.activity
        self.molecule_client = new_client.molecule
        
        # Créer le répertoire de sortie
        os.makedirs(output_dir, exist_ok=True)
        # Dossier splits
        self.splits_dir = os.path.join(output_dir, 'splits')
        os.makedirs(self.splits_dir, exist_ok=True)
        # Logger d'audit
        self.logger = AuditLogger()
        # Catalogues filtres (PAINS, Brenk, NIH)
        self.filter_catalog = self._build_filter_catalog()
        # Seuil std pour exclusion des réplicats (agrégation)
        self.replicate_std_threshold = 0.5
        
    def download_bioactivity_data(self, limit=5000):
        """
        Télécharge les données de bioactivité depuis ChEMBL
        
        Args:
            limit (int): Nombre maximum de composés à télécharger
            
        Returns:
            pd.DataFrame: Données de bioactivité
        """
        print(f"Téléchargement des données de bioactivité pour {self.target_chembl_id}...")
        
        # Journaliser release ChEMBL
        try:
            status = new_client.status.all()[0]
            self.chembl_release = status.get('chembl_db_version')
            self.logger.log(f"ChEMBL status: release={self.chembl_release}, api_version={status.get('api_version')}")
        except Exception as e:
            self.chembl_release = None
            self.logger.log(f"ChEMBL status unavailable: {e}")

        # Récupérer les activités pour la cible spécifiée (filtres stricts)
        activities = self.activity_client.filter(
            target_chembl_id=self.target_chembl_id,
            standard_type="IC50",
            relation="=",
            standard_relation="=",
            assay_type="B",
            assay_confidence_score__gte=8,
            data_validity_comment__isnull=True
        ).only([
            'activity_id', 'assay_chembl_id', 'assay_description', 'assay_type',
            'assay_organism', 'target_organism', 'document_chembl_id',
            'assay_confidence_score', 'data_validity_comment',
            'molecule_chembl_id', 'type', 'standard_type', 'standard_units', 'standard_relation',
            'standard_value', 'pchembl_value'
        ])[:limit]
        
        # Convertir en DataFrame
        bioactivity_df = pd.DataFrame.from_records(activities)
        
        print(f"Téléchargé {len(bioactivity_df)} enregistrements d'activité")
        return bioactivity_df
    
    def get_compound_data(self, molecule_chembl_ids):
        """
        Récupère les données des composés (SMILES, propriétés)
        
        Args:
            molecule_chembl_ids (list): Liste des IDs ChEMBL des molécules
            
        Returns:
            pd.DataFrame: Données des composés
        """
        print("Récupération des structures moléculaires...")
        
        compounds_data = []
        batch_size = 50  # Traitement par batch pour éviter les timeouts
        
        for i in range(0, len(molecule_chembl_ids), batch_size):
            batch_ids = molecule_chembl_ids[i:i+batch_size]
            
            try:
                compounds = self.molecule_client.filter(
                    molecule_chembl_id__in=batch_ids
                ).only(['molecule_chembl_id', 'molecule_structures'])
                
                for compound in compounds:
                    if compound['molecule_structures']:
                        compounds_data.append({
                            'molecule_chembl_id': compound['molecule_chembl_id'],
                            'canonical_smiles': compound['molecule_structures']['canonical_smiles']
                        })
                        
            except Exception as e:
                print(f"Erreur lors du traitement du batch {i//batch_size + 1}: {e}")
                continue
        
        compounds_df = pd.DataFrame(compounds_data)
        print(f"Récupéré {len(compounds_df)} structures moléculaires")
        return compounds_df
    
    def clean_bioactivity_data(self, bioactivity_df):
        """
        Nettoie et filtre les données de bioactivité
        
        Args:
            bioactivity_df (pd.DataFrame): Données brutes de bioactivité
            
        Returns:
            pd.DataFrame: Données nettoyées
        """
        print("Nettoyage des données de bioactivité...")
        
        # Copie pour éviter les effets de chaîne et normaliser les colonnes
        df = bioactivity_df.copy()
        
        # Garder uniquement les lignes avec un identifiant molécule valide
        df = df.dropna(subset=['molecule_chembl_id'])
        
        # Conversion robuste des valeurs et normalisation des unités
        if 'standard_value' in df.columns:
            df['standard_value_num'] = pd.to_numeric(df['standard_value'], errors='coerce')
        else:
            df['standard_value_num'] = np.nan
        
        if 'standard_units' in df.columns:
            units = df['standard_units'].astype(str).str.strip().str.lower()
            # Normaliser des variantes d'écriture
            units = units.replace({
                'µm': 'um',
                'μm': 'um',
                'micromolar': 'um',
                'nanomolar': 'nm',
                'millimolar': 'mm',
                'molar': 'm',
                'mol/l': 'm'
            })
            df['standard_units_norm'] = units
        else:
            df['standard_units_norm'] = np.nan
        
        # Calcul pIC50 privilégiant pchembl_value quand disponible
        unit_to_M = {'pm': 1e-12, 'nm': 1e-9, 'um': 1e-6, 'mm': 1e-3, 'm': 1.0}
        
        def _compute_pic50(row):
            pv = row.get('pchembl_value', np.nan)
            if pd.notnull(pv):
                pv_num = pd.to_numeric(pv, errors='coerce')
                if pd.notnull(pv_num):
                    return float(pv_num)
            val = row['standard_value_num']
            unit = row['standard_units_norm']
            if pd.isnull(val) or pd.isnull(unit):
                return np.nan
            factor = unit_to_M.get(unit)
            if factor is None or val <= 0:
                return np.nan
            M = val * factor
            if M <= 0:
                return np.nan
            return -np.log10(M)
        
        df['pic50'] = df.apply(_compute_pic50, axis=1)
        
        # Homogénéiser l'espèce si possible (priorité Homo sapiens, sinon organisme dominant)
        if 'assay_organism' in df.columns:
            try:
                org_counts = df['assay_organism'].dropna().astype(str).str.strip().value_counts()
                top_org = 'Homo sapiens' if 'Homo sapiens' in org_counts.index else (org_counts.index[0] if len(org_counts) > 0 else None)
                if top_org:
                    df = df[df['assay_organism'].astype(str).str.strip() == top_org]
            except Exception:
                pass
        
        # Filtrer les valeurs plausibles (0.1 nM à 100 μM <=> pIC50 ∈ [4, 10])
        df = df[(df['pic50'] >= 4) & (df['pic50'] <= 10)]
        
        # Définir les classes d'activité (seuil à pIC50 = 6.5)
        df['active'] = (df['pic50'] >= 6.5).astype(int)
        
        print(f"Données nettoyées: {len(df)} enregistrements")
        print(f"Composés actifs: {df['active'].sum()}")
        print(f"Composés inactifs: {len(df) - df['active'].sum()}")
        
        return df

    def _build_filter_catalog(self):
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH)
        return FilterCatalog.FilterCatalog(params)

    def standardize_smiles_inchikey(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None
        try:
            parent = rdMolStandardize.FragmentParent(mol)
            normalizer = rdMolStandardize.Normalizer()
            nm = normalizer.normalize(parent)
            reion = rdMolStandardize.Reionizer().reionize(nm)
            taut = rdMolStandardize.TautomerEnumerator().Canonicalize(reion)
            clean_smiles = Chem.MolToSmiles(taut, isomericSmiles=True)
            ik = inchi.MolToInchiKey(taut)
            return taut, clean_smiles, ik
        except Exception:
            return None, None, None

    def _quality_filter(self, mol: Chem.Mol) -> (bool, list):
        reasons = []
        if mol is None:
            return True, ['invalid_mol']
        # Mixtures (plusieurs fragments)
        if len(Chem.GetMolFrags(mol)) > 1:
            reasons.append('mixture')
        # Exclure non organiques (pas de Carbone)
        if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
            reasons.append('non_organic')
        # Filtres PAINS/Brenk/NIH
        matches = self.filter_catalog.GetMatches(mol)
        if matches:
            reasons += [m.GetDescription() for m in matches]
        return (len(reasons) > 0), reasons
    
    def calculate_molecular_descriptors(self, smiles_list, n_bits: int = 2048):
        """
        Calcule les descripteurs moléculaires (Morgan fingerprints + descripteurs RDKit)
        
        Args:
            smiles_list (list): Liste des SMILES
            
        Returns:
            pd.DataFrame: Descripteurs moléculaires
        """
        print("Calcul des descripteurs moléculaires...")
        
        descriptors_data = []
        
        for i, smiles in enumerate(smiles_list):
            if i % 500 == 0:
                print(f"Traitement: {i}/{len(smiles_list)} molécules")
                
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                # Morgan fingerprints (ECFP4, rayon=2, n_bits)
                morgan_fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
                # Conversion robuste vers numpy (uint8)
                morgan_array = np.zeros((n_bits,), dtype=np.uint8)
                try:
                    from rdkit import DataStructs as _DS
                    _DS.ConvertToNumpyArray(morgan_fp, morgan_array)
                except Exception:
                    # Fallback: itère sur bits
                    morgan_array = np.array([int(morgan_fp.GetBit(i)) for i in range(n_bits)], dtype=np.uint8)
                
                # Descripteurs moléculaires RDKit
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                rotatable_bonds = Descriptors.NumRotatableBonds(mol)
                tpsa = Descriptors.TPSA(mol)
                
                # Combiner tous les descripteurs
                descriptor_dict = {
                    'smiles': smiles,
                    'molecular_weight': mw,
                    'logp': logp,
                    'hbd': hbd,
                    'hba': hba,
                    'rotatable_bonds': rotatable_bonds,
                    'tpsa': tpsa
                }
                
                # Ajouter les bits du Morgan fingerprint
                for j, bit in enumerate(morgan_array):
                    descriptor_dict[f'morgan_bit_{j}'] = bit
                
                descriptors_data.append(descriptor_dict)
                
            except Exception as e:
                print(f"Erreur lors du calcul des descripteurs pour {smiles}: {e}")
                continue
        
        descriptors_df = pd.DataFrame(descriptors_data)
        print(f"Descripteurs calculés pour {len(descriptors_df)} molécules")
        
        return descriptors_df

    def curate_bit_features(self, df: pd.DataFrame) -> (pd.DataFrame, dict):
        """
        Retire uniquement les bits constants (tout 0 / tout 1) pour conserver les bits rares.
        Normalise uniquement les 6 descripteurs physico-chimiques. Retourne le df et un rapport.
        """
        physchem_cols = ['molecular_weight', 'logp', 'hbd', 'hba', 'rotatable_bonds', 'tpsa']
        bit_cols = [c for c in df.columns if c.startswith('morgan_bit_')]
        # Fréquences de '1' (ne supprimer QUE les bits constants: tout 0 ou tout 1)
        if len(bit_cols) > 0:
            freq = df[bit_cols].mean(axis=0)
            to_drop = list(freq[(freq == 0.0) | (freq == 1.0)].index)
            df = df.drop(columns=to_drop)
        else:
            to_drop = []
        # Normalisation physchem (z-score)
        scaler = {}
        for c in physchem_cols:
            if c in df.columns:
                m = float(df[c].mean())
                s = float(df[c].std(ddof=0) or 1.0)
                df[c] = (df[c] - m) / (s if s != 0 else 1.0)
                scaler[c] = {'mean': m, 'std': s}
        report = {
            'bit_cols_initial': len(bit_cols),
            'bit_cols_dropped': len(to_drop),
            'bit_cols_kept': len([c for c in df.columns if c.startswith('morgan_bit_')]),
            'physchem_scaler': scaler
        }
        return df, report

    def butina_cluster_split(self, smiles: list, threshold: float = 0.6, test_frac: float = 0.2, seed: int = 42):
        """Cluster split via Butina clustering on Tanimoto distances.
        Returns train/test index arrays.
        """
        fps = []
        valid_idx = []
        for i, s in enumerate(smiles):
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            fp = GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
            fps.append(fp)
            valid_idx.append(i)
        # Distance matrix in upper-triangle list
        dists = []
        for i in range(1, len(fps)):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1 - x for x in sims])
        cs = Butina.ClusterData(dists, len(fps), threshold, isDistData=True)
        # Assign clusters
        clusters = sorted(cs, key=len, reverse=True)
        rng = np.random.RandomState(seed)
        rng.shuffle(clusters)
        train_set, test_set = set(), set()
        total = len(valid_idx)
        target_test = int(round(test_frac * total))
        count_test = 0
        for cl in clusters:
            indices = [valid_idx[i] for i in cl]
            if count_test < target_test:
                test_set.update(indices)
                count_test += len(indices)
            else:
                train_set.update(indices)
        # Remaining invalid idx (mol parse failed) to train by default
        all_idx = set(range(len(smiles)))
        others = all_idx - (train_set | test_set)
        train_set.update(list(others))
        return np.array(sorted(list(train_set))), np.array(sorted(list(test_set)))

    def export_split(self, name: str, train_idx: np.ndarray, test_idx: np.ndarray, dataset_hash: str | None = None):
        payload = {
            'strategy': name,
            'train_index': train_idx.tolist(),
            'test_index': test_idx.tolist(),
        }
        if dataset_hash:
            payload['dataset_hash'] = dataset_hash
        h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()
        payload['hash'] = h
        out_path = os.path.join(self.splits_dir, f"{name}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        self.logger.log(f"Split exporté: {out_path} | hash={h}")
        return h

    @staticmethod
    def compute_dataset_hash(inchikeys: list[str]) -> str:
        if not inchikeys:
            return hashlib.sha256(b"").hexdigest()
        normalized: list[str] = []
        for ik in inchikeys:
            if isinstance(ik, str):
                normalized.append(ik.strip().upper())
            elif ik is None:
                normalized.append("")
            else:
                normalized.append(str(ik).strip().upper())
        joined = "\n".join(normalized).encode('utf-8')
        return hashlib.sha256(joined).hexdigest()

    @staticmethod
    def bemis_murcko_scaffold(smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

    def compute_scaffolds(self, smiles_list: list) -> list:
        return [self.bemis_murcko_scaffold(s) for s in smiles_list]

    def scaffold_split(self, smiles: list, test_size: float = 0.2, random_state: int = 42):
        """Split train/test par scaffolds (approximation via GroupKFold).
        n_splits est borné par le nombre de scaffolds uniques.
        """
        scaffolds = np.array(self.compute_scaffolds(smiles))
        _, group_ids = np.unique(scaffolds, return_inverse=True)
        n_groups = len(np.unique(group_ids))
        desired = max(2, int(round(1 / max(1e-6, test_size))))
        n_splits = min(desired, max(2, n_groups))
        gkf = GroupKFold(n_splits=n_splits)
        best = None
        best_gap = 1.0
        for tr_idx, te_idx in gkf.split(X=np.zeros(len(smiles)), y=None, groups=group_ids):
            gap = abs(len(te_idx) / max(1, len(smiles)) - test_size)
            if gap < best_gap:
                best = (tr_idx, te_idx)
                best_gap = gap
        if best is not None:
            return best[0], best[1], scaffolds
        idx = np.arange(len(smiles))
        return idx[: int(len(idx) * (1 - test_size))], idx[int(len(idx) * (1 - test_size)) :], scaffolds

    def tanimoto_nn_similarity(self, fps_train, fp):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps_train) if len(fps_train) > 0 else []
        return max(sims) if len(sims) > 0 else 0.0

    def compute_applicability_domain(self, smiles_train: list, smiles_test: list, n_bits: int = None):
        """Calcule la similarité Tanimoto au plus proche voisin du train pour chaque test."""
        n_bits = n_bits or getattr(self, "n_bits_used", 2048)
        fps_train = []
        for s in smiles_train:
            m = Chem.MolFromSmiles(s)
            fps_train.append(GetMorganFingerprintAsBitVect(m, radius=2, nBits=n_bits) if m is not None else None)
        fps_train_valid = [fp for fp in fps_train if fp is not None]
        nn_sims = []
        for s in smiles_test:
            m = Chem.MolFromSmiles(s)
            if m is None or len(fps_train_valid) == 0:
                nn_sims.append(0.0)
            else:
                fp = GetMorganFingerprintAsBitVect(m, radius=2, nBits=n_bits)
                nn_sims.append(self.tanimoto_nn_similarity(fps_train_valid, fp))
        return np.array(nn_sims)

    @staticmethod
    def ef_at_fraction(y_true_binary: np.ndarray, y_scores: np.ndarray, frac: float) -> float:
        n = len(y_true_binary)
        k = max(1, int(np.ceil(frac * n)))
        order = np.argsort(-y_scores)
        top_k = order[:k]
        hit_rate_top = y_true_binary[top_k].mean() if k > 0 else 0.0
        hit_rate_all = y_true_binary.mean() if n > 0 else 0.0
        return (hit_rate_top / hit_rate_all) if hit_rate_all > 0 else np.nan

    @staticmethod
    def top_k_precision(y_true_binary: np.ndarray, y_scores: np.ndarray, k: int) -> float:
        k = min(k, len(y_true_binary))
        order = np.argsort(-y_scores)
        top_k = order[:k]
        return y_true_binary[top_k].mean() if k > 0 else np.nan

    @staticmethod
    def bedroc_score(y_true_binary: np.ndarray, y_scores: np.ndarray, alpha: float = 20.0) -> float:
        order = np.argsort(-y_scores)
        y = np.array(y_true_binary)[order]
        n = len(y)
        n_act = y.sum()
        if n_act == 0 or n == 0:
            return np.nan
        pos = np.where(y == 1)[0] + 1
        ra = n_act / n
        s = alpha / (1 - np.exp(-alpha))
        sum_exp = np.sum(np.exp(-alpha * pos / n))
        bedroc = (s * sum_exp / n_act) - ra
        return float(max(0.0, min(1.0, bedroc)))

    def nested_cv_rf(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, task: str = 'regression', random_state: int = 42):
        # Adapte le nombre de splits aux groupes uniques disponibles
        n_groups = len(np.unique(groups))
        outer_splits = min(5, max(2, n_groups))
        outer = GroupKFold(n_splits=outer_splits)
        if task == 'regression':
            base = RandomForestRegressor(random_state=random_state, n_estimators=300, n_jobs=-1)
            param_grid = {
                'n_estimators': [200, 400],
                'max_depth': [None, 20, 40],
                'min_samples_leaf': [1, 2]
            }
            scoring = 'neg_root_mean_squared_error'
        else:
            base = RandomForestClassifier(random_state=random_state, n_estimators=300, n_jobs=-1, class_weight='balanced')
            param_grid = {
                'n_estimators': [200, 400],
                'max_depth': [None, 20, 40],
                'min_samples_leaf': [1, 2]
            }
            scoring = 'average_precision'
        best_params_list = []
        outer_scores = []
        for tr, te in outer.split(X, y, groups=groups):
            # inner CV spécifique au split courant
            n_groups_tr = len(np.unique(groups[tr]))
            inner_splits = min(3, max(2, n_groups_tr))
            inner = GroupKFold(n_splits=inner_splits)
            grid = GridSearchCV(base, param_grid, cv=inner, scoring=scoring, n_jobs=-1, refit=True)
            grid.fit(X[tr], y[tr], groups=groups[tr])
            best_params_list.append(grid.best_params_)
            if task == 'regression':
                preds = grid.predict(X[te])
                rmse = mean_squared_error(y[te], preds, squared=False)
                outer_scores.append(-rmse)
            else:
                prob = grid.predict_proba(X[te])[:, 1]
                ap = average_precision_score(y[te], prob)
                outer_scores.append(ap)
        best_params = max(best_params_list, key=best_params_list.count) if len(best_params_list) > 0 else grid.best_params_
        best_estimator = clone(base).set_params(**best_params)
        best_estimator.fit(X, y)
        report = {
            'best_params': best_params,
            'outer_score_mean': float(np.mean(outer_scores)) if outer_scores else None,
            'outer_score_std': float(np.std(outer_scores)) if outer_scores else None
        }
        return best_estimator, report

    def calibrate_classifier(self, clf: RandomForestClassifier, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        method = 'isotonic' if np.sum(y) >= 50 else 'sigmoid'
        n_groups = len(np.unique(groups))
        n_splits = min(5, max(2, n_groups))
        gkf = GroupKFold(n_splits=n_splits)
        # Trouver un split où les deux classes sont présentes en train et calibration
        for tr, cal in gkf.split(X, y, groups=groups):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[cal])) < 2:
                continue
            clf.fit(X[tr], y[tr])
            calibrated = CalibratedClassifierCV(estimator=clf, cv='prefit', method=method)
            calibrated.fit(X[cal], y[cal])
            return calibrated
        clf.fit(X, y)
        return CalibratedClassifierCV(estimator=clf, cv=3, method=method).fit(X, y)

    def conformal_prediction_intervals(self, model: RandomForestRegressor, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float = 0.1, groups: np.ndarray = None):
        if groups is not None and len(groups) == len(y_train):
            gkf = GroupKFold(n_splits=5)
            for tr, cal in gkf.split(X_train, y_train, groups=groups):
                break
        else:
            n = len(y_train)
            idx = np.arange(n)
            rs = np.random.RandomState(42)
            rs.shuffle(idx)
            split = int(0.8 * n)
            tr, cal = idx[:split], idx[split:]
        base = clone(model)
        base.fit(X_train[tr], y_train[tr])
        cal_preds = base.predict(X_train[cal])
        resid = np.abs(y_train[cal] - cal_preds)
        q = np.quantile(resid, 1 - alpha)
        test_preds = base.predict(X_test)
        lower = test_preds - q
        upper = test_preds + q
        return test_preds, lower, upper, float(q)

    def run_ml_pipeline(self, X: np.ndarray, y_reg: np.ndarray, y_clf: np.ndarray, smiles: list):
        """
        Entraîne RF régression (pIC50) + RF classification (calibrée), split par scaffolds,
        nested CV pour HP, incertitudes (std RF + conformal), domaine d'applicabilité,
        et métriques (PR-AUC, EF@1/5%, Top-k, BEDROC). Journalisation dans chembl_dataset_preparation.logs.
        """
        self.logger.log("Split par scaffolds (Bemis–Murcko)")
        tr_idx, te_idx, scaffolds = self.scaffold_split(smiles, test_size=0.2)
        groups_all = np.array(self.compute_scaffolds(smiles))
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train_reg, y_test_reg = y_reg[tr_idx], y_reg[te_idx]
        y_train_clf, y_test_clf = y_clf[tr_idx], y_clf[te_idx]
        smiles_train = [smiles[i] for i in tr_idx]
        smiles_test = [smiles[i] for i in te_idx]
        # Vérifier l'absence de duplicats train/test
        overlap = set(smiles_train).intersection(set(smiles_test))
        if len(overlap) > 0:
            self.logger.log(f"Attention: {len(overlap)} SMILES partagés entre train et test (corriger split)")
        else:
            self.logger.log("Aucun SMILES en commun entre train et test")

        # Régression avec nested CV
        self.logger.log("Nested CV pour RandomForestRegressor (objectif: pIC50)")
        rf_reg, reg_report = self.nested_cv_rf(X_train, y_train_reg, groups=groups_all[tr_idx], task='regression')
        self.logger.log(f"Reg best_params={reg_report['best_params']}, outer_score_mean(-RMSE)={reg_report['outer_score_mean']}")
        # Conformal prediction
        self.logger.log("Conformal prediction (régression)")
        reg_preds_test, reg_lower, reg_upper, reg_q = self.conformal_prediction_intervals(rf_reg, X_train, y_train_reg, X_test, alpha=0.1, groups=groups_all[tr_idx])
        # Incertitude via std des arbres
        tree_preds = np.vstack([est.predict(X_test) for est in rf_reg.estimators_])
        reg_std = tree_preds.std(axis=0)

        # Classification calibrée
        self.logger.log("Nested CV pour RandomForestClassifier (cutoff 6.5)")
        rf_clf_base, clf_report = self.nested_cv_rf(X_train, y_train_clf, groups=groups_all[tr_idx], task='classification')
        self.logger.log(f"Clf best_params={clf_report['best_params']}, outer_score_mean(AP)={clf_report['outer_score_mean']}")
        self.logger.log("Calibration des probabilités (isotonic/sigmoid)")
        rf_clf = self.calibrate_classifier(rf_clf_base, X_train, y_train_clf, groups=groups_all[tr_idx])
        prob_test = rf_clf.predict_proba(X_test)[:, 1]

        # Applicability domain
        self.logger.log("Calcul du domaine d'applicabilité (Tanimoto NN)")
        nn_sim = self.compute_applicability_domain(
            smiles_train,
            smiles_test,
            n_bits=getattr(self, 'n_bits_used', 2048)
        )
        confidence = prob_test * nn_sim

        # Métriques
        self.logger.log("Calcul des métriques (régression et tri moléculaire)")
        rmse = mean_squared_error(y_test_reg, reg_preds_test, squared=False)
        r2 = r2_score(y_test_reg, reg_preds_test)
        spearman = pd.Series(y_test_reg).corr(pd.Series(reg_preds_test), method='spearman')
        ap = average_precision_score(y_test_clf, prob_test)
        ef1 = self.ef_at_fraction(y_test_clf, prob_test, frac=0.01)
        ef5 = self.ef_at_fraction(y_test_clf, prob_test, frac=0.05)
        top50 = self.top_k_precision(y_test_clf, prob_test, k=min(50, len(prob_test)))
        top100 = self.top_k_precision(y_test_clf, prob_test, k=min(100, len(prob_test)))
        bedroc = self.bedroc_score(y_test_clf, prob_test, alpha=20.0)
        self.logger.log(f"REG: RMSE={rmse:.3f}, R2={r2:.3f}, Spearman={spearman:.3f}")
        self.logger.log(f"CLF: PR-AUC={ap:.3f}, EF@1%={ef1:.3f}, EF@5%={ef5:.3f}, Top-50={top50:.3f}, Top-100={top100:.3f}, BEDROC={bedroc:.3f}")
        # Confiance moyenne ajustée AD
        self.logger.log(f"AD: Similarité NN moyenne={float(np.mean(nn_sim)):.3f}, Confiance moyenne ajustée={float(np.mean(confidence)):.3f}")

        return {
            'regression': {'rmse': float(rmse), 'r2': float(r2), 'spearman': float(spearman), 'q_conformal': float(reg_q)},
            'classification': {'ap': float(ap), 'ef1': float(ef1), 'ef5': float(ef5), 'top50': float(top50), 'top100': float(top100), 'bedroc': float(bedroc)},
            'applicability_domain': {'nn_sim_mean': float(np.mean(nn_sim)), 'confidence_mean': float(np.mean(confidence))}
        }
    
    def prepare_dataset(self, limit=5000):
        """
        Pipeline complet de préparation du dataset
        
        Args:
            limit (int): Nombre maximum de composés à traiter
            
        Returns:
            tuple: (X, y_reg, y_clf, final_df, dataset_info)
        """
        print("=== PRÉPARATION DU DATASET ChEMBL ===")
        print(f"Cible: {self.target_chembl_id}")
        print(f"Limite: {limit} composés")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        self.logger.log(f"Début préparation dataset | target={self.target_chembl_id} | limit={limit}")
        # Versions d'environnement pour audit
        try:
            self.logger.log(
                f"Versions | pandas={pd.__version__}, numpy={np.__version__}, rdkit={rdBase.rdkitVersion}, sklearn={__import__('sklearn').__version__}, chembl_webresource_client={chembl_webresource_client.__version__}"
            )
        except Exception as e:
            try:
                self.logger.log(f"Version logging failed: {e}")
            except Exception:
                pass

        # --- FAST PATH: réutiliser le processed existant s'il est là (verrou B) ---
        processed_parquet = os.path.join(self.output_dir, 'chembl_dataset_full.parquet')
        processed_csv = os.path.join(self.output_dir, 'chembl_dataset_full.csv')
        if os.path.exists(processed_parquet) or os.path.exists(processed_csv):
            self.logger.log("Reusing existing processed full dataset (skip download/clean)")
            full_df = (pd.read_parquet(processed_parquet)
                       if os.path.exists(processed_parquet)
                       else pd.read_csv(processed_csv))
            # Sanity B
            assert {'inchikey','standard_smiles','pic50_median'}.issubset(full_df.columns), \
                "Fichier processed incomplet (inchikey/standard_smiles/pic50_median manquants)"
            if os.getenv("LOCK_B_1073") == "1":
                assert len(full_df) == 1073, f"Taille attendue 1073, obtenue {len(full_df)}"
            # Repars de full_df comme 'final_df'
            final_df = full_df.copy()
            goto_features = True
        else:
            goto_features = False

        # Valeurs par défaut utiles pour dataset_info quand fast-path
        reasons_counter = {}
        n_inconsistent = 0
        
        # 1..7 : download/clean/aggregate/compute descriptors
        if not goto_features:
            # 1. Télécharger les données de bioactivité
            bioactivity_df = self.download_bioactivity_data(limit)
            self.logger.log(f"Bioactivité téléchargée: n={len(bioactivity_df)} | filtres: type=IC50, relation='=', assay_type='B'")
            
            # 2. Nettoyer les données
            clean_bioactivity_df = self.clean_bioactivity_data(bioactivity_df)
            self.logger.log(f"Nettoyage: pIC50 range=[4,10], classes(6.5)=binary(>=6.5), organisme homogénéisé si possible")
            
            # 3. Récupérer les structures moléculaires
            unique_molecule_ids = clean_bioactivity_df['molecule_chembl_id'].unique().tolist()
            compounds_df = self.get_compound_data(unique_molecule_ids)
            
            # 4. Fusionner les données et dédupliquer
            merged_df = clean_bioactivity_df.merge(
                compounds_df, 
                on='molecule_chembl_id', 
                how='inner'
            )
            merged_df = merged_df.dropna(subset=['canonical_smiles'])
            self.logger.log(f"Fusion bioactivité+SMILES: n={len(merged_df)}")

            # 4b. Standardisation chimique et InChIKey
            std_records = []
            rejected_quality = 0
            reasons_counter = {}
            for s in merged_df['canonical_smiles'].astype(str).tolist():
                taut, std_smiles, ik = self.standardize_smiles_inchikey(s)
                if taut is None or std_smiles is None or ik is None:
                    std_records.append({'standard_smiles': None, 'inchikey': None, 'reject': True, 'reasons': ['std_failed']})
                    rejected_quality += 1
                    reasons_counter['std_failed'] = reasons_counter.get('std_failed', 0) + 1
                    continue
                flag, reasons = self._quality_filter(taut)
                if flag:
                    rejected_quality += 1
                    for r in reasons:
                        reasons_counter[r] = reasons_counter.get(r, 0) + 1
                std_records.append({'standard_smiles': std_smiles, 'inchikey': ik, 'reject': flag, 'reasons': reasons})
            std_df = pd.DataFrame(std_records)
            merged_df = merged_df.reset_index(drop=True).join(std_df)
            before_quality = len(merged_df)
            merged_df = merged_df[~merged_df['reject'] & merged_df['inchikey'].notna()].copy()
            self.logger.log(f"Qualité chimique: rejetés={rejected_quality}/{before_quality} | raisons={reasons_counter}")

            # 5. Agrégation par InChIKey (médiane pIC50, seuil std paramétrable)
            grp = merged_df.groupby('inchikey')['pic50']
            agg_df = grp.agg(['median', 'count', 'std']).reset_index()
            agg_df.rename(columns={'median': 'pic50_median', 'count': 'n_measures', 'std': 'pic50_std'}, inplace=True)
            # Marquer rejets
            inconsistent = agg_df[(agg_df['n_measures'] >= 2) & (agg_df['pic50_std'] > self.replicate_std_threshold)]
            n_inconsistent = int(len(inconsistent))
            agg_df = agg_df[(agg_df['pic50_std'].isna()) | (agg_df['pic50_std'] <= self.replicate_std_threshold)]
            # Associer un standard_smiles représentatif
            rep_smiles = merged_df.drop_duplicates('inchikey')[['inchikey', 'standard_smiles']]
            agg_df = agg_df.merge(rep_smiles, on='inchikey', how='left')
            self.logger.log(f"Agrégation par InChIKey: n_unique={len(agg_df)} | réplicats incohérents retirés={n_inconsistent} (std>{self.replicate_std_threshold})")

            # 6. Calculer les descripteurs moléculaires (sur standard_smiles uniques)
            n_bits = 2048
            descriptors_df = self.calculate_molecular_descriptors(agg_df['standard_smiles'].tolist(), n_bits=n_bits)
            self.n_bits_used = n_bits
            
            # 7. Fusionner avec les pIC50 agrégées
            final_df = agg_df.merge(
                descriptors_df,
                left_on='standard_smiles',
                right_on='smiles',
                how='inner'
            )
        
        # 8. Préparer les features (X) et cibles (régression et classification)
        feature_columns = [col for col in final_df.columns if col.startswith('morgan_bit_') or 
                          col in ['molecular_weight', 'logp', 'hbd', 'hba', 'rotatable_bonds', 'tpsa']]
        # Curation des bits et normalisation physchem + VarianceThreshold(0.0)
        curated_df, bit_report = self.curate_bit_features(final_df[feature_columns].copy())
        n_features_before = int(curated_df.shape[1])
        vt = VarianceThreshold(threshold=0.0)
        X_vt = vt.fit_transform(curated_df.values)
        kept_mask = vt.get_support()
        kept_cols = curated_df.columns[kept_mask]
        curated_kept_df = curated_df[kept_cols]
        feature_columns = list(kept_cols)
        X = X_vt.astype(np.float32)
        n_features_after = int(X.shape[1])
        # Statistiques sur densité des bits Morgan (post-VT)
        bit_cols_kept = [c for c in feature_columns if c.startswith('morgan_bit_')]
        morgan_density = float(curated_kept_df[bit_cols_kept].mean().mean()) if bit_cols_kept else 0.0
        active_bits_per_mol = curated_kept_df[bit_cols_kept].sum(axis=1).mean() if bit_cols_kept else 0.0
        self.logger.log(f"Features: before_VT={n_features_before}, after_VT={n_features_after}, morgan_density={morgan_density:.4f} (cible [0.005, 0.05])")

        # Fallback: si densité > 0.05, régénérer en 4096 bits et répéter curation+VT
        if morgan_density > 0.05:
            self.logger.log("Morgan density > 5% après VT, régénération des empreintes en 4096 bits")
            n_bits = 4096
            if 'agg_df' not in locals() or goto_features:
                # reconstruit un agg_df minimal à partir du final_df existant
                cols_base = ['inchikey', 'standard_smiles']
                extra_cols = []
                for c in ['pic50_median', 'n_measures', 'pic50_std']:
                    if c in final_df.columns:
                        extra_cols.append(c)
                agg_df = final_df[cols_base + extra_cols].drop_duplicates('inchikey').copy()
            descriptors_df = self.calculate_molecular_descriptors(agg_df['standard_smiles'].tolist(), n_bits=n_bits)
            final_df = agg_df.merge(
                descriptors_df,
                left_on='standard_smiles',
                right_on='smiles',
                how='inner'
            )
            feature_columns = [col for col in final_df.columns if col.startswith('morgan_bit_') or 
                              col in ['molecular_weight', 'logp', 'hbd', 'hba', 'rotatable_bonds', 'tpsa']]
            curated_df, bit_report = self.curate_bit_features(final_df[feature_columns].copy())
            n_features_before = int(curated_df.shape[1])
            vt = VarianceThreshold(threshold=0.0)
            X_vt = vt.fit_transform(curated_df.values)
            kept_mask = vt.get_support()
            kept_cols = curated_df.columns[kept_mask]
            curated_kept_df = curated_df[kept_cols]
            feature_columns = list(kept_cols)
            X = X_vt.astype(np.float32)
            n_features_after = int(X.shape[1])
            bit_cols_kept = [c for c in feature_columns if c.startswith('morgan_bit_')]
            morgan_density = float(curated_kept_df[bit_cols_kept].mean().mean()) if bit_cols_kept else 0.0
            active_bits_per_mol = curated_kept_df[bit_cols_kept].sum(axis=1).mean() if bit_cols_kept else 0.0
            self.logger.log(f"Features(4096): before_VT={n_features_before}, after_VT={n_features_after}, morgan_density={morgan_density:.4f}")
            self.n_bits_used = n_bits
        # Journaliser si la densité est encore hors cible
        if morgan_density > 0.05 or morgan_density < 0.005:
            self.logger.log(
                f"Attention: densité Morgan {morgan_density:.4f} hors plage visée [0.5%, 5%] après fallback 4096 bits"
            )
        # 8b. Assertions d'identifiants et cohérence réplicats (post-agrégation)
        assert 'inchikey' in final_df.columns and 'standard_smiles' in final_df.columns, "Colonnes d'identification manquantes"
        assert final_df['inchikey'].notna().all() and final_df['standard_smiles'].notna().all(), "Valeurs nulles dans les colonnes d'identification"
        assert final_df['inchikey'].is_unique, "inchikey doit être unique dans le dataset traité"
        # Aucun réplicat incohérent ne doit rester
        if 'pic50_std' in final_df.columns:
            assert final_df['pic50_std'].dropna().le(self.replicate_std_threshold).all(), "Des lignes avec pic50_std > 0.5 persistent dans le dataset traité"
        
        y_reg = final_df['pic50_median'].values
        # Binary classification cutoff à 6.5 demandé
        y_clf = (y_reg >= 6.5).astype(int)
        # Option 3 classes: inactive<=5.5, 5.5-6.5, active>=6.5
        bins = [-np.inf, 5.5, 6.5, np.inf]
        y_three = np.digitize(y_reg, bins) - 1  # 0,1,2

        # 9. Splits: scaffold et Butina (0.6 et 0.7)
        smiles_std = final_df['standard_smiles'].tolist()
        inchikeys = final_df['inchikey'].tolist()
        dataset_hash = self.compute_dataset_hash(inchikeys)
        sc_tr, sc_te, scaffolds_all = self.scaffold_split(smiles_std, test_size=0.2)
        scaffold_labels = [sc if sc is not None else '' for sc in scaffolds_all]
        n = len(smiles_std)
        # Vérifications d'overlap et de couverture index
        assert len(set(np.array(inchikeys)[sc_tr]).intersection(set(np.array(inchikeys)[sc_te]))) == 0, "Overlap InChIKey entre splits scaffold"
        assert len(set(np.array(smiles_std)[sc_tr]).intersection(set(np.array(smiles_std)[sc_te]))) == 0, "Overlap SMILES entre splits scaffold"
        union_idx = set(sc_tr.tolist()) | set(sc_te.tolist())
        assert union_idx == set(range(n)), "Couverture incomplète des index pour split scaffold"
        assert max(union_idx) == n - 1, "Max index attendu n-1 pour split scaffold"
        scaffold_hash = self.export_split('scaffold_split', sc_tr, sc_te, dataset_hash)
        self.logger.log(f"Split scaffold: train={len(sc_tr)}, test={len(sc_te)}")
        # Butina cluster splits
        cl_tr06, cl_te06 = self.butina_cluster_split(smiles_std, threshold=0.6, test_frac=0.2)
        assert len(set(np.array(inchikeys)[cl_tr06]).intersection(set(np.array(inchikeys)[cl_te06]))) == 0, "Overlap InChIKey entre splits cluster 0.6"
        assert len(set(np.array(smiles_std)[cl_tr06]).intersection(set(np.array(smiles_std)[cl_te06]))) == 0, "Overlap SMILES entre splits cluster 0.6"
        union_idx = set(cl_tr06.tolist()) | set(cl_te06.tolist())
        assert union_idx == set(range(n)), "Couverture incomplète des index pour split cluster 0.6"
        assert max(union_idx) == n - 1, "Max index attendu n-1 pour split cluster 0.6"
        cluster06_hash = self.export_split('cluster_split_t06', cl_tr06, cl_te06, dataset_hash)
        self.logger.log(f"Split cluster@0.6: train={len(cl_tr06)}, test={len(cl_te06)}")
        cl_tr07, cl_te07 = self.butina_cluster_split(smiles_std, threshold=0.7, test_frac=0.2)
        assert len(set(np.array(inchikeys)[cl_tr07]).intersection(set(np.array(inchikeys)[cl_te07]))) == 0, "Overlap InChIKey entre splits cluster 0.7"
        assert len(set(np.array(smiles_std)[cl_tr07]).intersection(set(np.array(smiles_std)[cl_te07]))) == 0, "Overlap SMILES entre splits cluster 0.7"
        union_idx = set(cl_tr07.tolist()) | set(cl_te07.tolist())
        assert union_idx == set(range(n)), "Couverture incomplète des index pour split cluster 0.7"
        assert max(union_idx) == n - 1, "Max index attendu n-1 pour split cluster 0.7"
        cluster07_hash = self.export_split('cluster_split_t07', cl_tr07, cl_te07, dataset_hash)
        self.logger.log(f"Split cluster@0.7: train={len(cl_tr07)}, test={len(cl_te07)}")

        # 9b. Rapport de doublons (xhash) et anti-fuite xhash entre folds
        xhash = [hashlib.sha1(X[i].tobytes()).hexdigest() for i in range(len(X))]
        dup_df = pd.DataFrame({
            'index': np.arange(len(X), dtype=int),
            'inchikey': inchikeys,
            'standard_smiles': smiles_std,
            'scaffold': list(scaffolds_all),
            'xhash': xhash,
        })
        dup_df['group_size'] = dup_df.groupby('xhash')['xhash'].transform('size')
        dup_path = os.path.join(self.output_dir, 'duplicates_report.csv')
        dup_df.to_csv(dup_path, index=False)
        dup_groups = dup_df.groupby('xhash').size()
        n_dup_groups = int((dup_groups > 1).sum())
        max_group = int(dup_groups.max() if not dup_groups.empty else 0)
        self.logger.log(f"Doublons: groupes duplicatifs={n_dup_groups} | taille max groupe={max_group}")

        def _check_and_fix_xhash_leak(split_name, train_idx, test_idx):
            tr_set = set(train_idx.tolist())
            te_set = set(test_idx.tolist())
            tr_x = set([xhash[i] for i in tr_set])
            te_x = set([xhash[i] for i in te_set])
            inter = tr_x.intersection(te_x)
            if len(inter) == 0:
                self.logger.log(f"{split_name}: xhash_cross_fold_violations=0")
                return train_idx, test_idx
            self.logger.log(f"{split_name}: xhash violations={len(inter)} → régénération")
            if split_name == 'scaffold_split':
                groups = pd.factorize([f"{scaffolds_all[i]}|{xhash[i]}" for i in range(n)])[0]
                gkf = GroupKFold(n_splits=min(5, max(2, len(np.unique(groups)))))
                for tr, te in gkf.split(np.zeros(n), groups=groups):
                    return np.array(sorted(tr)), np.array(sorted(te))
                return train_idx, test_idx
            else:
                for h in inter:
                    idx_h = set(np.where(np.array(xhash) == h)[0].tolist())
                    in_tr = len(idx_h & tr_set)
                    in_te = len(idx_h & te_set)
                    if in_tr >= in_te:
                        te_set -= idx_h
                        tr_set |= idx_h
                    else:
                        tr_set -= idx_h
                        te_set |= idx_h
                all_idx = set(range(n))
                missing = all_idx - (tr_set | te_set)
                if missing:
                    tr_set |= missing
                return np.array(sorted(list(tr_set))), np.array(sorted(list(te_set)))

        sc_tr, sc_te = _check_and_fix_xhash_leak('scaffold_split', sc_tr, sc_te)
        scaffold_hash = self.export_split('scaffold_split', sc_tr, sc_te, dataset_hash)
        cl_tr06, cl_te06 = _check_and_fix_xhash_leak('cluster_split_t06', cl_tr06, cl_te06)
        cluster06_hash = self.export_split('cluster_split_t06', cl_tr06, cl_te06, dataset_hash)
        cl_tr07, cl_te07 = _check_and_fix_xhash_leak('cluster_split_t07', cl_tr07, cl_te07)
        cluster07_hash = self.export_split('cluster_split_t07', cl_tr07, cl_te07, dataset_hash)

        # 10. Domaine d'applicabilité (sur split scaffold)
        nn_sim = self.compute_applicability_domain(
            [smiles_std[i] for i in sc_tr],
            [smiles_std[i] for i in sc_te],
            n_bits=getattr(self, 'n_bits_used', 2048)
        )
        ad_stats = {
            'mean': float(np.mean(nn_sim)),
            'std': float(np.std(nn_sim)),
            'q05': float(np.quantile(nn_sim, 0.05)),
            'q50': float(np.quantile(nn_sim, 0.50)),
            'q95': float(np.quantile(nn_sim, 0.95)),
            'recommended_threshold': 0.3
        }
        # Sauvegarder la distribution AD
        np.save(os.path.join(self.output_dir, 'ad_nn_similarity.npy'), nn_sim)
        with open(os.path.join(self.output_dir, 'ad_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(ad_stats, f, indent=2)

        # 11. Informations sur le dataset
        try:
            versions = {
                'pandas': pd.__version__,
                'numpy': np.__version__,
                'rdkit': rdBase.rdkitVersion,
                'sklearn': __import__('sklearn').__version__,
                'chembl_webresource_client': chembl_webresource_client.__version__,
            }
        except Exception:
            versions = {}

        # 11b. Assertions de taille avant sauvegarde
        assert X.shape[0] == len(y_clf) == len(y_reg) == len(final_df), "Incohérence tailles X/y/final_df"
        if os.getenv("LOCK_B_1073") == "1":
            assert len(final_df) == 1073, f"Taille attendue 1073, obtenue {len(final_df)}"
        elif len(final_df) != 1073:
            self.logger.log(f"Info: final_df={len(final_df)} différent de 1073 (LOCK_B_1073 désactivé)")
        ident_ok = ('inchikey' in final_df.columns and 'standard_smiles' in final_df.columns \
                    and final_df['inchikey'].notna().all() and final_df['standard_smiles'].notna().all() \
                    and final_df['inchikey'].is_unique)

        dataset_info = {
            'target_chembl_id': self.target_chembl_id,
            'n_samples': int(len(X)),
            'n_features': int(X.shape[1]),
            'dataset_hash': dataset_hash,
            'cutoff_binary': 6.5,
            'class3_thresholds': {'inactive_max': 5.5, 'active_min': 6.5},
            'n_active_6p5': int(np.sum(y_clf)),
            'n_inactive_6p5': int(len(y_clf) - np.sum(y_clf)),
            'activity_ratio_6p5': float(np.mean(y_clf)),
            'class3_counts': {
                'inactive': int(np.sum(y_three == 0)),
                'intermediate': int(np.sum(y_three == 1)),
                'active': int(np.sum(y_three == 2)),
            },
            'feature_names': feature_columns,
            'inchikeys': inchikeys,
            'scaffold_labels': scaffold_labels,
            'creation_date': datetime.now().isoformat(),
            'chembl_release': getattr(self, 'chembl_release', None),
            'request_date': datetime.now().strftime('%Y-%m-%d'),
            'replicate_std_threshold': float(self.replicate_std_threshold),
            'quality_rejections': reasons_counter,
            'bit_curation': bit_report,
            'morgan_density_mean': morgan_density,
            'morgan_active_bits_mean': float(active_bits_per_mol),
            'ident_cols_in_parquet': bool(ident_ok),
            'data_lineage': {
                'full_raw': None,
                'processed': os.path.join(self.output_dir, 'chembl_dataset_full.parquet')
            },
            'counts': {
                'n_processed': int(len(final_df)),
                'n_full_raw': None,
                'excluded_replicates_std_gt_0p5': int(n_inconsistent)
            },
            'distributions': {
                'y_reg': {
                    'mean': float(np.mean(y_reg)),
                    'std': float(np.std(y_reg)),
                    'q05': float(np.quantile(y_reg, 0.05)),
                    'q50': float(np.quantile(y_reg, 0.50)),
                    'q95': float(np.quantile(y_reg, 0.95)),
                }
            },
            'versions': versions,
            'splits': {
                'scaffold': {'hash': scaffold_hash, 'dataset_hash': dataset_hash, 'file': os.path.join('splits','scaffold_split.json')},
                'cluster_t06': {'hash': cluster06_hash, 'dataset_hash': dataset_hash, 'file': os.path.join('splits','cluster_split_t06.json')},
                'cluster_t07': {'hash': cluster07_hash, 'dataset_hash': dataset_hash, 'file': os.path.join('splits','cluster_split_t07.json')},
            },
            'ad_stats': ad_stats,
            'fingerprint': {'type': 'Morgan', 'radius': 2, 'nBits': int(getattr(self, 'n_bits_used', n_bits))}
        }

        # 12. Sauvegarder les données
        self.save_dataset(X, y_clf, dataset_info, final_df, y_reg=y_reg, y_multi=y_three)

        print("\n=== RÉSUMÉ DU DATASET ===")
        print(f"Nombre d'échantillons: {dataset_info['n_samples']}")
        print(f"Nombre de features: {dataset_info['n_features']}")
        print(f"Actifs (cutoff 6.5): {dataset_info['n_active_6p5']} ({dataset_info['activity_ratio_6p5']:.2%})")
        print("=" * 30)

        return X, y_reg, y_clf, final_df, dataset_info
 
    def save_dataset(self, X, y, dataset_info, full_df, y_reg=None, y_multi=None):
        """
        Sauvegarde le dataset préparé (features et labels multiples)
        """
        print("Sauvegarde du dataset...")

        # Purge auto des .npy incohérents (legacy 1634)
        for fn in ['X_features.npy','y_labels.npy','y_reg.npy','y_labels_3class.npy']:
            fpath = os.path.join(self.output_dir, fn)
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
            except Exception as e:
                self.logger.log(f"Warning: impossible de supprimer {fn}: {e}")

        # Sauvegarder les arrays numpy
        np.save(os.path.join(self.output_dir, 'X_features.npy'), X)
        np.save(os.path.join(self.output_dir, 'y_labels.npy'), y)
        if y_reg is not None:
            np.save(os.path.join(self.output_dir, 'y_reg.npy'), y_reg)
        if y_multi is not None:
            np.save(os.path.join(self.output_dir, 'y_labels_3class.npy'), y_multi)

        # Renommer d'anciens fichiers "full" non alignés si nécessaires
        parquet_path = os.path.join(self.output_dir, 'chembl_dataset_full.parquet')
        csv_path = os.path.join(self.output_dir, 'chembl_dataset_full.csv')
        raw_parquet_path = None
        raw_csv_path = None
        # Parquet
        if os.path.exists(parquet_path):
            try:
                old_parq = pd.read_parquet(parquet_path)
                if len(old_parq) != len(full_df):
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    raw_parquet_path = os.path.join(self.output_dir, f'chembl_dataset_full_raw_{ts}.parquet')
                    os.replace(parquet_path, raw_parquet_path)
            except Exception as e:
                self.logger.log(f"Lecture ancien parquet échouée: {e}")
        # CSV
        if os.path.exists(csv_path):
            try:
                old_csv = pd.read_csv(csv_path)
                if len(old_csv) != len(full_df):
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    raw_csv_path = os.path.join(self.output_dir, f'chembl_dataset_full_raw_{ts}.csv')
                    os.replace(csv_path, raw_csv_path)
            except Exception as e:
                self.logger.log(f"Lecture ancien csv échouée: {e}")

        # Sauvegarder le DataFrame complet aligné
        full_df.to_csv(csv_path, index=False)
        try:
            full_df.to_parquet(parquet_path, index=False)
        except Exception as e:
            self.logger.log(f"Parquet write failed: {e}")
        # Vérification post-écriture
        try:
            df_chk = pd.read_parquet(parquet_path)
        except Exception:
            df_chk = pd.read_csv(csv_path)
        assert {'inchikey','standard_smiles'}.issubset(df_chk.columns), "Identifiants manquants dans le fichier écrit"
        assert df_chk['inchikey'].is_unique, "inchikey doit rester unique après écriture"

        # Mettre à jour lineage et counts dans dataset_info et sauvegarder
        try:
            if 'data_lineage' in dataset_info:
                dataset_info['data_lineage']['full_raw'] = raw_parquet_path or raw_csv_path
            if 'counts' in dataset_info:
                # n_full_raw vaut la taille de l'ancien full s'il a été renommé
                n_full_raw = None
                try:
                    if raw_parquet_path:
                        n_full_raw = int(pd.read_parquet(raw_parquet_path).shape[0])
                    elif raw_csv_path:
                        n_full_raw = int(pd.read_csv(raw_csv_path).shape[0])
                except Exception:
                    n_full_raw = None
                dataset_info['counts']['n_full_raw'] = n_full_raw
        except Exception:
            pass

        # Sauvegarder les métadonnées
        with open(os.path.join(self.output_dir, 'dataset_info.pkl'), 'wb') as f:
            pickle.dump(dataset_info, f)

        print(f"Dataset sauvegardé dans {self.output_dir}/")
        try:
            self.logger.log(f"Dataset sauvegardé dans {self.output_dir}/ | n_samples={dataset_info['n_samples']} | n_features={dataset_info['n_features']}")
        except Exception:
            pass
 

    def verify_outputs(self, expected_n=None):
        """Vérifications express après exécution.
        - Vérifie formes des .npy
        - Vérifie intégrité du parquet/csv
        - Vérifie dataset_info.pkl (clés critiques)
        - Vérifie absence de fuite xhash entre folds enregistrés
        """
        base = self.output_dir
        # 1) Arrays
        x_path = os.path.join(base, 'X_features.npy')
        y_bin_path = os.path.join(base, 'y_labels.npy')
        y_reg_path = os.path.join(base, 'y_reg.npy')
        y_3c_path = os.path.join(base, 'y_labels_3class.npy')
        assert os.path.exists(x_path), "X_features.npy manquant"
        assert os.path.exists(y_bin_path), "y_labels.npy manquant"
        assert os.path.exists(y_reg_path), "y_reg.npy manquant"
        assert os.path.exists(y_3c_path), "y_labels_3class.npy manquant"
        X = np.load(x_path)
        y_bin = np.load(y_bin_path)
        y_reg = np.load(y_reg_path)
        y_3c = np.load(y_3c_path)
        n, p = X.shape
        if expected_n is not None:
            assert n == int(expected_n), f"Taille X attendue {expected_n}, obtenue {n}"
            assert y_bin.shape == (expected_n,), f"Taille y_labels attendue {(expected_n,)}, obtenue {y_bin.shape}"
            assert y_reg.shape == (expected_n,), f"Taille y_reg attendue {(expected_n,)}, obtenue {y_reg.shape}"
            assert y_3c.shape == (expected_n,), f"Taille y_labels_3class attendue {(expected_n,)}, obtenue {y_3c.shape}"
        # 2) Parquet/CSV
        parquet_path = os.path.join(base, 'chembl_dataset_full.parquet')
        csv_path = os.path.join(base, 'chembl_dataset_full.csv')
        if os.path.exists(parquet_path):
            try:
                df = pd.read_parquet(parquet_path)
            except Exception:
                df = pd.read_csv(csv_path)
        else:
            df = pd.read_csv(csv_path)
        if expected_n is not None:
            assert len(df) == int(expected_n), f"Parquet/CSV: taille attendue {expected_n}, obtenue {len(df)}"
        assert {'inchikey','standard_smiles'}.issubset(df.columns), "Parquet/CSV: colonnes inchikey/standard_smiles manquantes"
        assert df['inchikey'].is_unique, "Parquet/CSV: inchikey doit être unique"
        # 3) dataset_info.pkl
        info_path = os.path.join(base, 'dataset_info.pkl')
        assert os.path.exists(info_path), "dataset_info.pkl manquant"
        with open(info_path, 'rb') as f:
            info = pickle.load(f)
        assert 'n_samples' in info and 'n_features' in info and 'feature_names' in info, "dataset_info incomplet (n_samples/n_features/feature_names)"
        if expected_n is not None:
            assert int(info['n_samples']) == int(expected_n), f"dataset_info.n_samples={info['n_samples']} ≠ {expected_n}"
        assert int(info['n_features']) == int(p), f"dataset_info.n_features={info['n_features']} ≠ {p}"
        assert 'splits' in info and all(k in info['splits'] for k in ['scaffold','cluster_t06','cluster_t07']), "dataset_info.splits incomplet"
        assert 'dataset_hash' in info and isinstance(info['dataset_hash'], str) and len(info['dataset_hash']) > 0, "dataset_info.dataset_hash manquant"
        assert 'inchikeys' in info and len(info['inchikeys']) == n, "dataset_info.inchikeys manquant/incoh??rent"
        assert 'scaffold_labels' in info and len(info['scaffold_labels']) == n, "dataset_info.scaffold_labels manquant/incoh??rent"
        assert 'ad_stats' in info and all(k in info['ad_stats'] for k in ['mean','std','recommended_threshold']), "dataset_info.ad_stats incomplet"
        # 4) duplicates_report.csv
        dup_path = os.path.join(base, 'duplicates_report.csv')
        assert os.path.exists(dup_path), "duplicates_report.csv manquant"
        # 5) Pas de fuite xhash entre folds enregistrés
        xhash = [hashlib.sha1(X[i].tobytes()).hexdigest() for i in range(n)]
        def _check_no_leak(split_file):
            with open(split_file, 'r', encoding='utf-8') as f:
                js = json.load(f)
            ds_hash = js.get('dataset_hash')
            if ds_hash and ds_hash != info.get('dataset_hash'):
                raise AssertionError(f"dataset_hash mismatch pour {split_file}: {ds_hash} != {info.get('dataset_hash')}")
            tr = set(js['train_index'])
            te = set(js['test_index'])
            tr_x = {xhash[i] for i in tr}
            te_x = {xhash[i] for i in te}
            inter = tr_x & te_x
            return len(inter)
        splits_dir = os.path.join(base, 'splits')
        for name in ['scaffold_split','cluster_split_t06','cluster_split_t07']:
            fpath = os.path.join(splits_dir, f"{name}.json")
            assert os.path.exists(fpath), f"Split manquant: {fpath}"
            leaks = _check_no_leak(fpath)
            assert leaks == 0, f"Fuite xhash détectée sur {name}: {leaks}"
        msg = f"Vérifs express OK | n={n}, p={p}"
        print(msg)
        try:
            self.logger.log(msg)
        except Exception:
            pass


def main():
    """
    Fonction principale pour exécuter la préparation du dataset
    """
    # Initialiser le préparateur
    preparator = ChEMBLDatasetPreparator(
        target_chembl_id="CHEMBL279",  # Cyclooxygenase-2 (COX-2)
        output_dir="data"
    )
    
    # Préparer le dataset
    X, y_reg, y_clf, final_df, info = preparator.prepare_dataset(limit=2000)
    
    # Lancer le pipeline ML (régression + classification calibrée) avec split par scaffold
    results = preparator.run_ml_pipeline(X, y_reg, y_clf, final_df['standard_smiles'].tolist())
    
    print("\nPréparation et entraînement terminés avec succès!")
    print("Fichiers générés:")
    print("- data/X_features.npy")
    print("- data/y_labels.npy (binaire 6.5)")
    print("- data/y_reg.npy (pIC50)")
    print("- data/y_labels_3class.npy (0/1/2)")
    print("- data/dataset_info.pkl")
    print("- data/chembl_dataset_full.csv")

    # Vérifications express post-exécution
    expected_n = 1073 if os.getenv("LOCK_B_1073") == "1" else None
    preparator.verify_outputs(expected_n=expected_n)


if __name__ == "__main__":
    main()
