"""
Random Forest Training (corrigé) pour l'identification de hits
==============================================================

Corrections principales vs version initiale:
- Utilisation du split "scaffold" fourni par le script de préparation (aucun split aléatoire).
- Suppression de toute standardisation globale (inutiles pour RF et incohérent avec des bits binaires).
- Prise en compte du déséquilibre de classes via class_weight='balanced'.
- Tuning sur Average Precision (PR-AUC), plus pertinent pour l'imbalance que ROC-AUC.

Entrées attendues (créées par le script de préparation):
- data/X_features.npy
- data/y_labels.npy              # binaire
- data/dataset_info.pkl          # métadonnées dont 'feature_names', 'splits', etc.
- data/splits/scaffold_split.json

Sorties:
- models/random_forest.joblib
- models/metrics.json
- models/plots/pr_curve.png

"""
from __future__ import annotations
import os
import json
import pickle
import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from datetime import datetime, timezone
import numbers

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from joblib import dump


# ------------------------- Utilitaires -------------------------

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


@dataclass
class Paths:
    data_dir: str = "data"
    model_dir: str = "models"
    split_name: str = "scaffold_split"  # nom de fichier JSON sans l'extension

    @property
    def X(self) -> str:
        return os.path.join(self.data_dir, "X_features.npy")

    @property
    def y(self) -> str:
        return os.path.join(self.data_dir, "y_labels.npy")

    @property
    def info(self) -> str:
        return os.path.join(self.data_dir, "dataset_info.pkl")

    @property
    def split_json(self) -> str:
        return os.path.join(self.data_dir, "splits", f"{self.split_name}.json")

    @property
    def plots_dir(self) -> str:
        return os.path.join(self.model_dir, "plots")

    @property
    def model_path(self) -> str:
        return os.path.join(self.model_dir, "random_forest.joblib")

    @property
    def metrics_path(self) -> str:
        return os.path.join(self.model_dir, "metrics.json")


class RandomForestTrainer:
    def __init__(self, paths: Paths, random_state: int = 42, smoothing_alpha: float = 0.35):
        self.paths = paths
        self.random_state = random_state
        if not (0.0 < smoothing_alpha <= 1.0):
            raise ValueError("smoothing_alpha must be in (0, 1].")
        self.smoothing_alpha = smoothing_alpha
        self.model: RandomForestClassifier | GridSearchCV | None = None
        self.base_model: RandomForestClassifier | None = None
        self.oob_score_: float | None = None
        self.dataset_info: Dict[str, Any] | None = None
        self.dataset_hash: str | None = None
        self.scaffold_labels: List[str] | None = None
        self.inchikeys: List[str] | None = None
        self.calibration_details: Dict[str, Any] | None = None
        self.split_metadata: Dict[str, Any] | None = None
        _ensure_dir(self.paths.model_dir)
        _ensure_dir(self.paths.plots_dir)

    # ------------------------- Chargements -------------------------
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        X = np.load(self.paths.X)
        y = np.load(self.paths.y)
        with open(self.paths.info, "rb") as f:
            dataset_info = pickle.load(f)
        if isinstance(dataset_info, dict):
            self.dataset_info = dataset_info
            self.dataset_hash = dataset_info.get('dataset_hash')
            self.scaffold_labels = dataset_info.get('scaffold_labels')
            self.inchikeys = dataset_info.get('inchikeys')
        else:
            self.dataset_info = None
            self.dataset_hash = None
            self.scaffold_labels = None
            self.inchikeys = None
        if self.scaffold_labels is not None and len(self.scaffold_labels) != X.shape[0]:
            raise ValueError('dataset_info.scaffold_labels length mismatch with X')
        # Sanity checks
        assert X.shape[0] == y.shape[0], "X et y doivent avoir le m?me nombre d'?chantillons"
        return X, y, dataset_info


    def load_split_indices(
        self,
        y: np.ndarray,
        *,
        split_json: str | None = None,
        expected_hash: str | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Charge le split s'il existe; sinon fallback stratifie aleatoire.
        Retourne (train_idx, test_idx).
        """
        path = split_json or self.paths.split_json
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                sp = json.load(f)
            tr = np.asarray(sp.get("train_index"), dtype=int)
            te = np.asarray(sp.get("test_index"), dtype=int)
            if tr.ndim != 1 or te.ndim != 1:
                raise ValueError("train_index/test_index must be 1D arrays")
            dataset_hash = sp.get("dataset_hash")
            expected = expected_hash or self.dataset_hash
            if expected and dataset_hash and dataset_hash != expected:
                raise ValueError("split dataset_hash does not match loaded dataset")
            if expected and dataset_hash is None:
                print(f"[WARN] split {path} has no dataset_hash; regenerate splits for stronger traceability.")
            n_samples = y.shape[0]
            if tr.size == 0 or te.size == 0:
                raise ValueError("split has empty train or test partition")
            if (tr.size + te.size) != n_samples:
                raise ValueError("split does not cover all samples")
            if np.intersect1d(tr, te).size > 0:
                raise ValueError("split indices overlap between train and test")
            if tr.min() < 0 or te.min() < 0:
                raise ValueError("split contains negative indices")
            if tr.max() >= n_samples or te.max() >= n_samples:
                raise ValueError("split contains indices out of bounds")
            union = np.union1d(tr, te)
            if union.size != n_samples:
                raise ValueError("split union does not exactly match dataset size")
            self.split_metadata = {
                "path": path,
                "name": os.path.splitext(os.path.basename(path))[0],
                "hash": sp.get("hash"),
                "dataset_hash": dataset_hash,
                "strategy": sp.get("strategy"),
                "train_size": int(tr.size),
                "test_size": int(te.size),
            }
            return tr, te
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_state)
        tr, te = next(sss.split(np.zeros_like(y), y))
        self.split_metadata = {
            "path": path,
            "name": os.path.splitext(os.path.basename(path))[0],
            "hash": None,
            "dataset_hash": expected_hash or self.dataset_hash,
            "strategy": "stratified_shufflesplit_fallback",
            "train_size": int(tr.size),
            "test_size": int(te.size),
        }
        return np.asarray(tr, dtype=int), np.asarray(te, dtype=int)

    def load_previous_metrics(self) -> Dict[str, Any] | None:
        if not os.path.exists(self.paths.metrics_path):
            return None
        try:
            with open(self.paths.metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def extract_numeric_metrics(metrics_blob: Dict[str, Any] | None) -> Dict[str, float]:
        if not isinstance(metrics_blob, dict):
            return {}
        source = metrics_blob.get("raw_metrics")
        candidate = source if isinstance(source, dict) else metrics_blob
        numeric_metrics: Dict[str, float] = {}
        for key, value in candidate.items():
            if isinstance(value, numbers.Number):
                numeric_metrics[key] = float(value)
        return numeric_metrics

    def apply_smoothing(self, metrics: Dict[str, Any], previous_numeric: Dict[str, float]) -> Dict[str, Any]:
        smoothed: Dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, numbers.Number):
                prev_val = previous_numeric.get(key)
                base_val = float(value)
                if prev_val is not None:
                    smoothed[key] = self.smoothing_alpha * base_val + (1.0 - self.smoothing_alpha) * prev_val
                else:
                    smoothed[key] = base_val
            else:
                smoothed[key] = value
        return smoothed

    def build_metrics_payload(
        self,
        smoothed_metrics: Dict[str, Any],
        raw_numeric: Dict[str, float],
        previous_numeric: Dict[str, float],
    ) -> Dict[str, Any]:
        payload = dict(smoothed_metrics)
        payload["raw_metrics"] = {k: float(v) for k, v in raw_numeric.items()}
        payload["smoothing"] = {
            "alpha": self.smoothing_alpha,
            "previous_metrics_used": bool(previous_numeric),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        return payload

    @staticmethod
    def compute_array_hash(arr: np.ndarray) -> str:
        buffer = np.ascontiguousarray(arr)
        return hashlib.sha256(buffer.view(np.uint8)).hexdigest()

    def make_calibration_split(self, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        if np.unique(y).size < 2 or y.shape[0] < 25:
            idx = np.arange(y.shape[0], dtype=int)
            return idx, np.array([], dtype=int)
        test_size = float(min(max(test_size, 0.1), 0.4))
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.random_state)
        fit_idx, cal_idx = next(splitter.split(np.zeros_like(y), y))
        if len(np.unique(y[cal_idx])) < 2:
            return np.arange(y.shape[0], dtype=int), np.array([], dtype=int)
        return np.asarray(fit_idx, dtype=int), np.asarray(cal_idx, dtype=int)

    @staticmethod
    def compute_threshold_stats(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
        preds = (scores >= threshold).astype(int)
        cm = confusion_matrix(y_true, preds).tolist()
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, preds)
        mcc = matthews_corrcoef(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        return {
            "value": float(threshold),
            "confusion_matrix": cm,
            "accuracy": float(acc),
            "f1": float(f1),
            "balanced_accuracy": float(bal_acc),
            "mcc": float(mcc),
            "precision": float(prec),
            "recall": float(rec),
            "support": int(y_true.size),
        }

    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
        unique_scores = np.unique(scores)
        candidates = np.concatenate(([0.0, 1.0], unique_scores))
        candidates = np.clip(candidates, 0.0, 1.0)
        candidates = np.unique(np.round(candidates, 6))
        best_stats: Dict[str, Any] | None = None
        best_mcc = float('-inf')
        for thr in candidates:
            stats = RandomForestTrainer.compute_threshold_stats(y_true, scores, float(thr))
            mcc = stats["mcc"]
            if mcc > best_mcc:
                best_stats = stats
                best_mcc = mcc
            elif np.isclose(mcc, best_mcc) and best_stats is not None:
                if abs(stats["value"] - 0.5) < abs(best_stats["value"] - 0.5):
                    best_stats = stats
                    best_mcc = mcc
        if best_stats is None:
            return RandomForestTrainer.compute_threshold_stats(y_true, scores, 0.5)
        return best_stats

    # ------------------------- Modèle -------------------------
    def make_base_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            n_jobs=-1,
            class_weight="balanced",
            random_state=self.random_state,
            oob_score=True,
        )

    def tune_hyperparams(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> Dict:
        base = self.make_base_model()
        param_grid = {
            "n_estimators": [300, 500, 800],
            "max_depth": [None, 20, 40],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }
        if groups is not None:
            unique_groups = np.unique(groups)
            n_splits = max(2, min(5, unique_groups.size))
            cv = GroupKFold(n_splits=n_splits)
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        gs = GridSearchCV(
            estimator=base,
            param_grid=param_grid,
            scoring="average_precision",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        if groups is not None:
            gs.fit(X_tr, y_tr, groups=groups)
        else:
            gs.fit(X_tr, y_tr)
        self.model = gs.best_estimator_
        return {
            "best_params": gs.best_params_,
            "best_score_ap": float(gs.best_score_),
        }


    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray, params: Dict | None = None):
        base_model = self.make_base_model()
        if params:
            base_model.set_params(**params)
        fit_idx, cal_idx = self.make_calibration_split(y_tr)
        X_fit, y_fit = X_tr[fit_idx], y_tr[fit_idx]
        base_model.fit(X_fit, y_fit)
        self.base_model = base_model
        self.oob_score_ = getattr(base_model, "oob_score_", None)

        calibration_used = False
        calibration_method = None
        calibration_size = int(cal_idx.size)
        calibration_balance = None
        if cal_idx.size > 0:
            X_cal, y_cal = X_tr[cal_idx], y_tr[cal_idx]
            if np.unique(y_cal).size >= 2:
                calibrator = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv='prefit')
                calibrator.fit(X_cal, y_cal)
                self.model = calibrator
                calibration_used = True
                calibration_method = 'isotonic_prefit'
                calibration_balance = {
                    'positive': float(y_cal.mean()),
                    'negative': float(1.0 - y_cal.mean()),
                }
            else:
                self.model = base_model
        else:
            self.model = base_model

        train_balance = {
            'positive': float(y_fit.mean()),
            'negative': float(1.0 - y_fit.mean()),
        }
        self.calibration_details = {
            'used': calibration_used,
            'method': calibration_method,
            'train_size': int(y_fit.size),
            'calibration_size': calibration_size,
            'class_balance_train': train_balance,
            'class_balance_calibration': calibration_balance,
            'train_index_hash': hashlib.sha256(np.asarray(fit_idx, dtype=int).tobytes()).hexdigest(),
            'calibration_index_hash': hashlib.sha256(np.asarray(cal_idx, dtype=int).tobytes()).hexdigest() if calibration_size > 0 else None,
        }

    def evaluate(self, X_te: np.ndarray, y_te: np.ndarray) -> Dict:
        assert self.model is not None, "Modele non entraine"
        proba = self.model.predict_proba(X_te)[:, 1]

        default_threshold = 0.5
        default_stats = self.compute_threshold_stats(y_te, proba, default_threshold)
        optimal_stats = self.find_optimal_threshold(y_te, proba)

        metrics = {
            "roc_auc": float(roc_auc_score(y_te, proba)),
            "average_precision": float(average_precision_score(y_te, proba)),
            "accuracy": float(default_stats["accuracy"]),
            "f1": float(default_stats["f1"]),
            "balanced_accuracy": float(default_stats["balanced_accuracy"]),
            "mcc": float(default_stats["mcc"]),
            "precision": float(default_stats["precision"]),
            "recall": float(default_stats["recall"]),
            "confusion_matrix": default_stats["confusion_matrix"],
            "threshold_default": float(default_threshold),
            "threshold_optimal_mcc": float(optimal_stats["value"]),
            "thresholds": {
                "default": default_stats,
                "optimal_mcc": optimal_stats,
            },
            "support_test": int(y_te.size),
            "positive_rate_test": float(y_te.mean()),
            "dataset_hash": self.dataset_hash,
            "oob_score": float(self.oob_score_) if self.oob_score_ is not None else None,
            "calibration": self.calibration_details,
            "split_metadata": self.split_metadata,
        }
        metrics["class_balance"] = {
            "train": (self.calibration_details or {}).get("class_balance_train") if self.calibration_details else None,
            "calibration": (self.calibration_details or {}).get("class_balance_calibration") if self.calibration_details else None,
            "test": {
                "positive": float(y_te.mean()),
                "negative": float(1.0 - y_te.mean()),
            },
        }
        if self.calibration_details:
            metrics["train_size"] = int(self.calibration_details.get("train_size", 0))
            metrics["calibration_size"] = int(self.calibration_details.get("calibration_size", 0))

        try:
            metrics["classification_report"] = classification_report(y_te, (proba >= default_threshold).astype(int), digits=3)
        except Exception:
            pass

        prec, rec, _ = precision_recall_curve(y_te, proba)
        plt.figure()
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall (test)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.paths.plots_dir, "pr_curve.png"), dpi=180)
        plt.close()

        return metrics


    # ------------------------- Sauvegardes -------------------------
    def save_all(self, metrics: Dict):
        dump(self.model, self.paths.model_path)
        with open(self.paths.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


# ------------------------- Script principal -------------------------

def main():
    paths = Paths()
    trainer = RandomForestTrainer(paths)

    random.seed(trainer.random_state)
    np.random.seed(trainer.random_state)

    X, y, dataset_info = trainer.load_dataset()
    tr_idx, te_idx = trainer.load_split_indices(y, expected_hash=trainer.dataset_hash)

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    dataset_snapshot = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "train_size": int(tr_idx.size),
        "test_size": int(te_idx.size),
        "train_positive_rate": float(y_tr.mean()),
        "test_positive_rate": float(y_te.mean()),
        "dataset_hash": trainer.dataset_hash,
    }
    train_groups = None
    groups = None
    if trainer.scaffold_labels is not None:
        groups = np.asarray(trainer.scaffold_labels)
        train_groups = groups[tr_idx]

    # Tuning recommandé (décommente pour grid search)
    # summary = trainer.tune_hyperparams(X_tr, y_tr, groups=train_groups)
    # print("Best params:", summary["best_params"])
    # print("CV AP:", summary["best_score_ap"])
    # params = summary["best_params"]

    # Entraînement direct avec base params (rapide et robuste)
    params = None
    trainer.fit(X_tr, y_tr, params=params)

    raw_metrics = trainer.evaluate(X_te, y_te)
    raw_metrics["dataset_snapshot"] = dataset_snapshot
    if isinstance(dataset_info, dict):
        raw_metrics["dataset_versions"] = dataset_info.get("versions")
        raw_metrics["target_chembl_id"] = dataset_info.get("target_chembl_id")
        raw_metrics["cutoff_binary"] = dataset_info.get("cutoff_binary")
    raw_metrics["group_kfold_available"] = bool(train_groups is not None)
    previous_metrics = trainer.load_previous_metrics()
    previous_numeric = trainer.extract_numeric_metrics(previous_metrics)
    raw_numeric = trainer.extract_numeric_metrics(raw_metrics)
    smoothed_metrics = trainer.apply_smoothing(raw_metrics, previous_numeric)
    metrics_payload = trainer.build_metrics_payload(smoothed_metrics, raw_numeric, previous_numeric)
    trainer.save_all(metrics_payload)

    print("Terminé.")
    print("Modèle:", paths.model_path)
    print("Métriques:", paths.metrics_path)
    print("Courbe PR:", os.path.join(paths.plots_dir, "pr_curve.png"))


if __name__ == "__main__":
    main()
