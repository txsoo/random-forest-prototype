import os
import json
import pickle
import numpy as np
import pytest

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DATA_DIR = os.path.abspath(DATA_DIR)


def load_dataset():
    X = np.load(os.path.join(DATA_DIR, 'X_features.npy'))
    y_bin = np.load(os.path.join(DATA_DIR, 'y_labels.npy'))
    y_reg = np.load(os.path.join(DATA_DIR, 'y_reg.npy'))
    with open(os.path.join(DATA_DIR, 'dataset_info.pkl'), 'rb') as f:
        info = pickle.load(f)
    return X, y_bin, y_reg, info


def test_shapes_and_alignment():
    X, y_bin, y_reg, info = load_dataset()
    assert X.shape[0] == y_bin.shape[0] == y_reg.shape[0] > 0
    assert X.shape[1] == len(info['feature_names'])


def test_no_leakage_in_features():
    _, _, _, info = load_dataset()
    forbidden = {'pic50', 'pic50_median', 'active', 'molecule_chembl_id', 'canonical_smiles', 'standard_smiles', 'inchikey', 'assay', 'target'}
    for name in info['feature_names']:
        for bad in forbidden:
            assert bad not in name.lower()


def test_threshold_consistency():
    _, y_bin, y_reg, info = load_dataset()
    assert info['cutoff_binary'] == 6.5
    y_from_reg = (y_reg >= 6.5).astype(int)
    assert np.array_equal(y_bin, y_from_reg)


def _read_split(name):
    with open(os.path.join(DATA_DIR, 'splits', f'{name}.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


def test_no_overlap_inchikey_between_splits():
    import pandas as pd
    df = pd.read_parquet(os.path.join(DATA_DIR, 'chembl_dataset_full.parquet'))
    ik = df['inchikey'].tolist()
    # scaffold
    sc = _read_split('scaffold_split')
    tr, te = np.array(sc['train_index']), np.array(sc['test_index'])
    assert len(set(np.array(ik)[tr]).intersection(set(np.array(ik)[te]))) == 0
    # cluster 0.6
    c06 = _read_split('cluster_split_t06')
    tr, te = np.array(c06['train_index']), np.array(c06['test_index'])
    assert len(set(np.array(ik)[tr]).intersection(set(np.array(ik)[te]))) == 0
    # cluster 0.7
    c07 = _read_split('cluster_split_t07')
    tr, te = np.array(c07['train_index']), np.array(c07['test_index'])
    assert len(set(np.array(ik)[tr]).intersection(set(np.array(ik)[te]))) == 0


def test_no_zero_variance_columns():
    X, *_ = load_dataset()
    # numerical stability: allow tiny epsilon
    vars_ = X.var(axis=0)
    assert np.all(vars_ > 0.0)


def test_morgan_binary_and_density():
    X, _, _, info = load_dataset()
    bit_idx = [i for i, n in enumerate(info['feature_names']) if n.startswith('morgan_bit_')]
    if not bit_idx:
        pytest.skip('No Morgan bits retained after curation')
    X_bits = X[:, bit_idx]
    # binary check (values 0/1 within tolerance)
    assert np.all((X_bits == 0) | (X_bits == 1))
    density = float(X_bits.mean())
    assert 0.005 <= density <= 0.05, f'density {density} out of range [0.5%, 5%]'
