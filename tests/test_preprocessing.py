"""Unit tests for the preprocessing code (run with: python -m pytest -q tests)."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import check_leakage, clean_text, map_labels, prepare, stratified_split  # noqa: E402


def test_url_masking():
    assert clean_text('Verify at https://evil.example.com/login now') == 'Verify at URL now'
    assert clean_text('see www.example.org') == 'see URL'


def test_html_removed_and_whitespace_collapsed():
    assert clean_text('<p>Hello <b>there</b></p>\n\n  friend') == 'Hello there friend'


def test_empty_and_missing_bodies():
    assert clean_text(None) == ''
    assert clean_text(float('nan')) == ''
    raw = pd.DataFrame({'text': ['', '   ', 'real email'], 'label': [1, 0, 1], 'source': ['a'] * 3})
    data, steps = prepare(raw)
    assert len(data) == 1 and steps['after_empty_removed'] == 1


def test_label_mapping():
    mapped = map_labels(pd.Series(['1', 0, '0.0', 'phishing', 'ham', 'unknown']))
    assert mapped.tolist()[:5] == [1, 0, 0, 1, 0]
    assert pd.isna(mapped.iloc[5])


def test_duplicates_and_conflicting_labels_removed():
    raw = pd.DataFrame({'text': ['a b', 'a b', 'c d', 'c d'], 'label': [1, 1, 1, 0], 'source': ['x'] * 4})
    data, steps = prepare(raw)
    assert data['clean'].tolist() == ['a b']
    assert steps['after_conflicting_labels_removed'] == 2 and steps['after_deduplication'] == 1


def test_split_is_stratified_and_leak_free():
    raw = pd.DataFrame({'text': [f'email {i}' for i in range(200)], 'label': [i % 2 for i in range(200)], 'source': ['x'] * 200})
    data, _ = prepare(raw)
    train, val, test = stratified_split(data)
    assert (len(train), len(val), len(test)) == (140, 30, 30)
    assert train['label'].mean() == val['label'].mean() == test['label'].mean() == 0.5
    assert check_leakage(train, val, test) == {'train-val': 0, 'train-test': 0, 'val-test': 0}


def test_leakage_is_detected():
    df = pd.DataFrame({'clean': ['same'], 'label': [1]})
    with pytest.raises(ValueError):
        check_leakage(df, df, df.iloc[0:0])
