"""Data loading, cleaning, deduplication and stratified splitting (owner: Anurag)."""
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
SOURCES = ['CEAS_08', 'Enron', 'Ling', 'Nazario', 'Nigerian_Fraud', 'SpamAssasin']
URL_RE = re.compile(r'(https?://\S+|www\.\S+)', re.IGNORECASE)
TAG_RE = re.compile(r'<[^>]+>')


def map_labels(labels):
    """Convert raw labels to integers 1 (phishing) and 0 (legitimate). Rows with any other value become NaN."""
    mapping = {'1': 1, '0': 0, '1.0': 1, '0.0': 0, 'phishing': 1, 'spam': 1, 'legitimate': 0, 'ham': 0}
    return labels.astype(str).str.strip().str.lower().map(mapping)


def clean_text(text):
    """Remove HTML tags, replace web addresses with a URL token and collapse whitespace."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ''
    text = TAG_RE.sub(' ', str(text))
    text = URL_RE.sub(' URL ', text)
    return re.sub(r'\s+', ' ', text).strip()


def load_sources(data_dir):
    """Load the six Kaggle source files, keep subject, body and label, and tag each email with its source."""
    frames = []
    for name in SOURCES:
        files = glob.glob(os.path.join(data_dir, f'{name}*.csv'))
        if not files:
            raise FileNotFoundError(f'No CSV found for source {name} in {data_dir}')
        df = pd.read_csv(files[0], on_bad_lines='skip', low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        for col in ('subject', 'body', 'label'):
            if col not in df.columns:
                df[col] = np.nan
        df = df[['subject', 'body', 'label']].copy()
        df['source'] = name
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw['label'] = map_labels(raw['label'])
    raw['text'] = (raw['subject'].fillna('').astype(str) + ' ' + raw['body'].fillna('').astype(str)).str.strip()
    return raw


def prepare(raw):
    """Clean the text, then drop unlabelled rows, empty emails, conflicting labels and exact duplicates.
    Returns the clean data and a count after each step."""
    data = raw.dropna(subset=['label']).copy()
    data['label'] = data['label'].astype(int)
    data['clean'] = data['text'].map(clean_text)
    steps = {'loaded': len(raw), 'labelled': len(data)}
    data = data[data['clean'].str.len() > 0]
    steps['after_empty_removed'] = len(data)
    n_labels = data.groupby('clean')['label'].nunique()
    data = data[~data['clean'].isin(n_labels[n_labels > 1].index)]
    steps['after_conflicting_labels_removed'] = len(data)
    data = data.drop_duplicates('clean').reset_index(drop=True)
    steps['after_deduplication'] = len(data)
    return data, {k: int(v) for k, v in steps.items()}


def stratified_split(data, seed=SEED):
    """Stratified 70/15/15 train/validation/test split."""
    train, temp = train_test_split(data, test_size=0.30, stratify=data['label'], random_state=seed)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp['label'], random_state=seed)
    return [d.reset_index(drop=True) for d in (train, val, test)]


def check_leakage(train, val, test):
    """Raise an error if any email text appears in more than one split."""
    tr, va, te = set(train['clean']), set(val['clean']), set(test['clean'])
    overlaps = {'train-val': len(tr & va), 'train-test': len(tr & te), 'val-test': len(va & te)}
    if any(overlaps.values()):
        raise ValueError(f'Data leakage found: {overlaps}')
    return overlaps
