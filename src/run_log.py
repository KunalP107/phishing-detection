"""Experiment run log: one row per run with its settings and results."""
import json
import os
from datetime import datetime

import pandas as pd


def log_run(log_path, stage, model, params, results, owner):
    row = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'stage': stage, 'model': model,
           'owner': owner, 'params': json.dumps(params, default=str), 'results': json.dumps(results, default=str)}
    df = pd.DataFrame([row])
    df.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
    return row
