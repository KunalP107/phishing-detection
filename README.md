# Phishing Email Detection: TF-IDF + SVM vs DistilBERT

ICT6001 Applied Project, APIC, Trimester 2 2026 · Kunal Parmar, Anurag Ganjare, Samir Karki

## Structure
| Path | Purpose | Owner |
|---|---|---|
| `src/preprocessing.py` | Load, clean, deduplicate, split, leakage checks | Anurag |
| `src/obfuscation.py` | Attack simulation for the robustness test | Anurag |
| `src/evaluation.py` | Shared metrics, McNemar's test, latency | Kunal |
| `src/run_log.py` | Experiment run log | Samir |
| `tests/` | pytest unit tests for preprocessing | Anurag |
| `notebooks/01_preprocess.ipynb` | Stage 1: data preparation | Anurag |
| `notebooks/02_svm.ipynb` | Stage 2: TF-IDF + SVM | Samir |
| `notebooks/03_distilbert.ipynb` | Stage 3: DistilBERT fine-tuning (Colab T4 GPU) | Kunal |
| `notebooks/04_evaluate.ipynb` | Stage 4: test-set evaluation and analysis | Kunal, Samir, Anurag |

## Running
1. Copy this repository into Google Drive as `MyDrive/phishing_project/` and put the six Kaggle CSVs in `phishing_project/data/`.
2. Open the notebooks in Google Colab and run them in order 01 → 04. Use a T4 GPU runtime for 03 and 04.
3. Outputs (splits, models, figures, run log) are written to `phishing_project/outputs/`.

Dataset: Phishing Email Dataset (Alam, 2024), Kaggle. Seed 42 throughout.
