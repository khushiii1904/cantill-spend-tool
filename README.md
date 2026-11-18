# Spend cube tool (Cantill.ai Data & ML Intern, Round 1)

Takes raw purchase files + a category tree, and writes one row per input record: include flag + exclusion code,
supplier key, item key, net spend in the reporting currency, l1..l4 category, confidence.
A new company needs a new **config file**, not new code.

## Setup
```
pip install -r requirements.txt
# put the task data in data/company_A and data/company_B  (data/ is git-ignored - never commit it)
```
Optional: `pip install sentence-transformers` and set `embedding_model: paraphrase-multilingual-MiniLM-L12-v2` in a config
(adds multilingual semantic similarity; falls back to TF-IDF automatically if not installed). Not required.

## The one command
```
python run.py --config configs/A.yaml --name yourname            # -> output/yourname__A.csv
python run.py --config configs/B.yaml --name yourname --suffix 2 # -> output/yourname__B2.csv
python run.py --config configs/C.yaml --name yourname            # -> output/yourname__C.csv
```
Useful flags: `--no-labels` (zero-label mode, ignores labels files), `--eval` (score against labels),
`--set file.path=data/new.csv` (point a config at new files without editing it), `--out folder`.

## New company (no labels)
```
python -m src.spend.suggest_config data/company_C/<invoices>.csv data/company_C/<tree>.csv > configs/C.yaml
```
Read the draft, fix column roles / status values / gross-vs-net, then run. Config fields are documented in `configs/A.yaml` (two systems)
and `configs/B.yaml` (one system, FX).

## Layout
- `src/spend/util.py` numbers, VAT, dates, PO and name normalisers
- `src/spend/two_system.py` company with two systems (match, count once)  - `single_system.py` one file, FX, exclusions
- `src/spend/suppliers.py`, `items.py` de-duplication  - `exclusions.py` rules  - `classifier.py` tree classifier
- `src/spend/evaluate.py`, `holdout.py` dev scoring tools - `tests/` small unit tests
