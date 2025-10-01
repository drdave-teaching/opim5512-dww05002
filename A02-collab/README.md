# A02 — Buddy Collab (Decision Tree)

**Goal:** practice issues → branches → PRs while building a tiny ML pipeline together.

## Folder
```
A02-collab/
  data/   # tables, metrics
  figs/   # charts
```
Keep every PR tiny: one artifact (CSV/JSON/PNG) + 2–3 sentence note in this README.

## Ping-pong plan (Lap 1)
1. **Load** → save `data/raw.csv` + 2–3 sentence note about shape/columns.
2. **Clean** → save `data/clean.csv` (NA/outliers/renames) + note decisions.
3. **Model (DecisionTreeRegressor)** → `data/metrics.json`, `data/preds_test.csv`, `figs/residuals.png`, `figs/feat_importance.png`.
4. **Evaluate** → `figs/pred_vs_actual.png` + add R² / RMSE to this README.
5. **Write-up v1** → 150–200 words summarizing what you tried and learned.

## Optional Lap 2 (upgrade)
- **Feature engineering** → `data/clean_fe.csv` (log/one-hot/standardize).  
- **Hyperparam search** (max_depth grid) → update metrics + note.  
- **Error analysis** → `data/error_by_decile.csv`, `figs/error_by_decile.png`.  
- **Write-up v2** → what improved, what didn't, next steps.

## How to run (each step is a command)
From repo root (after `pip install -r requirements.txt`):
```bash
python src/pipeline.py load
python src/pipeline.py clean
python src/pipeline.py model
python src/pipeline.py evaluate
```
Artifacts will appear under `A02-collab/data` and `A02-collab/figs`.

## Notes (update as you go)
- Load:
- Clean:
- Model:
- Evaluate: