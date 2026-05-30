# extra — Telecom Tower Battery Predictive Maintenance (Solution A)

A small, self-contained prototype for predicting battery-bank failures at
telecom-tower sites (grid → VRLA → Li-ion → DG power chain). It forecasts
**Remaining Useful Life (RUL)** and emits an actionable **dispatch alarm** from
cheap, always-on telemetry.

## Approach

A single **XGBoost Accelerated-Failure-Time (AFT)** model with a Weibull
(`extreme`) error distribution — gradient boosting trained on the survival
likelihood. It:

- **uses right-censored sites** (batteries that haven't failed yet) instead of
  discarding them — avoids the low-RUL bias of a plain regressor;
- captures the **Weibull wear-out** shape (shape `k = 1/σ`);
- keeps tree nonlinearity / feature interactions;
- yields a full survival curve per row → both **P(fail within 7 days)** and
  **median RUL (days)**.

A **Cox proportional-hazards** model is fit alongside purely for
interpretability (hazard ratios per signal).

> `soh` (state of health) is deliberately excluded from the features — it's the
> hidden ground-truth driver. The model must infer health from the measurable
> sensor signals (internal resistance, terminal voltage, backup minutes, temp…).

## Files

| File | Purpose |
|------|---------|
| `gen_data.py` | Simulates ~1 year of daily telemetry for 400 sites with heat/cycle-accelerated wear; writes `telemetry.parquet` with `event` + `time_to_event` survival labels. |
| `train.py` | Trains the AFT model, evaluates discrimination + RUL, prints a dispatch-rule table, AFT feature importances, and Cox hazard ratios. |

## Run

```bash
pip install -r requirements.txt
python gen_data.py     # -> telemetry.parquet
python train.py        # -> metrics
```

## Example results (held-out sites)

- Discrimination (P fail ≤ 7d): **ROC-AUC ≈ 0.96**
- RUL MAE near end-of-life (≤30d): **≈ 12 days**
- Cox hazard ratios (per +1 SD): internal resistance ≈ **3×**, temperature ≈
  **2×** (Arrhenius), higher backup-time/voltage protective — physically sane.

## Using real data

Replace `gen_data.py` with a loader that maps your columns to the same feature
names and provides `event` (1 = failed) + `time_to_event` (days). `train.py`
runs unchanged.
