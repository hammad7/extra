# Technical Design Doc — Solution A (NAIVE MODEL): Predictive Maintenance & RUL

**System:** 24×7 telecom-tower power assurance · **Component:** battery-bank failure prognosis
**Model:** Two independent XGBoost models (classifier + regressor) · **Code:** `baseline/gen_data.py`, `baseline/train.py`

---

## 1. Problem Statement

Tower sites stay powered through a prioritised chain — **grid → VRLA → Li-ion → DG (→ solar)**. The **battery bank is the #1 failure point** and a major capex driver: when a battery silently loses health, the next grid outage drops the site (SLA breach) or forces an emergency DG run. Today, replacement is **reactive or fixed-interval**, causing both *surprise outages* and *premature swaps of healthy batteries*.

**Objective:** From always-on telemetry, raise an **actionable alarm** (fail within 7 days) and estimate **Remaining Useful Life (RUL)** so swaps happen on planned visits, not emergencies.

---

## 2. Available Signals (Features)

The full set of signals available from the site power chain is below. State-of-Health (SoH) is the *hidden* driver — the model must infer it from cheap, always-on signals. The **Used in model** column marks the 8 features the naive model actually consumes (and their column name in the dataset).

| Source | Signal | Used in model |
|--------|--------|:---:|
| **Grid / Mains** | 3-phase voltage | — |
| | current | — |
| | frequency | — |
| | phase availability | — |
| | mains-fail / restore events | ✅ `mains_fail_count` |
| | EB-hours | — |
| | power factor | — |
| | kWh import | — |
| **Rectifier / SMPS** | DC output (−48 V) | — |
| | load current | — |
| | per-module current | — |
| | module temperature | — |
| | efficiency | — |
| | active-module count | — |
| | fault flags | — |
| **VRLA & Li-ion (BMS)** | terminal voltage | ✅ `term_voltage` |
| | per-cell voltage | — |
| | string current | — |
| | SoC | — |
| | **SoH** | ❌ excluded (hidden ground-truth label proxy) |
| | internal resistance / impedance | ✅ `internal_res_mohm` |
| | per-cell temperature | — |
| | charge/discharge cycles | ✅ `cycles` |
| | depth-of-discharge | ✅ `dod` |
| | backup-minutes | ✅ `backup_min` |
| | cell-balancing status (Li-ion) | — |
| | BMS alarms (Li-ion) | — |
| **Battery asset** | battery age (days) | ✅ `age_days` |
| | chemistry / install batch | — |
| **Diesel Generator** | fuel level | — |
| | consumption rate | — |
| | run-hours | — |
| | start/stop count | — |
| | oil pressure | — |
| | coolant temp | — |
| | RPM | — |
| | output V / I / Hz | — |
| | load % | — |
| | starter-battery V | — |
| | exhaust temp | — |
| | vibration | — |
| | auto/manual mode | — |
| **Solar PV** | panel V / I | — |
| | MPPT output | — |
| | irradiance | — |
| | panel temp | — |
| | daily generation | — |
| | inverter status | — |
| **Site / Environment** | ambient & cabinet temperature | ✅ `avg_temp_c` |
| | humidity | — |
| | fan speed | — |
| | AC / free-cooling status | — |
| | door / intrusion | — |
| | smoke / fire | — |
| | water ingress | — |
| | total BTS load | — |
| **Contextual / derived** | weather forecast | — |
| | historical grid reliability (per site) | — |
| | time-of-day / season | — |
| | urban / rural class | — |
| | fuel price | — |
| | SLA tier | — |

**Model feature vector (8):** `age_days, internal_res_mohm, backup_min, cycles, dod, avg_temp_c, term_voltage, mains_fail_count`.

---

## 3. Example — how the data looks

One row = one site on one day. Below is a single battery's trajectory (site 7) at three points. `soh` is shown for illustration only — it is **not** a model input; the labels `rul` and `fail_within_7d` are derived from it.

| site_id | day | age_days | internal_res_mohm | term_voltage | backup_min | cycles | dod | avg_temp_c | mains_fail_count | _soh (hidden)_ | **rul** | **fail_within_7d** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 7 | 0 | 677 | 4.35 | 52.86 | 191.9 | 1.27 | 0.64 | 35.1 | 1 | 0.919 | 245 | 0 |
| 7 | 238 | 915 | 5.89 | 51.77 | 160.3 | 2.15 | 0.49 | 29.1 | 2 | 0.706 | 7 | 1 |
| 7 | 245 | 922 | 5.49 | 51.60 | 157.0 | 1.59 | 0.36 | 26.0 | 4 | 0.699 | 0 | 1 |

Read top-to-bottom: as the (hidden) SoH falls 0.92 → 0.70, **internal resistance rises** (4.35 → 5.5–5.9 mΩ), **terminal voltage and backup-minutes drop**, and `rul` counts down to 0 — exactly the signal the model learns. `fail_within_7d` flips to 1 once `rul ≤ 7`.

---

## 4. Model Details (exactly how `baseline/train.py` works)

**Framing — two independent supervised models on the same tabular telemetry.** Gradient-boosted trees (XGBoost) are the SOTA workhorse for tabular sensor data, so we use them directly:

1. **Alarm — classifier** `XGBClassifier` predicting `fail_within_7d` (label = `rul ≤ 7`).
   - Output `P(fail within 7 days)` = the actionable score.
   - Params: `n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="aucpr"`.
   - **Class imbalance (~3% positives)** handled with `scale_pos_weight = #neg / #pos`.
2. **RUL — regressor** `XGBRegressor` predicting days-to-failure.
   - Params: `n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9`.
   - Trained **only on uncensored sites** (`rul ≤ 365`).

**Key design choices:**
- **Leakage-safe split** — `site_split()` splits **by `site_id`** (80/20); all rows of a site stay on one side. A row-level split would leak a site's trajectory across train/test.
- **`soh` excluded** — it is the hidden ground-truth driver; the models must infer health from the cheap, always-on signals listed above.
- **Operating point** — the classifier is tuned for recall, so a threshold sweep (`0.5 … 0.95`) exposes the precision/recall frontier for ops to pick a cut-off.

**Results (held-out sites, seed=0):**

```
data: rows=98,616  sites=400  failure-events=352  alarm-positives=2,816 (2.9%)

Alarm: fail within 7 days     ROC-AUC 0.956 | PR-AUC 0.350
  thr   precision  recall    TP    FP    FN
  0.50     0.21     0.86    479  1752    81   <- recall mode: catch ~86%
  0.80     0.29     0.63    355   858   205
  0.90     0.35     0.40    222   415   338
  0.95     0.47     0.17     95   109   465   <- precision mode
RUL regression:  MAE 20.6 d overall | 14.2 d when <=30 d to failure
Top signals (classifier): internal_res (0.67) > term_voltage (0.14)
  > backup_min (0.06) > avg_temp (0.04) > age (0.03)
```

`internal_resistance` dominating importance is a physics sanity check — exactly how a degrading battery behaves.

**Known limitation:** the regressor **drops censored sites** (`rul == 9999`, batteries that had not failed within the year). Those survivors are the *healthiest* units, so excluding them **biases RUL estimates low** and wastes data. This is the main reason to graduate to a survival model (see §6).

---

## 5. Run

```bash
cd baseline
pip install -r ../requirements.txt
python gen_data.py     # -> telemetry.parquet
python train.py        # -> metrics above
```

To use real data, replace `gen_data.py` with a loader mapping your columns to the 8 feature names + `rul`/`fail_within_7d`; `train.py` runs unchanged.

---

## 6. Recommendations & Roadmap

### Recommended upgrade (advanced model): Weibull AFT survival
Replace the two naive models with a single **XGBoost Accelerated-Failure-Time (AFT)** model using a **Weibull** (`extreme`) error distribution. It:
- **keeps right-censored sites** via interval labels `[t, +inf)` — fixes the low-RUL bias of the naive regressor;
- captures the **Weibull wear-out** shape (shape `k = 1/sigma > 1`);
- emits **both** the 7-day risk *and* median RUL from one survival curve;
- adds a **Cox PH** companion for interpretable hazard ratios.

Measured gains over this naive baseline: discrimination **ROC-AUC 0.956 -> 0.961**, near-EoL **RUL MAE 14.2 -> 11.6 days**, Cox HRs physically sensible (internal resistance ~ 3x, temperature ~ 2x per +1 SD).

### Solution B — Agentic Energy Orchestration & Diesel Optimisation
- **Problem:** choose *which source runs when* (grid <-> battery <-> DG <-> solar) and how to cycle batteries to minimise diesel/cost/CO2 without breaching backup-time SLA.
- **Solution (3 layers):** (1) **forecast** grid-outage probability + site load (LSTM/Temporal-Fusion); (2) **optimise** with a **MILP** (or RL) for source dispatch + battery charge schedule under SLA and SoH-preserving constraints; (3) a **tool-using agent** executes via the site controller/SCADA with human-in-loop on high-impact actions; validate on a digital twin first.
- **KPI:** diesel litres down, CO2 down, backup-time adherence.

### Solution C — Anomaly Detection: Fuel Theft + Outage Prediction with Auto-Dispatch
- **Problem:** chronic diesel pilferage losses; need proactive (not post-mortem) outage response.
- **Solution:** *Fuel theft* — model expected consumption from DG run-hours x load; flag negative residual drops with **Isolation Forest / robust rules**, refine with confirmed-case labels. *Outage prediction* — classifier on SoC-decay + mains-outage duration + DG fuel/health -> **time-to-site-down ETA**. An **agent** auto-creates/prioritises NOC tickets and optimises **crew routing** by ETA and geography.
- **KPI:** fuel-loss value down, MTTR down, outage lead-time up.
