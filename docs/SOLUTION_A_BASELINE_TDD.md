# Technical Design Doc — Solution A (NAIVE BASELINE): Predictive Maintenance & RUL

**System:** 24×7 telecom-tower power assurance · **Component:** battery-bank failure prognosis
**Status:** Baseline prototype (`baseline/gen_data.py`, `baseline/train.py`) · **Owner:** Power-AI
**Relation:** This is the *first-cut* baseline. The improved survival model is documented in
[`SOLUTION_A_TDD.md`](./SOLUTION_A_TDD.md).

---

## 1. Problem Statement

Tower sites stay powered through a prioritised chain — **grid → VRLA → Li-ion → DG (→ solar)**. The **battery bank is the #1 failure point** and a major capex driver: when a battery silently loses health, the next grid outage drops the site (SLA breach) or forces an emergency DG run. Today, replacement is **reactive or fixed-interval**, causing both *surprise outages* and *premature swaps of healthy batteries*.

**Objective:** From always-on telemetry, raise an **actionable alarm** (fail within 7 days) and estimate **Remaining Useful Life (RUL)** so swaps happen on planned visits, not emergencies.

---

## 2. Available Signals (Features)

State-of-Health (SoH) is the *hidden* driver; the model infers it from cheap, always-measured signals.

| Group | Signals | Used in model |
|-------|---------|:---:|
| **Battery / BMS** | internal resistance (mΩ), terminal voltage (−48 V bus), backup-minutes, charge/discharge cycles, depth-of-discharge (DoD), SoC, SoH | ✅ res, voltage, backup, cycles, DoD |
| **Asset** | battery age (days), chemistry, install batch | ✅ age |
| **Environment** | cabinet/ambient temperature, humidity, fan speed | ✅ avg temp |
| **Grid context** | mains-fail count/day, EB-hours, outage duration | ✅ mains-fail count |
| **Excluded on purpose** | **SoH** (ground-truth label proxy) | ❌ leakage |

**Model feature vector (8):** `age_days, internal_res_mohm, backup_min, cycles, dod, avg_temp_c, term_voltage, mains_fail_count`.

---

## 3. Model Details (exactly how `baseline/train.py` works)

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
- **`soh` excluded** — it is the hidden ground-truth driver; the models must infer health from the cheap, always-on signals.
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
RUL regression:  MAE 20.6 d overall | 14.2 d when ≤30 d to failure
Top signals (classifier): internal_res (0.67) > term_voltage (0.14)
  > backup_min (0.06) > avg_temp (0.04) > age (0.03)
```

`internal_resistance` dominating importance is a physics sanity check — exactly how a degrading battery behaves.

---

## 4. Known Limitation → why we moved to a survival model

The regressor **drops censored sites** (`rul == 9999`, i.e. batteries that had *not* failed within the year). Those survivors are the **healthiest** units, so excluding them **biases RUL estimates low** and wastes data. The classifier also needs a separately tuned threshold, and the two models can disagree.

**Fix (see [`SOLUTION_A_TDD.md`](./SOLUTION_A_TDD.md)):** a single **XGBoost Accelerated-Failure-Time (AFT, Weibull)** survival model that keeps censored data via interval labels (`[t, +∞)`), captures the Weibull wear-out shape, and emits *both* the 7-day risk and median RUL from one survival curve. It improves discrimination (ROC-AUC 0.956 → 0.961) and near-EoL RUL error (14.2 → 11.6 days), and adds a Cox PH companion for interpretable hazard ratios.

---

## 5. Run

```bash
cd baseline
pip install -r ../requirements.txt
python gen_data.py     # -> telemetry.parquet
python train.py        # -> metrics above
```

---

## 6. Roadmap — the other two systems (shared pipeline)

### Solution B — Agentic Energy Orchestration & Diesel Optimisation
- **Problem:** choose *which source runs when* (grid ↔ battery ↔ DG ↔ solar) and how to cycle batteries to minimise diesel/cost/CO₂ without breaching backup-time SLA.
- **Solution (3 layers):** (1) **forecast** grid-outage probability + site load (LSTM/Temporal-Fusion); (2) **optimise** with a **MILP** (or RL) for source dispatch + battery charge schedule under SLA and SoH-preserving constraints; (3) a **tool-using agent** executes via the site controller/SCADA with human-in-loop on high-impact actions; validate on a digital twin first.
- **KPI:** diesel litres ↓, CO₂ ↓, backup-time adherence.

### Solution C — Anomaly Detection: Fuel Theft + Outage Prediction with Auto-Dispatch
- **Problem:** chronic diesel pilferage losses; need proactive (not post-mortem) outage response.
- **Solution:** *Fuel theft* — model expected consumption from DG run-hours × load; flag negative residual drops with **Isolation Forest / robust rules**, refine with confirmed-case labels. *Outage prediction* — classifier on SoC-decay + mains-outage duration + DG fuel/health → **time-to-site-down ETA**. An **agent** auto-creates/prioritises NOC tickets and optimises **crew routing** by ETA and geography.
- **KPI:** fuel-loss value ↓, MTTR ↓, outage lead-time ↑.
