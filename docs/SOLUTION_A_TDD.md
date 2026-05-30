# Technical Design Doc — Solution A: Battery Predictive Maintenance & RUL

**System:** 24×7 telecom-tower power assurance · **Component:** battery-bank failure prognosis
**Status:** Working prototype (`gen_data.py`, `train.py`) · **Owner:** Power-AI

---

## 1. Problem Statement

Tower sites stay powered through a prioritised chain — **grid → VRLA → Li-ion → DG (→ solar)**. The **battery bank is the #1 failure point** and a major capex driver: when a battery silently loses health, the next grid outage drops the site (SLA breach) or forces an emergency DG run. Today, replacement is **reactive or fixed-interval**, causing both *surprise outages* and *premature swaps of healthy batteries*.

**Objective:** From always-on telemetry, predict each battery's **Remaining Useful Life (RUL)** and emit an **actionable dispatch alarm** — early enough to swap on a planned visit, late enough to avoid wasted truck-rolls.

**Targets (prototype, held-out sites):** discrimination ROC-AUC ≈ 0.96; near-end-of-life RUL error ≈ 12 days.

---

## 2. Available Signals (Features)

Health (State-of-Health, SoH) is the *hidden* driver; the model must infer it from cheap, always-measured signals.

| Group | Signals | Used in model |
|-------|---------|:---:|
| **Battery / BMS** | internal resistance (mΩ), terminal voltage (−48 V bus), backup-minutes, charge/discharge cycles, depth-of-discharge (DoD), SoC, SoH | ✅ res, voltage, backup, cycles, DoD |
| **Asset** | battery age (days), chemistry, install batch | ✅ age |
| **Environment** | cabinet/ambient temperature, humidity, fan speed | ✅ avg temp |
| **Grid context** | mains-fail count/day, EB-hours, outage duration | ✅ mains-fail count |
| **Excluded on purpose** | **SoH** (ground-truth label proxy) | ❌ leakage |

**Model feature vector (8):** `age_days, internal_res_mohm, backup_min, cycles, dod, avg_temp_c, term_voltage, mains_fail_count`.

---

## 3. Model Details (exactly how `train.py` works)

**Framing — survival analysis, not plain regression.** Each daily row is one observation of "time from now until this battery fails." Batteries that have **not yet failed are right-censored** (we only know they lasted *at least* this long). A classifier+regressor would either discard those rows or mislabel them; we instead keep them via a survival likelihood.

**Algorithm — XGBoost Accelerated Failure Time (AFT), Weibull.** Gradient boosting trained on the AFT likelihood (`objective="survival:aft"`, `aft_loss_distribution="extreme"` = Weibull on the time scale). This keeps tree nonlinearity/interactions *and* models the Weibull wear-out shape.

**Pipeline steps:**

1. **Leakage-safe split** — `site_split()` splits **by `site_id`** (80/20), so all rows of a site stay on one side. Splitting by row would leak a site's trajectory across train/test.
2. **Interval labels** — `aft_matrix()` builds a `DMatrix` with two bounds per row:
   - uncensored (failed): `label_lower = label_upper = t`
   - right-censored: `label_lower = t`, `label_upper = +∞`
   - `t = max(time_to_event, 1.0)` (AFT requires `t > 0`).
3. **Training** — `num_boost_round=400`, `max_depth=5`, `eta=0.05`, `subsample=0.9`, `colsample_bytree=0.9`, `aft_loss_distribution_scale = σ = 0.6`. Here σ sets the **Weibull shape `k = 1/σ ≈ 1.67 > 1`**, i.e. an aging/wear-out hazard (failure rate rises with time).
4. **Prediction → two business outputs** from one survival curve `S(t|x) = exp(−(t/λ)^(1/σ))`, where `λ = bst.predict(x)` is the Weibull scale (characteristic life):
   - **7-day failure risk:** `risk7 = 1 − exp(−(7/λ)^(1/σ))` → the alarm score.
   - **Median RUL:** `rul_med = λ · (ln2)^σ` → planning horizon.
5. **Operating lever — dispatch rule.** Because the survival probabilities are well-ranked but conservatively scaled, the natural ops knob is the predicted RUL itself: **alarm if `rul_med ≤ D` days**. The table over `D ∈ {10,14,21,30}` lets ops trade precision vs recall.
6. **Interpretability — Cox PH companion** (`lifelines`). Per-site **mean** features, standardised (→ hazard ratio per +1 SD), `duration = max(time_to_event)+1`, `event = max(event)`, `penalizer=0.1`. Outputs hazard ratios per signal.

**Results (held-out sites):**

```
Discrimination (P fail ≤ 7d):  ROC-AUC 0.961 | PR-AUC 0.403
Dispatch rule (alarm if RUL ≤ D):
  D=10  precision 0.58  recall 0.19
  D=14  precision 0.42  recall 0.40
  D=21  precision 0.29  recall 0.69
  D=30  precision 0.22  recall 0.88   <- catch ~88% of imminent failures
RUL MAE: 29.2 d overall | 11.6 d when ≤30 d to failure
AFT gain:  internal_res >> backup_min ~ term_voltage > cycles > temp
Cox hazard ratios (per +1 SD): internal_res 3.0x | temp 2.1x (Arrhenius) |
  cycles 1.8x | mains_fail 1.6x | backup_min 0.33x | term_voltage 0.34x (protective)
```

The Cox ratios are a physics sanity check: rising internal resistance and heat sharply raise hazard; more backup-minutes/higher voltage are protective — exactly battery behavior.

**Productionising:** swap `gen_data.py` for a loader mapping real columns to the same 8 feature names + `event`/`time_to_event`; `train.py` runs unchanged. Score daily in batch; raise work-orders ranked by `time-to-failure × SLA tier`.

---

## 4. Improvements / Roadmap — the other two systems

These reuse the **same feature store and telemetry pipeline** as Solution A.

### Solution B — Agentic Energy Orchestration & Diesel Optimisation
- **Problem:** choose *which source runs when* (grid ↔ battery ↔ DG ↔ solar) and how to cycle batteries to minimise diesel/cost/CO₂ without breaching backup-time SLA.
- **Solution (3 layers):** (1) **forecast** grid-outage probability + site load (LSTM/Temporal-Fusion); (2) **optimise** with a **MILP** (or RL) for source dispatch + battery charge schedule under SLA and SoH-preserving constraints; (3) a **tool-using agent** executes via the site controller/SCADA with human-in-loop on high-impact actions. Validate on a digital twin before actuation.
- **KPI:** diesel litres ↓, CO₂ ↓, backup-time adherence.

### Solution C — Anomaly Detection: Fuel Theft + Outage Prediction with Auto-Dispatch
- **Problem:** chronic diesel pilferage losses; need proactive (not post-mortem) outage response.
- **Solution:** *Fuel theft* — model expected consumption from DG run-hours × load; flag negative residual drops with **Isolation Forest / robust rules**, refine with confirmed-case labels. *Outage prediction* — classifier on SoC-decay + mains-outage duration + DG fuel/health → **time-to-site-down ETA**. An **agent** auto-creates/prioritises NOC tickets and optimises **crew routing** by ETA and geography.
- **KPI:** fuel-loss value ↓, MTTR ↓, outage lead-time ↑.

**Solution-A specific upgrades:** time-varying Cox / discrete-time hazard for dynamic prediction; conformal prediction for calibrated RUL intervals; per-chemistry models (VRLA vs Li-ion); add rectifier-module and DG-engine RUL using the same AFT template.
