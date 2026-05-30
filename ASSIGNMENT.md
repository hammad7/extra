# AI/ML for 24×7 Telecom Tower Power Assurance

**Interview Assignment — Technical Overview**
Scope: An operator with end-to-end responsibility for keeping power live at telecom tower sites (e.g., Indus Towers–style tower-co). Three high-impact AI/agentic systems, the signals they consume, the models behind them, and a recommended architecture.

---

## 1. Problem Statement

A tower site must stay powered 24×7 through a prioritised energy chain:

> **1. Grid (mains) → 2. VRLA battery → 3. Li-ion battery → 4. Diesel Generator (DG)** *(plus increasingly Solar PV / fuel cells)*

The operator is accountable for **uptime (site availability SLA)**, **opex (diesel is the single largest controllable cost)**, **asset life** (batteries/DG capex), and **sustainability (CO₂)**. Today most sites run on rules/thresholds and reactive field visits, which cause surprise outages, over-runs of diesel, fuel pilferage losses, and premature asset replacement.

**Goal:** use the telemetry already flowing from the Remote Monitoring System (RMS) to *predict, optimise, and automate* power operations.

---

## 2. Available Signals (Features)

| Source | Key signals |
|--------|-------------|
| **Grid / Mains** | 3-phase voltage, current, frequency, phase availability, mains-fail/restore events, EB-hours, power factor, kWh import |
| **Rectifier / SMPS** | DC output (−48 V), load current, per-module current, module temperature, efficiency, active-module count, fault flags |
| **VRLA & Li-ion (BMS)** | terminal & per-cell voltage, string current, **SoC, SoH**, internal resistance/impedance, per-cell temperature, charge/discharge cycles, depth-of-discharge, backup-minutes; Li-ion adds cell-balancing + BMS alarms |
| **Diesel Generator** | fuel level, consumption rate, run-hours, start/stop count, oil pressure, coolant temp, RPM, output V/I/Hz, load %, starter-battery V, exhaust temp, vibration, auto/manual mode |
| **Solar PV** | panel V/I, MPPT output, irradiance, panel temp, daily generation, inverter status |
| **Site / Environment** | ambient & cabinet temperature, humidity, fan speed, AC/free-cooling status, door/intrusion, smoke/fire, water ingress, total BTS load |
| **Contextual / derived** | weather forecast, historical grid reliability per site, time-of-day/season, urban/rural class, fuel price, SLA tier |

> Note: **SoH/RUL is the hidden target**; models infer it from cheap always-on signals (internal resistance, voltage, backup-minutes, temperature).

---

## 3. Three High-Impact Solutions

### Solution A — Predictive Maintenance & Remaining Useful Life (RUL)

- **Problem:** Batteries are the #1 failure point and a major capex driver; DG/rectifier faults cause outages. Reactive replacement = surprise downtime + wasted truck-rolls.
- **Features:** internal resistance, terminal voltage, backup-minutes, cycles, DoD, ambient/cabinet temp, age, mains-fail frequency.
- **Model:** **XGBoost Accelerated-Failure-Time (AFT, Weibull)** survival model — handles right-censored (not-yet-failed) assets, captures wear-out shape, outputs a per-asset **survival curve** → both *P(fail ≤ 7 days)* (dispatch alarm) and *median RUL (days)* (planning). A **Cox PH** model runs alongside for interpretable hazard ratios (e.g., +1 SD internal resistance ≈ 3× hazard; +10 °C ≈ 2× — Arrhenius). *(Prototyped in this repo: ROC-AUC ≈ 0.96, near-EoL RUL MAE ≈ 12 days.)*
- **Architecture:** Batch scoring (daily) on the feature store → health score + RUL per asset → maintenance work-orders auto-prioritised by *time-to-failure × SLA tier*.

### Solution B — Agentic Energy Orchestration & Diesel Optimisation

- **Problem:** Deciding *which source runs when* (grid ↔ battery ↔ DG ↔ solar) and how to cycle batteries to minimise diesel, cost, and CO₂ — without breaching backup-time/SLA.
- **Features:** grid-availability history & forecast, site load forecast, battery SoC/SoH, fuel level/price, solar generation forecast, tariff/time-of-day.
- **Model (3 layers):**
  1. *Forecasting* — grid-outage probability & site-load prediction (temporal models: LSTM/Temporal-Fusion/Prophet).
  2. *Optimisation* — **MILP** (or RL for adaptive policies) producing source-dispatch + battery charge/discharge schedule that minimises `diesel + cost + CO₂` subject to backup-time ≥ SLA and SoH-preserving cycling constraints.
  3. *Agent* — reasons over forecasts + constraints, **executes** via the site controller/SCADA, and escalates exceptions to the NOC. Tool-using LLM agent for planning/explanation; deterministic solver in the loop for the math.
- **Architecture:** Near-real-time loop per site/cluster; digital-twin/simulation for safe policy validation before actuation; human-in-the-loop approval for high-impact actions.

### Solution C — Anomaly Detection: Fuel Theft + Outage Prediction with Auto-Dispatch

- **Problem:** Diesel pilferage is a large, chronic loss; and imminent site-down events need proactive crews, not post-mortems.
- **Features:** fuel-level trajectory vs DG run-hours & load (expected-consumption model), refuel events, SoC trajectory + mains-fail duration + DG fuel/health, door/intrusion, GPS/time.
- **Model:**
  - *Fuel theft:* model **expected consumption** from run-hours × load; flag negative-deviation drops with **Isolation Forest / robust residual rules**, refined by supervised labels from confirmed cases.
  - *Outage prediction:* classifier on SoC-decay + mains-outage duration + DG fuel/health → **time-to-site-down ETA**.
- **Architecture:** Streaming anomaly detection (event-driven) → **agent** auto-creates & prioritises NOC tickets, optimises **crew routing** (refuel/repair) by ETA-to-failure and geography.

---

## 4. Recommended Reference Architecture (shared)

```
[Site RMS / Edge gateway] --MQTT/Modbus--> [Ingestion: Kafka/IoT Hub]
        |                                            |
   (edge anomaly                                [Stream proc: Flink]
    + buffering)                                     |
        v                                            v
[Time-series store (Timescale/Influx) + Data lake (S3/Delta)]
        |
        v
[Feature Store] --> [Training (MLflow registry)] --> [Model serving]
        |                                               |
        |                          +--------------------+--------------------+
        |                          |                    |                    |
        v                          v                    v                    v
[Batch scoring: RUL/PdM]   [Forecasts B]        [Stream scoring: anomaly C]  |
        \__________________________\____________________/                    |
                                    v                                         v
                         [Agent / Orchestration layer]  <----- constraints/solver (MILP)
                          (LLM tools + optimiser + policies, human-in-loop)
                                    |
              +---------------------+----------------------+
              v                     v                      v
     [NOC ticketing/CRM]   [Site controller/SCADA]   [Crew dispatch/routing]
                                    |
                                    v
                    [Monitoring + MLOps: drift, feedback, retrain]
```

**Design principles:** edge pre-processing for resilience on flaky links; one feature store shared across all three systems; deterministic solver inside the agent loop (LLM plans/explains, optimiser decides the numbers); digital-twin validation before any actuation; closed feedback loop (confirmed failures/thefts → labels → retrain).

---

## 5. Success Metrics

| System | Primary KPI | Secondary |
|--------|-------------|-----------|
| A — PdM/RUL | ↓ unplanned battery/DG outages; recall of imminent failures | ↓ truck-rolls; asset-life extension |
| B — Orchestration | ↓ diesel litres & opex; ↓ CO₂ | SLA backup-time adherence; battery-cycle health |
| C — Anomaly/Outage | ↓ fuel-loss value; MTTR ↓ | outage-ETA lead time; ticket precision |

**Sequencing:** A and C first (lowest integration risk, fastest ROI, build trust in the data/feedback loop), then B (closed-loop actuation) once the data foundation and digital twin are validated.
