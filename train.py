"""
Predictive maintenance for telecom battery banks (Solution A) -- survival edition.

ONE model instead of two: XGBoost Accelerated-Failure-Time (AFT) with a Weibull
error distribution. It is gradient boosting trained on the Weibull/AFT
likelihood, so it:
  * uses RIGHT-CENSORED sites (batteries that have not failed yet) instead of
    throwing them away  -> the key fix over the classifier+regressor version,
  * captures the wear-out shape (Weibull shape k = 1/sigma > 1),
  * keeps tree nonlinearity / interactions,
  * yields a full survival curve per row, from which we read BOTH:
        - P(fail within 7 days)   -> the actionable alarm
        - median RUL (days)       -> planning horizon.

A Cox proportional-hazards model is fit alongside purely for interpretability
(hazard ratios per signal).

Weibull survival:  S(t|x) = exp( -(t / lambda(x))^(1/sigma) )
where lambda(x) = model.predict(x) is the predicted scale (characteristic life).
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

FEATURES = [
    "age_days", "internal_res_mohm", "backup_min", "cycles", "dod",
    "avg_temp_c", "term_voltage", "mains_fail_count",
]
# 'soh' is excluded on purpose - it is the hidden ground-truth driver.
SIGMA = 0.6          # AFT error scale; 'extreme' dist => Weibull, shape k = 1/SIGMA (=1.67 => wear-out)
HORIZON = 7          # alarm window (days)


def site_split(df, frac=0.8, seed=0):
    sites = df["site_id"].unique()
    np.random.default_rng(seed).shuffle(sites)
    cut = int(len(sites) * frac)
    return df[df.site_id.isin(sites[:cut])], df[df.site_id.isin(sites[cut:])]


def aft_matrix(d):
    """DMatrix with interval labels: uncensored -> [t,t], right-censored -> [t, inf)."""
    t = np.maximum(d.time_to_event.values.astype(float), 1.0)   # AFT needs t > 0
    dm = xgb.DMatrix(d[FEATURES])
    dm.set_float_info("label_lower_bound", t)
    dm.set_float_info("label_upper_bound", np.where(d.event.values == 1, t, np.inf))
    return dm


def main():
    df = pd.read_parquet("telemetry.parquet")
    tr, te = site_split(df)
    print(f"train rows={len(tr):,} (censored rows kept={int((tr.event==0).sum()):,})")

    # ---------- XGBoost AFT (Weibull) ----------
    params = dict(
        objective="survival:aft", eval_metric="aft-nloglik",
        aft_loss_distribution="extreme", aft_loss_distribution_scale=SIGMA,
        max_depth=5, eta=0.05, subsample=0.9, colsample_bytree=0.9,
    )
    bst = xgb.train(params, aft_matrix(tr), num_boost_round=400)

    lam = bst.predict(aft_matrix(te))                 # Weibull scale lambda(x), days
    risk7 = 1.0 - np.exp(-(HORIZON / lam) ** (1.0 / SIGMA))   # P(fail <= 7d)
    rul_med = lam * (np.log(2.0) ** SIGMA)            # median survival time (days)

    # ---------- alarm: dispatch rule on predicted RUL ----------
    # risk7 ranks well (see AUC) but a survival model's natural lever is the
    # predicted time-to-failure itself: "send a crew if RUL <= D days".
    y = te.fail_within_7d.values
    print("\n=== Discrimination (P(fail within 7 days) from AFT curve) ===")
    print(f"ROC-AUC : {roc_auc_score(y, risk7):.3f}")
    print(f"PR-AUC  : {average_precision_score(y, risk7):.3f}")
    print("\n=== Dispatch rule: alarm if predicted median RUL <= D days ===")
    print(f"{'D':>4} {'precision':>10} {'recall':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    for D in (10, 14, 21, 30):
        pred = (rul_med <= D).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        print(f"{D:>4} {prec:>10.2f} {rec:>8.2f} {tp:>5} {fp:>5} {fn:>5}")

    # ---------- RUL (evaluate on uncensored test rows only) ----------
    m = te.event.values == 1
    print("\n=== RUL: median survival (days-to-failure) ===")
    print(f"MAE              : {mean_absolute_error(te.time_to_event[m], rul_med[m]):.1f} days")
    near = m & (te.time_to_event.values <= 30)
    print(f"MAE(<=30d to fail): {mean_absolute_error(te.time_to_event[near], rul_med[near]):.1f} days")

    # ---------- top signals (AFT gain) ----------
    gain = bst.get_score(importance_type="gain")
    imp = pd.Series(gain).reindex(FEATURES).fillna(0).sort_values(ascending=False)
    print("\n=== Top predictive signals (AFT gain) ===")
    for k, v in imp.head(5).items():
        print(f"  {k:20s} {v:.1f}")

    # ---------- Cox PH: interpretable hazard ratios ----------
    try:
        from lifelines import CoxPHFitter
        g = df.groupby("site_id")
        cox = g[FEATURES].mean()                       # per-site average signals
        cox = (cox - cox.mean()) / cox.std()           # standardize -> HR per +1 SD
        cox["duration"] = g["time_to_event"].max() + 1.0
        cox["event"] = g["event"].max()
        cph = CoxPHFitter(penalizer=0.1).fit(cox, "duration", "event")
        hr = cph.hazard_ratios_.sort_values(ascending=False)
        print("\n=== Cox PH hazard ratios (per +1 SD; >1 = fails sooner) ===")
        for k, v in hr.items():
            print(f"  {k:20s} HR={v:.2f}")
    except Exception as e:
        print(f"\n[Cox PH skipped: {e}]")


if __name__ == "__main__":
    main()
