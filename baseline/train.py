"""
Predictive maintenance for telecom battery banks (Solution A) -- NAIVE BASELINE.

Two SOTA-but-simple XGBoost models on the same telemetry:
  1) CLASSIFIER  -> P(fail within 7 days)  : the actionable alarm
  2) REGRESSOR   -> Remaining Useful Life (days)

Split is done BY SITE (no leakage) and the regressor is trained only on
batteries that actually fail (uncensored), which is standard for RUL.

LIMITATION: dropping censored sites (rul == 9999) for the regressor biases RUL
estimates low -- the survivors are the *healthy* batteries. The survival
(XGBoost-AFT) version at the repo root fixes this by keeping censored data.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

FEATURES = [
    "age_days", "internal_res_mohm", "backup_min", "cycles", "dod",
    "avg_temp_c", "term_voltage", "mains_fail_count",
]
# NOTE: 'soh' is excluded on purpose - it is the hidden ground-truth driver.
# Models must learn it from the cheap, always-available sensor signals.


def site_split(df, frac=0.8, seed=0):
    sites = df["site_id"].unique()
    np.random.default_rng(seed).shuffle(sites)
    cut = int(len(sites) * frac)
    tr = df[df.site_id.isin(sites[:cut])]
    te = df[df.site_id.isin(sites[cut:])]
    return tr, te


def main():
    df = pd.read_parquet("telemetry.parquet")
    tr, te = site_split(df)

    # ---------- 1) Failure-within-7-days classifier ----------
    pos_w = (tr.fail_within_7d == 0).sum() / max(1, (tr.fail_within_7d == 1).sum())
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, scale_pos_weight=pos_w,
        eval_metric="aucpr", n_jobs=-1,
    )
    clf.fit(tr[FEATURES], tr.fail_within_7d)
    p = clf.predict_proba(te[FEATURES])[:, 1]
    print("\n=== Alarm: fail within 7 days ===")
    print(f"ROC-AUC : {roc_auc_score(te.fail_within_7d, p):.3f}")
    print(f"PR-AUC  : {average_precision_score(te.fail_within_7d, p):.3f}")
    print(f"{'thr':>5} {'precision':>10} {'recall':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    for thr in (0.5, 0.8, 0.9, 0.95):
        pred = (p >= thr).astype(int)
        tp = int(((pred == 1) & (te.fail_within_7d == 1)).sum())
        fp = int(((pred == 1) & (te.fail_within_7d == 0)).sum())
        fn = int(((pred == 0) & (te.fail_within_7d == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        print(f"{thr:>5} {prec:>10.2f} {rec:>8.2f} {tp:>5} {fp:>5} {fn:>5}")

    # ---------- 2) RUL regressor (uncensored sites only) ----------
    unc = lambda d: d[d.rul <= 365]
    reg = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
    )
    reg.fit(unc(tr)[FEATURES], unc(tr).rul)
    r = reg.predict(unc(te)[FEATURES])
    mae = mean_absolute_error(unc(te).rul, r)
    print("\n=== RUL regression (days-to-failure) ===")
    print(f"MAE     : {mae:.1f} days")
    near = unc(te).rul <= 30
    print(f"MAE(<=30d to fail): {mean_absolute_error(unc(te).rul[near], r[near]):.1f} days")

    # ---------- top signals ----------
    imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n=== Top predictive signals (classifier) ===")
    for k, v in imp.head(5).items():
        print(f"  {k:20s} {v:.3f}")


if __name__ == "__main__":
    main()
