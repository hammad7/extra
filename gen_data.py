"""
Generate ~1 year of daily battery telemetry for telecom-tower sites.
Each site has a battery that degrades over time; when State-of-Health (SOH)
crosses a failure threshold, a power-failure event is logged. We label each
daily row with Remaining-Useful-Life (RUL, days to failure) so the same
dataset trains both the classifier and the regressor.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_SITES = 400
DAYS = 365
SOH_FAIL = 0.70          # battery considered failed below 70% health
RUL_HORIZON = 7          # "will it fail within 7 days?" alarm window


def simulate_site(site_id: int) -> pd.DataFrame:
    # --- per-site latent characteristics ---
    soh0 = RNG.uniform(0.80, 0.97)                 # starting health
    base_temp = RNG.uniform(22, 40)                # site climate (deg C)
    cycles_per_day = RNG.uniform(0.3, 1.8)         # grid reliability proxy
    age0 = RNG.integers(300, 1800)                 # battery age at day 0
    # degradation rate accelerates with heat and cycling (SOTA wear models)
    daily_wear = (
        0.00040
        + 0.000022 * max(0, base_temp - 25)        # Arrhenius-style heat stress
        + 0.00020 * cycles_per_day
        + RNG.normal(0, 0.00004)
    )

    rows, soh = [], soh0
    for d in range(DAYS):
        soh = max(0.4, soh - max(0, daily_wear + RNG.normal(0, 0.00004)))
        age = age0 + d
        temp = base_temp + RNG.normal(0, 2.5) + 4 * np.sin(d / 58.0)  # seasonal
        # measurable signals that track the (hidden) SOH
        internal_res = 4.0 / soh + RNG.normal(0, 0.15)        # mOhm, rises as SOH drops
        backup_min = 240 * soh - 1.5 * max(0, temp - 25) + RNG.normal(0, 8)
        cycles = cycles_per_day * RNG.uniform(0.6, 1.4)
        dod = np.clip(RNG.normal(0.45, 0.12), 0.1, 0.95)      # depth of discharge
        term_v = 53.5 - 6 * (1 - soh) + RNG.normal(0, 0.2)    # -48V bus
        mains_fail = RNG.poisson(cycles_per_day)
        rows.append(dict(
            site_id=site_id, day=d, age_days=age, soh=soh,
            internal_res_mohm=internal_res, backup_min=backup_min,
            cycles=cycles, dod=dod, avg_temp_c=temp, term_voltage=term_v,
            mains_fail_count=mains_fail,
        ))

    df = pd.DataFrame(rows)
    # --- failure event = first day SOH drops below threshold ---
    failed = df.index[df["soh"] < SOH_FAIL]
    if len(failed):
        fail_day = int(df.loc[failed[0], "day"])
        df = df[df["day"] <= fail_day].copy()             # battery replaced after
        df["event"] = 1                                   # failure observed
        df["time_to_event"] = fail_day - df["day"]        # exact days-to-failure
    else:
        df["event"] = 0                                   # right-censored
        df["time_to_event"] = (DAYS - 1) - df["day"]      # observed-alive horizon
    return df


def main():
    df = pd.concat([simulate_site(i) for i in range(N_SITES)], ignore_index=True)
    # alarm label only fires when a failure truly occurs within the window
    df["fail_within_7d"] = ((df.event == 1) & (df.time_to_event <= RUL_HORIZON)).astype(int)
    df.to_parquet("telemetry.parquet")
    pos = int(df.fail_within_7d.sum())
    print(f"rows={len(df):,}  sites={df.site_id.nunique()}  "
          f"failed-sites={df[df.event==1].site_id.nunique()}  "
          f"censored-sites={df[df.event==0].site_id.nunique()}  "
          f"alarm-positives={pos:,} ({100*pos/len(df):.1f}%)")


if __name__ == "__main__":
    main()
