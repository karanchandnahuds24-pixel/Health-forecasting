"""Generate the written deliverables from the data: research paper + executive summary.

Run:  python report.py [path/to/data.csv] [--horizon 14] [--capacity N] [--out reports]

Writes reports/research_paper.md, reports/executive_summary.md and reports/figures/*.png.
Every number in the documents is computed from the dataset, so re-run it whenever the data updates.
The narrative is a data-driven first draft: review it and add domain context before submitting.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import forecasting as fx  # noqa: E402

HORIZONS = (1, 3, 7, 14)
LEVEL = 0.90
RATIOS = {"Case managers": 12, "Youth care workers": 8, "Clinicians": 25, "Medical staff": 50}
LABELS = {"apprehended": "Apprehended (CBP intake)", "in_cbp": "In CBP custody",
          "transferred": "Transferred to HHS", "in_hhs": "In HHS care", "discharged": "Discharged from HHS"}


def md_table(df: pd.DataFrame, fmt: str = "{:,.1f}") -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "–" if np.isnan(v) else fmt.format(v)
        if isinstance(v, pd.Timestamp):
            return f"{v:%Y-%m-%d}"
        return str(v)
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([cell(idx)] + [cell(v) for v in row]) + " |")
    return "\n".join(lines)


def savefig(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return f"figures/{path.name}"


# ---------------------------------------------------------------- analysis
def eda(df: pd.DataFrame, figdir: Path) -> dict:
    r = {}
    obs = df[~df["imputed"]]
    r["start"], r["end"], r["n"] = df.index[0], df.index[-1], len(df)
    r["imputed"] = int(df["imputed"].sum())
    r["summary"] = df[list(LABELS)].describe().T[["mean", "std", "min", "50%", "max"]].rename(index=LABELS)

    y = df["in_hhs"]
    r["peak_date"], r["peak"] = y.idxmax(), y.max()
    r["current"] = y.iloc[-1]
    r["chg30"] = y.iloc[-1] - y.iloc[-31] if len(y) > 31 else np.nan
    net = df["transferred"] - df["discharged"]
    r["net30"] = net.iloc[-30:].mean()
    r["pos_share"] = (net > 0).mean() * 100
    # Little's law: average stay ≈ average load / average daily exits
    r["los_all"] = y.mean() / max(df["discharged"].mean(), 1e-9)
    r["los_90"] = y.iloc[-90:].mean() / max(df["discharged"].iloc[-90:].mean(), 1e-9)

    dow = obs.groupby(obs.index.dayofweek)[["apprehended", "transferred", "discharged"]].mean()
    dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(dow)]
    r["dow"] = dow
    wk = obs[obs.index.dayofweek < 5]["discharged"].mean()
    we = obs[obs.index.dayofweek >= 5]["discharged"].mean()
    r["weekend_dip"] = (1 - we / wk) * 100 if wk else np.nan
    r["corr"] = df[list(LABELS)].corr().rename(index=LABELS, columns=LABELS)
    # how many days does HHS load lag CBP transfers? strongest correlation of Δload with lagged transfers
    lags = {k: df["in_hhs"].diff().corr(df["transferred"].shift(k)) for k in range(0, 15)}
    r["best_lag"] = max(lags, key=lambda k: lags[k] if not np.isnan(lags[k]) else -1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(df.index, df["in_hhs"], label="In HHS care")
    axes[0].plot(df.index, df["in_cbp"], label="In CBP custody")
    axes[0].set_title("Care load (stock)")
    axes[0].legend()
    for c in ["apprehended", "transferred", "discharged"]:
        axes[1].plot(df.index, df[c].rolling(7).mean(), label=LABELS[c])
    axes[1].set_title("Daily flows (7-day average)")
    axes[1].legend()
    r["fig_trend"] = savefig(fig, figdir / "trend.png")

    dec = fx.decompose(df["in_hhs"])
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for ax, comp, t in zip(axes, [dec.observed, dec.trend, dec.seasonal, dec.resid],
                           ["Observed", "Trend", "Weekly seasonality", "Residual"]):
        ax.plot(comp.index, comp)
        ax.set_title(t, fontsize=10)
    r["fig_decomp"] = savefig(fig, figdir / "decomposition.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    dow.plot.bar(ax=ax)
    ax.set_title("Average daily flow by weekday")
    r["fig_dow"] = savefig(fig, figdir / "weekday.png")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    n7 = net.rolling(7).mean()
    ax.fill_between(n7.index, n7, 0, where=n7 > 0, color="#d62728", alpha=0.6, label="Load rising")
    ax.fill_between(n7.index, n7, 0, where=n7 <= 0, color="#2ca02c", alpha=0.6, label="Load falling")
    ax.set_title("Net pressure: transfers − discharges (7-day average)")
    ax.legend()
    r["fig_net"] = savefig(fig, figdir / "net_pressure.png")
    return r


def evaluation(df: pd.DataFrame, figdir: Path) -> dict:
    r = {}
    for tgt in ["in_hhs", "discharged"]:
        recs = fx.evaluate_models(df, tgt, fx.MODELS, HORIZONS, level=LEVEL)
        mo = fx.metrics_overall(recs).set_index("model")
        mo["Accuracy (%)"] = [fx.forecast_accuracy(recs[recs.model == m]) for m in mo.index]
        mo["Stability index"] = [fx.stability_index(recs[recs.model == m]) for m in mo.index]
        mo["Error degradation"] = [fx.error_degradation(recs[recs.model == m]) for m in mo.index]
        mh = fx.metrics_by_horizon(recs).pivot(index="model", columns="horizon", values="MAE")
        mh.columns = [f"MAE d{h}" for h in mh.columns]
        r[tgt] = dict(overall=mo.sort_values("MAE"), horizon=mh.loc[mo.sort_values("MAE").index])

        fig, ax = plt.subplots(figsize=(8, 4))
        for m in mh.index:
            ax.plot(list(HORIZONS), mh.loc[m].values, marker="o", label=m)
        ax.set_xlabel("Horizon (days)")
        ax.set_ylabel("MAE")
        ax.set_title(f"Error by horizon – {fx.TARGET_LABELS[tgt]}")
        ax.legend(fontsize=8)
        r[tgt]["fig"] = savefig(fig, figdir / f"horizon_error_{tgt}.png")
    return r


def outlook(df: pd.DataFrame, model: str, H: int, capacity: float, figdir: Path) -> dict:
    r = {"model": model, "H": H, "capacity": capacity}
    fc = fx.forecast(df, "in_hhs", H, model, LEVEL)
    tr = fx.forecast(df, "transferred", H, model, LEVEL)
    dis = fx.forecast(df, "discharged", H, model, LEVEL)
    prob = fx.breach_probability(fc, capacity, LEVEL)
    r.update(fc=fc, end=fc.yhat.iloc[-1], lo=fc.lo.iloc[-1], hi=fc.hi.iloc[-1], peak=fc.yhat.max(),
             prob=prob.max(), lead=fx.surge_lead_time(fc, capacity),
             dis_total=dis.yhat.sum(), net_total=(tr.yhat - dis.yhat).sum())
    now = fx.staffing_needs(df["in_hhs"].iloc[[-1]], RATIOS).iloc[0]
    peak = fx.staffing_needs(pd.Series([fc.yhat.max()]), RATIOS).iloc[0]
    hi = fx.staffing_needs(pd.Series([fc.hi.max()]), RATIOS).iloc[0]
    r["staff"] = pd.DataFrame({"Today": now, "At forecast peak": peak, "Additional": peak - now,
                               "Upper bound": hi}).astype(int)
    r["staff"].index.name = "Role"

    scen = {
        "Baseline": fc.yhat,
        "Intake surge (+25% transfers)": fx.scenario_path(fc, tr, dis, 0.25, 0),
        "Slow discharges (−20%)": fx.scenario_path(fc, tr, dis, 0, -0.20),
        "Combined": fx.scenario_path(fc, tr, dis, 0.25, -0.20),
    }
    r["scen"] = pd.DataFrame({
        "End-of-horizon load": {k: v.iloc[-1] for k, v in scen.items()},
        "Peak load": {k: v.max() for k, v in scen.items()},
        "Days above capacity": {k: float((v > capacity).sum()) for k, v in scen.items()},
    })
    r["scen"].index.name = "Scenario"

    fig, ax = plt.subplots(figsize=(10, 4.5))
    hist = df["in_hhs"].iloc[-90:]
    ax.plot(hist.index, hist, color="black", label="Actual")
    ax.fill_between(fc.index, fc.lo, fc.hi, alpha=0.2, label=f"{int(LEVEL*100)}% interval")
    for k, v in scen.items():
        ax.plot(v.index, v, label=k, ls="-" if k == "Baseline" else "--")
    ax.axhline(capacity, color="red", ls=":", label="Capacity threshold")
    ax.set_title(f"{H}-day care-load forecast and scenarios ({model})")
    ax.legend(fontsize=8)
    r["fig"] = savefig(fig, figdir / "forecast.png")
    return r


# ---------------------------------------------------------------- writing
def recommendations(e: dict, o: dict, ev: dict) -> list[str]:
    rec = []
    if o["prob"] >= 0.5 or o["lead"]:
        when = f"within {o['lead']} day(s)" if o["lead"] else "within the horizon"
        rec.append(f"**Activate surge capacity now.** The forecast crosses the {o['capacity']:,.0f}-child "
                   f"threshold {when} (breach probability {o['prob']:.0%}). Bring influx beds and "
                   "contract staff online before the crossing, not after.")
    elif o["prob"] >= 0.2:
        rec.append(f"**Prepare standby capacity.** Breach probability reaches {o['prob']:.0%}; confirm "
                   "standby shelter and staffing agreements can be activated within a week.")
    else:
        rec.append(f"**Hold current capacity.** Breach probability stays at {o['prob']:.0%} or below "
                   f"over the next {o['H']} days.")
    if e["net30"] > 0:
        rec.append(f"**Increase discharge throughput.** Over the last 30 days intake outpaced discharges by "
                   f"{e['net30']:,.0f} children/day on average. Prioritise sponsor vetting and case-manager "
                   "capacity, which drive exits.")
    else:
        rec.append(f"**Sustain discharge throughput.** Exits exceeded intake by {-e['net30']:,.0f} "
                   "children/day over the last 30 days; protect the case-management capacity behind this.")
    if e["weekend_dip"] > 15:
        rec.append(f"**Address the weekend discharge dip.** Discharges fall {e['weekend_dip']:.0f}% on "
                   "weekends while intake continues. Weekend case-management and release staffing would "
                   "flatten weekly peaks in load.")
    add = o["staff"]["Additional"]
    if (add > 0).any():
        parts = ", ".join(f"{v} {k.lower()}" for k, v in add.items() if v > 0)
        rec.append(f"**Plan staffing for the forecast peak.** At the assumed ratios the peak needs {parts} "
                   "more than today. Begin recruitment or redeployment now, given onboarding lead times.")
    best = ev["in_hhs"]["overall"].index[0]
    cov = ev["in_hhs"]["overall"].loc[best, "Interval coverage (%)"]
    rec.append(f"**Use {best} as the primary planning model** (lowest walk-forward MAE) and read the "
               f"interval, not only the point forecast. Its {int(LEVEL*100)}% interval contained the "
               f"actual value {cov:.0f}% of the time in backtests.")
    rec.append("**Operationalise the forecast.** Refresh the data and forecasts daily, review the early-warning "
               "panel in weekly capacity meetings, and re-run model validation monthly.")
    return rec


def write_paper(path: Path, e, ev, o, rec):
    b = ev["in_hhs"]["overall"]
    best = b.index[0]
    bd = ev["discharged"]["overall"]
    naive_mae = b.loc["Naive", "MAE"]
    gain = (1 - b.loc[best, "MAE"] / naive_mae) * 100 if naive_mae else np.nan
    stat_best = b.loc[[m for m in b.index if m in ("SARIMA", "Exp. Smoothing")]].index[0]
    ml_best = b.loc[[m for m in b.index if m in fx.ML_FACTORY]].index[0]
    lead_txt = f"in {o['lead']} day(s)" if o["lead"] else "at no point in the horizon (point forecast)"

    s = f"""# Forecasting Care Load and Discharge Demand in the HHS Unaccompanied Children Program

*Research paper – generated from data covering {e['start']:%d %B %Y} to {e['end']:%d %B %Y}*

## Abstract

This paper moves the Unaccompanied Alien Children (UAC) Program's daily reporting from description to
prediction. Using {e['n']:,} days of data, we compare naïve, moving-average, SARIMA, exponential-smoothing,
random-forest and gradient-boosting forecasters under expanding-window walk-forward validation at 1-, 3-, 7-
and 14-day horizons. The best model for children in HHS care, **{best}**, achieved a mean absolute error of
{b.loc[best, 'MAE']:,.0f} children ({b.loc[best, 'Accuracy (%)']:.1f}% accuracy), {gain:.0f}% lower than
the naïve persistence baseline. The {o['H']}-day outlook projects {o['end']:,.0f} children in care
({int(LEVEL*100)}% interval {o['lo']:,.0f}–{o['hi']:,.0f}) with a maximum capacity-breach probability of
{o['prob']:.0%} against a threshold of {o['capacity']:,.0f}.

## 1. Introduction and problem statement

The UAC Program operates under high uncertainty: changes in border activity, enforcement policy or
humanitarian crises can raise the number of children entering federal care within days. Descriptive
reporting shows what already happened. Planners also need to know how many children will be in care in
the coming days and weeks, whether discharges will keep pace with transfers, and when to scale shelters,
medical staff and caseworkers. Without forecasts, responses lag the surge, raising the risk of
overcrowding, staff burnout and longer stays for children.

**Objectives.** (1) Forecast the number of children in HHS care; (2) estimate the future imbalance between
intake and exits; (3) predict short-term discharge demand; and, secondarily, provide early warnings,
quantify forecast uncertainty and compare statistical with machine-learning approaches.

## 2. Data

The dataset holds one row per reporting day with five series: children apprehended and placed in CBP
custody (intake), children in CBP custody, children transferred out of CBP custody (flow into HHS),
children in HHS care (load) and children discharged from HHS care (sponsor placements).

- Period: {e['start']:%d %b %Y} – {e['end']:%d %b %Y} ({e['n']:,} calendar days)
- Missing reporting days: {e['imputed']} ({e['imputed'] / e['n'] * 100:.1f}%), filled by time
  interpolation and **masked** from model training targets and accuracy scoring

{md_table(e['summary'])}

## 3. Exploratory data analysis

![Trends]({e['fig_trend']})

**Care load.** Children in HHS care peaked at {e['peak']:,.0f} on {e['peak_date']:%d %b %Y}; the latest
value is {e['current']:,.0f}, a change of {e['chg30']:+,.0f} over the last 30 days.

**Stock–flow balance.** Daily change in HHS load is governed by transfers in minus discharges out. Net
pressure was positive (load rising) on {e['pos_share']:.0f}% of days and averaged {e['net30']:+,.0f}
children/day over the last 30 days. Changes in HHS load correlate most strongly with CBP transfers lagged
{e['best_lag']} day(s), so CBP data offers a short lead on HHS intake.

![Net pressure]({e['fig_net']})

**Length of stay.** By Little's law (average load ÷ average daily exits), the implied average stay is
{e['los_all']:.0f} days over the full period and {e['los_90']:.0f} days over the last 90 days.

**Weekly seasonality.** Discharges are {e['weekend_dip']:.0f}% lower on weekends than on weekdays, which
produces a regular weekly cycle in care load (see decomposition).

![Weekday]({e['fig_dow']})

![Decomposition]({e['fig_decomp']})

**Correlations between series.**

{md_table(e['corr'], "{:.2f}")}

## 4. Methodology

**Time-series preparation.** Dates parsed to a datetime index, re-indexed to a continuous daily
frequency, gaps filled by time interpolation with an `imputed` flag, and additive decomposition (period 7)
into trend, weekly seasonality and residual.

**Feature engineering (ML models).** Lags t, t−1, t−7, t−14 of all five series; 7- and 14-day rolling
mean and variance; net pressure (transfers − discharges) and its 7/14-day means; calendar effects
(day of week, weekend, month); and US federal holiday proxies (holiday flag, days to/since nearest
holiday). All features use only information available at the forecast origin.

**Models.** Baselines: naïve persistence and 7-day moving average. Statistical: SARIMA(1,1,1)(1,0,1)₇
and damped additive-trend exponential smoothing with weekly seasonality (ETS). Machine learning: random
forest and gradient boosting, trained as *direct* multi-horizon models (one model per horizon predicting
the change from the current level). If a statistical model fails to converge it falls back to the moving
average, and this is flagged.

**Uncertainty.** SARIMA and ETS use their analytical prediction intervals; baselines use empirical
quantiles of historical h-step errors; ML models use residual quantiles from a time-ordered hold-out
block. Calibration is checked by **interval coverage** in the backtest.

**Validation.** Strict time ordering, no random sampling. Expanding-window walk-forward with 8 origins
spaced 7 days apart; each model is refitted at each origin and scored at 1, 3, 7 and 14 days ahead.
Metrics: MAE, RMSE, MAPE and error by horizon.

**KPIs.** Forecast accuracy = 100 − MAPE; Forecast stability index = 1 − coefficient of variation of
per-origin MAE; Robustness (error degradation) = MAE at day 14 ÷ MAE at day 1; Capacity-breach
probability = P(load > threshold) from the forecast distribution; Surge lead time = days until the
forecast first exceeds the threshold.

## 5. Results

### 5.1 Children in HHS care

{md_table(b)}

{md_table(ev['in_hhs']['horizon'])}

![Horizon error]({ev['in_hhs']['fig']})

**{best}** has the lowest overall MAE. The best statistical model is {stat_best}
(MAE {b.loc[stat_best, 'MAE']:,.0f}) and the best ML model is {ml_best} (MAE {b.loc[ml_best, 'MAE']:,.0f}).
Error grows with horizon for all models. The ratio of day-14 to day-1 error ranges from
{b['Error degradation'].min():.1f}× to {b['Error degradation'].max():.1f}×. Short-term forecasts (1–3
days) are reliable enough for daily operations; 14-day forecasts should be read with their interval.

### 5.2 Daily discharges

{md_table(bd)}

![Horizon error]({ev['discharged']['fig']})

For discharge demand the best model is **{bd.index[0]}** (MAE {bd.iloc[0]['MAE']:,.0f} discharges/day).
Daily discharges are noisier than the care-load stock, so relative errors are larger.

## 6. Forecast outlook and early warning

![Forecast]({o['fig']})

Using {o['model']}, the {o['H']}-day forecast is {o['end']:,.0f} children in HHS care
({int(LEVEL*100)}% interval {o['lo']:,.0f}–{o['hi']:,.0f}), peaking at {o['peak']:,.0f}. The point forecast
crosses the capacity threshold of {o['capacity']:,.0f} {lead_txt}; the maximum daily breach probability is
{o['prob']:.0%}. Projected discharges total {o['dis_total']:,.0f} over the horizon, with a cumulative net
change of {o['net_total']:+,.0f} children.

**Staffing implied by the forecast** (assumed ratios of children per staff member:
{', '.join(f'{k} 1:{v}' for k, v in RATIOS.items())} – replace with program ratios):

{md_table(o['staff'], "{:,.0f}")}

**Scenario analysis** (stock-flow: shifts the baseline by the cumulative change in transfers − discharges):

{md_table(o['scen'], "{:,.0f}")}

## 7. Recommendations

{chr(10).join(f'{i}. {r}' for i, r in enumerate(rec, 1))}

## 8. Limitations and future work

- The data contain no exogenous drivers (policy changes, border encounters by nationality, shelter bed
  counts), so structural breaks are only detected after they begin. Adding them is the largest available
  accuracy gain.
- The capacity threshold and staffing ratios are planning inputs, not values from the dataset.
- Interval coverage below the nominal level means the intervals are too narrow, and planners should
  widen buffers accordingly.
- Walk-forward validation uses the most recent 8 weeks of origins. Performance in earlier regimes,
  such as the 2021 surge, may differ.
- Future work: hierarchical forecasts by facility, probabilistic ensembles, and automated daily retraining.

## 9. Conclusion

Daily UAC reporting supports useful short-term forecasting. {best} reduces error by {gain:.0f}% relative
to naïve persistence. With calibrated intervals, breach probabilities and staffing translation, the
forecasts give HHS planners advance notice to scale shelters and staff before capacity stress occurs.
"""
    path.write_text(s, encoding="utf-8")


def write_exec(path: Path, e, ev, o, rec):
    b = ev["in_hhs"]["overall"]
    best = b.index[0]
    status = ("HIGH RISK" if (o["prob"] >= 0.5 or o["lead"]) else "ELEVATED" if o["prob"] >= 0.2 else "STABLE")
    s = f"""# Executive Summary – UAC Care Load Forecast

**For:** HHS / ORR leadership and capacity planners  ·  **Data through:** {e['end']:%d %B %Y}  ·
**Outlook:** next {o['H']} days  ·  **Status: {status}**

## Bottom line

- **{e['current']:,.0f}** children are in HHS care today. The forecast for day {o['H']} is
  **{o['end']:,.0f}** (likely range {o['lo']:,.0f}–{o['hi']:,.0f}).
- The chance of exceeding the {o['capacity']:,.0f}-child capacity threshold peaks at **{o['prob']:.0%}**
  {f"and the forecast crosses it in **{o['lead']} day(s)**" if o['lead'] else "and the central forecast stays below it"}.
- Over the last 30 days intake {'exceeded' if e['net30'] > 0 else 'trailed'} discharges by
  **{abs(e['net30']):,.0f} children/day**. Projected net change over the horizon: **{o['net_total']:+,.0f}**.
- Expected discharges over the next {o['H']} days: **{o['dis_total']:,.0f}** sponsor placements.

## Recommended actions

{chr(10).join(f'{i}. {r}' for i, r in enumerate(rec, 1))}

## Staffing outlook (at assumed ratios)

{md_table(o['staff'], "{:,.0f}")}

## How reliable is the forecast?

Six models were tested on the most recent 8 weeks of history. **{best}** was the most accurate:
{b.loc[best, 'Accuracy (%)']:.1f}% accuracy, a typical error of ±{b.loc[best, 'MAE']:,.0f} children,
and an interval that contained the actual value {b.loc[best, 'Interval coverage (%)']:.0f}% of the time.
Forecasts 1–3 days ahead are most reliable, and error grows about
{b.loc[best, 'Error degradation']:.1f}× by day 14.

*Source: HHS UAC Program daily data. Generated by report.py. The live dashboard (streamlit run app.py)
has interactive horizons, model comparison and scenarios.*
"""
    path.write_text(s, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="HHS_Unaccompanied_Alien_Children_Program.csv")
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--capacity", type=float, default=None, help="default: 90th percentile of HHS load")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()

    out = Path(a.out)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    df = fx.prepare(fx.load_data(a.csv))
    capacity = a.capacity or round(df["in_hhs"].quantile(0.9), -2)

    print("EDA…")
    e = eda(df, figdir)
    print("Walk-forward evaluation of all models (a few minutes)…")
    ev = evaluation(df, figdir)
    best = ev["in_hhs"]["overall"].index[0]
    print(f"Forecasting with best model: {best}…")
    o = outlook(df, best, a.horizon, capacity, figdir)
    rec = recommendations(e, o, ev)
    write_paper(out / "research_paper.md", e, ev, o, rec)
    write_exec(out / "executive_summary.md", e, ev, o, rec)
    print(f"Wrote {out / 'research_paper.md'} and {out / 'executive_summary.md'}")


if __name__ == "__main__":
    main()
