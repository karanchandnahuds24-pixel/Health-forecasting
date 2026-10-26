"""UAC Program – Care Load Forecasting dashboard.  Run:  streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import forecasting as fx

st.set_page_config(page_title="UAC Care Load Forecasting", layout="wide")
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
DEFAULT_CSV = "HHS_Unaccompanied_Alien_Children_Program.csv"


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Loading data…")
def get_data(file_bytes, path):
    src = file_bytes if file_bytes is not None else path
    return fx.prepare(fx.load_data(src))


@st.cache_data(show_spinner="Forecasting…")
def run_forecast(df, target, H, model, level):
    return fx.forecast(df, target, H, model, level)


@st.cache_data(show_spinner="Running walk-forward validation (this can take a minute)…")
def run_eval(df, target, models, horizons, level):
    return fx.evaluate_models(df, target, list(models), horizons, level=level)


@st.cache_data(show_spinner="Backtesting early warnings on past capacity crossings…")
def run_surge_backtest(df, model, capacity, level):
    return fx.historical_surge_lead_time(df, model, capacity, level=level)


# --------------------------------------------------------------- sidebar
st.sidebar.title("Controls")
upload = st.sidebar.file_uploader("Upload UAC CSV", type="csv")
try:
    df = get_data(upload, DEFAULT_CSV) if upload is None else get_data(upload, None)
except Exception as e:
    st.title("UAC Care Load Forecasting")
    st.info(f"Upload the dataset CSV in the sidebar (or place `{DEFAULT_CSV}` next to app.py).\n\nDetails: {e}")
    st.stop()

H = st.sidebar.slider("Forecast horizon (days)", 7, 30, 14)
models = st.sidebar.multiselect("Models (first = primary)", fx.MODELS, default=["SARIMA", "Gradient Boosting"])
if not models:
    st.warning("Select at least one model.")
    st.stop()
level = st.sidebar.select_slider("Confidence level", [0.80, 0.90, 0.95], value=0.90, format_func=lambda v: f"{int(v*100)}%")
default_cap = int(round(df["in_hhs"].quantile(0.9), -2))
capacity = st.sidebar.number_input("Capacity threshold (children in HHS care)", 1000, 100000, default_cap, 100)
primary = models[0]
hist_days = st.sidebar.slider("History shown (days)", 30, 365, 120)
with st.sidebar.expander("Staffing ratios (children per staff member)"):
    st.caption("Planning assumptions – replace with your program's actual ratios.")
    ratios = {
        "Case managers": st.number_input("Case manager", 1, 100, 12),
        "Youth care workers": st.number_input("Youth care worker", 1, 100, 8),
        "Clinicians": st.number_input("Clinician", 1, 200, 25),
        "Medical staff": st.number_input("Medical staff", 1, 500, 50),
    }

st.title("UAC Program – Care Load Forecasting")
st.caption(f"Data through {df.index[-1]:%d %b %Y} · {len(df):,} daily observations · "
           f"{int(df['imputed'].sum())} imputed days")


def forecast_chart(target, title):
    fig = go.Figure()
    hist = df[target].iloc[-hist_days:]
    fig.add_scatter(x=hist.index, y=hist, name="Actual", line=dict(color="black"))
    for i, m in enumerate(models):
        fc = run_forecast(df, target, H, m, level)
        c = COLORS[i % len(COLORS)]
        if m == primary:
            fig.add_scatter(x=list(fc.index) + list(fc.index[::-1]), y=list(fc.hi) + list(fc.lo[::-1]),
                            fill="toself", fillcolor="rgba(31,119,180,0.15)", line=dict(width=0),
                            name=f"{int(level*100)}% interval ({m})", hoverinfo="skip")
        fig.add_scatter(x=fc.index, y=fc.yhat, name=m, line=dict(color=c, dash="dash" if m != primary else "solid"))
    if target == "in_hhs":
        fig.add_hline(y=capacity, line_dash="dot", line_color="red", annotation_text="Capacity threshold")
    fig.update_layout(title=title, height=460, hovermode="x unified", legend=dict(orientation="h", y=-0.15))
    return fig


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Care load forecast", "Discharge demand", "Model comparison", "Scenarios", "Data & EDA"]
)

# ------------------------------------------------------- 1. care load
with tab1:
    fc = run_forecast(df, "in_hhs", H, primary, level)
    prob = fx.breach_probability(fc, capacity, level)
    lead = fx.surge_lead_time(fc, capacity)
    net7 = (df["transferred"] - df["discharged"]).iloc[-7:].mean()

    c = st.columns(5)
    c[0].metric("Current load", f"{df['in_hhs'].iloc[-1]:,.0f}")
    c[1].metric(f"Forecast (day {H})", f"{fc.yhat.iloc[-1]:,.0f}", f"{fc.yhat.iloc[-1]-df['in_hhs'].iloc[-1]:+,.0f}")
    c[2].metric("7-day net pressure", f"{net7:+,.0f}/day", help="Transfers into HHS minus discharges, 7-day average")
    c[3].metric("Max breach probability", f"{prob.max():.0%}")
    c[4].metric("Surge lead time", f"{lead} days" if lead else "None in horizon")

    if prob.max() >= 0.5 or lead:
        st.error(f"⚠️ Early warning: {primary} projects care load above {capacity:,} "
                 + (f"in {lead} days." if lead else "with high probability."))
    elif prob.max() >= 0.2 or net7 > 0:
        st.warning("Watch: intake is outpacing discharges or breach risk is non-trivial.")
    else:
        st.success("No capacity stress projected within the selected horizon.")

    if fc.attrs.get("fallback"):
        st.info(f"{primary} failed to fit on this data; the 7-day moving average is shown in its place.")

    st.plotly_chart(forecast_chart("in_hhs", "Children in HHS care – forecast"), width="stretch")
    with st.expander("Forecast table"):
        table = fc.assign(breach_prob=prob).round(2)
        st.dataframe(table)
        st.download_button("Download forecast (CSV)", table.to_csv().encode(), "hhs_care_load_forecast.csv",
                           "text/csv")

    st.subheader("Staffing implied by the forecast")
    st.caption("Staff needed = forecast load ÷ children per staff member (ratios set in the sidebar). "
               "Upper bound uses the top of the prediction interval.")
    need_now = fx.staffing_needs(df["in_hhs"].iloc[[-1]], ratios).iloc[0]
    need_peak = fx.staffing_needs(pd.Series([fc.yhat.max()]), ratios).iloc[0]
    need_hi = fx.staffing_needs(pd.Series([fc.hi.max()]), ratios).iloc[0]
    st.dataframe(pd.DataFrame({
        "Needed today": need_now,
        f"Needed at forecast peak ({primary})": need_peak,
        "Additional vs today": need_peak - need_now,
        f"Upper bound ({int(level*100)}% interval)": need_hi,
    }).astype(int), width="stretch")

# ----------------------------------------------------- 2. discharges
with tab2:
    st.plotly_chart(forecast_chart("discharged", "Daily discharges (sponsor placements) – forecast"),
                    width="stretch")
    tr = run_forecast(df, "transferred", H, primary, level)
    dis = run_forecast(df, "discharged", H, primary, level)
    net = tr.yhat - dis.yhat
    fig = go.Figure()
    fig.add_bar(x=net.index, y=net, marker_color=np.where(net > 0, "#d62728", "#2ca02c"), name="Net pressure")
    fig.update_layout(title=f"Projected net pressure (transfers − discharges), {primary}", height=320)
    st.plotly_chart(fig, width="stretch")
    st.caption("Red days: more children entering HHS than leaving → care load rising.")
    st.metric("Cumulative net change over horizon", f"{net.sum():+,.0f} children")

# ------------------------------------------------- 3. model comparison
with tab3:
    tgt = st.radio("Target", ["in_hhs", "discharged"], format_func=fx.TARGET_LABELS.get, horizontal=True)
    horizons = (1, 3, 7, 14)
    recs = run_eval(df, tgt, tuple(models), horizons, level)
    if recs.empty:
        st.warning("Not enough history for walk-forward validation.")
    else:
        mo = fx.metrics_overall(recs)
        mo["Accuracy (%)"] = [fx.forecast_accuracy(recs[recs.model == m]) for m in mo.model]
        mo["Stability index"] = [fx.stability_index(recs[recs.model == m]) for m in mo.model]
        mo["Error degradation"] = [fx.error_degradation(recs[recs.model == m]) for m in mo.model]
        best = mo.sort_values("MAE").iloc[0]
        st.subheader("KPI summary – best model by MAE")
        k = st.columns(4)
        k[0].metric("Best model", best.model)
        k[1].metric("Forecast accuracy", f"{best['Accuracy (%)']:.1f}%")
        k[2].metric("Stability index", f"{best['Stability index']:.2f}")
        k[3].metric(f"Interval coverage (target {int(level*100)}%)", f"{best['Interval coverage (%)']:.0f}%")
        st.subheader("Overall (walk-forward, 8 origins)")
        st.dataframe(mo.round(2).set_index("model"), width="stretch")
        st.caption("Stability: 1 − CV of per-origin MAE (higher = steadier). "
                   "Error degradation: MAE at day 14 ÷ MAE at day 1 (lower = more robust). "
                   f"Interval coverage: share of actuals inside the {int(level*100)}% interval "
                   "(close to the nominal level = well-calibrated uncertainty).")
        fell_back = sorted(recs.loc[recs.fallback.astype(bool), "model"].unique())
        if fell_back:
            st.info(f"{', '.join(fell_back)} fell back to the moving average at some origins.")
        mh = fx.metrics_by_horizon(recs)
        metric = st.selectbox("Metric by horizon", ["MAE", "RMSE", "MAPE (%)"])
        fig = go.Figure()
        for i, m in enumerate(models):
            s = mh[mh.model == m]
            fig.add_scatter(x=s.horizon, y=s[metric], name=m, mode="lines+markers", line=dict(color=COLORS[i % 6]))
        fig.update_layout(xaxis_title="Horizon (days)", yaxis_title=metric, height=380)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(mh.round(2), width="stretch")

    st.subheader("Surge lead time – backtest on past capacity crossings")
    st.caption("For each past day the care load crossed the capacity threshold, how many days earlier "
               f"did {primary}'s upper interval already show the breach? (0 = no advance warning)")
    if st.button("Run surge backtest"):
        sb = run_surge_backtest(df, primary, capacity, level)
        if sb.empty:
            st.write("No past crossings of this threshold in the data.")
        else:
            st.metric("Average advance warning", f"{sb.lead_days.mean():.1f} days")
            st.dataframe(sb, width="stretch")

# ------------------------------------------------------- 4. scenarios
with tab4:
    st.write("Stock-flow what-if: care load changes by (transfers in − discharges out). "
             "Adjust flows relative to the baseline forecast.")
    a, b = st.columns(2)
    t_chg = a.slider("Change in transfers into HHS (%)", -50, 100, 25) / 100
    d_chg = b.slider("Change in discharges (%)", -50, 50, -20) / 100
    base = run_forecast(df, "in_hhs", H, primary, level)
    tr = run_forecast(df, "transferred", H, primary, level)
    dis = run_forecast(df, "discharged", H, primary, level)
    scen = {
        "Baseline": base.yhat,
        "Intake surge": fx.scenario_path(base, tr, dis, t_chg, 0),
        "Slow discharges": fx.scenario_path(base, tr, dis, 0, d_chg),
        "Combined": fx.scenario_path(base, tr, dis, t_chg, d_chg),
    }
    fig = go.Figure()
    hist = df["in_hhs"].iloc[-60:]
    fig.add_scatter(x=hist.index, y=hist, name="Actual", line=dict(color="black"))
    for i, (k, v) in enumerate(scen.items()):
        fig.add_scatter(x=v.index, y=v, name=k, line=dict(color=COLORS[i]))
    fig.add_hline(y=capacity, line_dash="dot", line_color="red")
    fig.update_layout(height=450, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame({
        "End-of-horizon load": {k: v.iloc[-1] for k, v in scen.items()},
        "Peak load": {k: v.max() for k, v in scen.items()},
        "Days above capacity": {k: int((v > capacity).sum()) for k, v in scen.items()},
    }).round(0), width="stretch")

# ----------------------------------------------------------- 5. EDA
with tab5:
    cols = st.multiselect("Series", ["in_hhs", "in_cbp", "apprehended", "transferred", "discharged"],
                          default=["in_hhs", "transferred", "discharged"])
    fig = go.Figure()
    for c_ in cols:
        fig.add_scatter(x=df.index, y=df[c_], name=c_)
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")

    dec = fx.decompose(df["in_hhs"])
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=["Observed", "Trend", "Weekly seasonality", "Residual"])
    for r, comp in enumerate([dec.observed, dec.trend, dec.seasonal, dec.resid], start=1):
        fig.add_scatter(x=comp.index, y=comp, showlegend=False, row=r, col=1)
    fig.update_layout(height=650, title="Decomposition – children in HHS care")
    st.plotly_chart(fig, width="stretch")
    st.write("Correlation of daily flows")
    st.dataframe(df[["apprehended", "in_cbp", "transferred", "in_hhs", "discharged"]].corr().round(2))

    st.write("Average flow by day of week (reported days only)")
    obs = df[~df["imputed"]]
    dow = obs.groupby(obs.index.dayofweek)[["apprehended", "transferred", "discharged"]].mean()
    dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(dow)]
    st.bar_chart(dow)

    with st.expander(f"Data quality – {int(df['imputed'].sum())} missing days filled by time interpolation"):
        gaps = df.index[df["imputed"]]
        st.write("Imputed days are excluded from model training targets and from accuracy scoring.")
        st.dataframe(pd.DataFrame({"date": gaps, "weekday": gaps.day_name()}), width="stretch")
