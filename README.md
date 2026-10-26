# UAC Care Load Forecasting

Predictive analytics for the HHS Unaccompanied Alien Children (UAC) Program: short-term forecasts of
children in HHS care and discharge demand, early warnings of capacity stress, and quantified uncertainty.

## Run
    pip install -r requirements.txt
    # put HHS_Unaccompanied_Alien_Children_Program.csv next to app.py (or upload it in the sidebar)
    streamlit run app.py                 # live dashboard
    python report.py                     # research paper + executive summary (takes a few minutes)
    python report.py data.csv --horizon 14 --capacity 12000 --out reports

## Files
- forecasting.py : loading, cleaning, features, 6 models, walk-forward validation, KPIs, scenarios, staffing
- app.py         : Streamlit dashboard (forecast, discharges, model comparison, scenarios, EDA)
- report.py      : writes the deliverables from the data:
  - reports/research_paper.md     (EDA, methodology, results, recommendations, limitations)
  - reports/executive_summary.md  (one-page brief for government stakeholders)
  - reports/figures/*.png
  All numbers are computed from the dataset. The narrative is a first draft: review it and add domain
  context before submitting. Convert to Word/PDF with e.g. `pandoc research_paper.md -o paper.docx`.

## Deliverables checklist
| Deliverable | Where |
|---|---|
| Research paper (EDA, insights, recommendations) | `python report.py` → reports/research_paper.md |
| Streamlit dashboard (live analytics) | `streamlit run app.py` (deploy: Streamlit Community Cloud) |
| Executive summary for stakeholders | `python report.py` → reports/executive_summary.md |

## Method notes
- Preparation: continuous daily index; missing days time-interpolated **and masked** (flagged as
  `imputed`, never used as training targets or scored).
- Features: lags t/t-1/t-7/t-14, 7/14-day rolling mean & variance, net pressure (transfers − discharges),
  day of week, weekend, month, US federal holiday proxies (holiday flag, days to/since holiday).
- ML models use direct multi-horizon forecasting (one model per horizon, target = future value - anchor).
- Intervals: SARIMA/ETS native; baselines & ML use empirical residual quantiles. Calibration is checked
  by interval coverage in walk-forward validation.
- If SARIMA/ETS fail to fit, they fall back to the moving average, and the dashboard says so.
- Validation: expanding-window walk-forward, 8 origins x horizons 1/3/7/14 days.
- KPIs: Accuracy = 100 - MAPE; Stability = 1 - CV(per-origin MAE); Robustness = MAE(d14)/MAE(d1);
  Breach probability = normal approx from interval; Surge lead time = days until forecast > capacity,
  plus a backtest of how many days ahead past capacity crossings were flagged.
- Staffing: forecast load ÷ children-per-staff ratios. The default ratios are placeholders, so set your
  program's real ratios in the sidebar or in `RATIOS` in report.py.
