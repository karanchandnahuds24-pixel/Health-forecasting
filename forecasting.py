"""Core forecasting library for the UAC care-load project.

Covers: data loading/cleaning, feature engineering, baseline / statistical /
ML forecasters (with prediction intervals), walk-forward validation, metrics,
KPIs and a simple stock-flow scenario engine.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

MODELS = [
    "Naive",
    "Moving Average",
    "SARIMA",
    "Exp. Smoothing",
    "Random Forest",
    "Gradient Boosting",
]
TARGET_LABELS = {
    "in_hhs": "Children in HHS care",
    "discharged": "Children discharged from HHS care",
    "transferred": "Children transferred out of CBP (HHS intake)",
}


# --------------------------------------------------------------------------
# 1. Loading & time-series preparation
# --------------------------------------------------------------------------
def _match(col: str) -> str | None:
    c = col.lower().strip()
    if c == "date":
        return "date"
    if "apprehended" in c:
        return "apprehended"
    if "transferred" in c:
        return "transferred"
    if "discharged" in c:
        return "discharged"
    if "in hhs" in c:
        return "in_hhs"
    if "in cbp" in c:
        return "in_cbp"
    return None


def load_data(src) -> pd.DataFrame:
    """Read the raw CSV (path or file-like) and return tidy short column names."""
    raw = pd.read_csv(src, thousands=",")
    mapping = {c: _match(c) for c in raw.columns if _match(c)}
    df = raw.rename(columns=mapping)[list(mapping.values())]
    missing = {"date", "apprehended", "in_cbp", "transferred", "in_hhs", "discharged"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")
    for c in df.columns.drop("date"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Datetime index, continuous daily frequency, interpolate gaps, flag imputed days."""
    d = df.set_index("date").asfreq("D")
    d["imputed"] = d[["in_hhs", "discharged", "transferred"]].isna().any(axis=1)
    cols = [c for c in d.columns if c != "imputed"]
    d[cols] = d[cols].interpolate(method="time", limit_direction="both")
    return d


def decompose(series: pd.Series, period: int = 7):
    """Additive decomposition into trend / seasonal / residual."""
    return seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")


# --------------------------------------------------------------------------
# 2. Feature engineering
# --------------------------------------------------------------------------
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """All features at row t use information available at or before t only."""
    X = pd.DataFrame(index=df.index)
    for col in ["in_hhs", "discharged", "transferred", "apprehended", "in_cbp"]:
        for lag in (0, 1, 7, 14):
            X[f"{col}_lag{lag}"] = df[col].shift(lag)
        for w in (7, 14):
            r = df[col].rolling(w)
            X[f"{col}_rmean{w}"] = r.mean()
            X[f"{col}_rvar{w}"] = r.var()
    net = df["transferred"] - df["discharged"]  # net pressure on HHS
    X["net_pressure"] = net
    X["net_pressure_rmean7"] = net.rolling(7).mean()
    X["net_pressure_rmean14"] = net.rolling(14).mean()
    X["dow"] = df.index.dayofweek
    X["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    X["month"] = df.index.month
    X = X.join(holiday_features(df.index))
    return X


def holiday_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """US federal holiday proxies: holiday flag and distance to the nearest holiday (capped at 14)."""
    hol = USFederalHolidayCalendar().holidays(idx.min() - pd.Timedelta(days=30), idx.max() + pd.Timedelta(days=30))
    days = idx.values.astype("datetime64[D]")
    h = hol.values.astype("datetime64[D]")
    pos = np.searchsorted(h, days)
    nxt = (h[np.minimum(pos, len(h) - 1)] - days).astype(int)
    prv = (days - h[np.maximum(pos - 1, 0)]).astype(int)
    return pd.DataFrame(
        {
            "is_holiday": idx.isin(hol).astype(int),
            "days_to_holiday": np.clip(np.where(nxt >= 0, nxt, 14), 0, 14),
            "days_since_holiday": np.clip(np.where(prv >= 0, prv, 14), 0, 14),
        },
        index=idx,
    )


def _base(df: pd.DataFrame, target: str) -> pd.Series:
    """Anchor the ML models forecast: they predict (future value - base)."""
    y = df[target]
    return y if target == "in_hhs" else y.rolling(7).mean()


# --------------------------------------------------------------------------
# 3. Forecasters  (each returns DataFrame[yhat, lo, hi] indexed by future date)
# --------------------------------------------------------------------------
def _future_index(df: pd.DataFrame, H: int) -> pd.DatetimeIndex:
    return pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=H, freq="D")


def _empirical_band(y: pd.Series, base: pd.Series, H: int, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-horizon quantiles of historical errors (y[t+h] - base[t])."""
    a = (1 - level) / 2
    lo, hi = [], []
    for h in range(1, H + 1):
        e = (y.shift(-h) - base).dropna().iloc[-365:]
        lo.append(e.quantile(a) if len(e) > 10 else 0.0)
        hi.append(e.quantile(1 - a) if len(e) > 10 else 0.0)
    return np.array(lo), np.array(hi)


def _baseline(df, target, H, level, window):
    y = df[target]
    base = y.rolling(window).mean()
    point = np.repeat(base.iloc[-1], H)
    lo_e, hi_e = _empirical_band(y, base, H, level)
    return point, point + lo_e, point + hi_e


def _sarima(df, target, H, level):
    m = SARIMAX(
        df[target], order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=100)
    f = m.get_forecast(H)
    ci = f.conf_int(alpha=1 - level)
    return f.predicted_mean.values, ci.iloc[:, 0].values, ci.iloc[:, 1].values


def _ets(df, target, H, level):
    y = df[target].astype(float)
    m = ETSModel(
        y, error="add", trend="add", damped_trend=True,
        seasonal="add", seasonal_periods=7,
    ).fit(disp=False)
    sf = m.get_prediction(start=len(y), end=len(y) + H - 1).summary_frame(alpha=1 - level)
    return sf["mean"].values, sf["pi_lower"].values, sf["pi_upper"].values


ML_FACTORY = {
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=42
    ),
    "Gradient Boosting": lambda: GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42
    ),
}


def _direct_ml(df, target, model, horizons, level, with_ci):
    """Direct multi-horizon: one model per horizon h predicting y[t+h] - base[t]."""
    X = make_features(df)
    base = _base(df, target)
    y = df[target]
    x_last = X.iloc[[-1]]
    base_last = base.iloc[-1]
    out = {}
    imputed = df["imputed"] if "imputed" in df else pd.Series(False, index=df.index)
    for h in horizons:
        d = pd.concat([X, (y.shift(-h) - base).rename("_t")], axis=1).dropna()
        # masking: never train on a target that was interpolated rather than reported
        d = d[~imputed.shift(-h, fill_value=False).reindex(d.index).astype(bool)]
        if len(d) < 60:
            raise ValueError("Not enough history to train the ML model.")
        feats = d.drop(columns="_t")
        t = d["_t"]
        lo_q = hi_q = 0.0
        if with_ci:  # residual quantiles from a time-ordered validation block
            k = int(len(d) * 0.8)
            m0 = ML_FACTORY[model]().fit(feats.iloc[:k], t.iloc[:k])
            res = t.iloc[k:] - m0.predict(feats.iloc[k:])
            a = (1 - level) / 2
            lo_q, hi_q = res.quantile(a), res.quantile(1 - a)
        m = ML_FACTORY[model]().fit(feats, t)
        p = base_last + m.predict(x_last[feats.columns])[0]
        out[h] = (p, p + lo_q, p + hi_q)
    return out


def forecast(
    df: pd.DataFrame, target: str, H: int, model: str,
    level: float = 0.90, horizons=None, with_ci: bool = True,
) -> pd.DataFrame:
    """Forecast `target` H days ahead with the chosen model.

    `horizons` (ML only / selection for others) restricts which horizons are returned.
    """
    idx = _future_index(df, H)
    if model in ML_FACTORY:
        hs = list(horizons) if horizons is not None else list(range(1, H + 1))
        res = _direct_ml(df, target, model, hs, level, with_ci)
        out = pd.DataFrame(
            {"yhat": [res[h][0] for h in hs], "lo": [res[h][1] for h in hs], "hi": [res[h][2] for h in hs]},
            index=[idx[h - 1] for h in hs],
        )
    else:
        fallback = False
        try:
            if model == "Naive":
                p, lo, hi = _baseline(df, target, H, level, 1)
            elif model == "Moving Average":
                p, lo, hi = _baseline(df, target, H, level, 7)
            elif model == "SARIMA":
                p, lo, hi = _sarima(df, target, H, level)
            elif model == "Exp. Smoothing":
                p, lo, hi = _ets(df, target, H, level)
            else:
                raise ValueError(f"Unknown model {model}")
        except Exception:  # numerical failure -> safe fallback
            if model in ("Naive", "Moving Average"):
                raise
            p, lo, hi = _baseline(df, target, H, level, 7)
            fallback = True
        out = pd.DataFrame({"yhat": p, "lo": lo, "hi": hi}, index=idx)
        if horizons is not None:
            out = out.loc[[idx[h - 1] for h in horizons]]
    out = out.clip(lower=0)
    # True when SARIMA/ETS failed to fit and the moving average was used instead
    out.attrs["fallback"] = model not in ML_FACTORY and fallback
    return out


# --------------------------------------------------------------------------
# 4. Walk-forward validation & metrics
# --------------------------------------------------------------------------
def walk_forward(
    df: pd.DataFrame, target: str, model: str,
    horizons=(1, 3, 7, 14), n_origins: int = 8, step: int = 7, level: float = 0.90,
) -> pd.DataFrame:
    """Expanding-window walk-forward: refit at each origin, forecast max(horizons) ahead.

    Also records the prediction interval so its empirical coverage can be checked.
    Interpolated (imputed) days are never scored.
    """
    H = max(horizons)
    last_origin = len(df) - H - 1
    origins = [last_origin - step * i for i in range(n_origins)][::-1]
    imputed = df["imputed"] if "imputed" in df else pd.Series(False, index=df.index)
    rows = []
    for o in origins:
        if o < 120:
            continue
        train = df.iloc[: o + 1]
        fc = forecast(train, target, H, model, level=level, horizons=horizons, with_ci=True)
        for h in horizons:
            date = train.index[-1] + pd.Timedelta(days=h)
            if imputed.loc[date]:
                continue
            rows.append(
                dict(model=model, origin=train.index[-1], horizon=h,
                     actual=df[target].loc[date], pred=fc.loc[date, "yhat"],
                     lo=fc.loc[date, "lo"], hi=fc.loc[date, "hi"], fallback=fc.attrs.get("fallback", False))
            )
    return pd.DataFrame(rows)


def evaluate_models(df, target, models, horizons=(1, 3, 7, 14), n_origins=8, step=7, level=0.90) -> pd.DataFrame:
    return pd.concat(
        [walk_forward(df, target, m, horizons, n_origins, step, level) for m in models], ignore_index=True
    )


def _metrics(g: pd.DataFrame) -> pd.Series:
    err = g["pred"] - g["actual"]
    nz = g["actual"] != 0
    out = {
        "MAE": err.abs().mean(),
        "RMSE": np.sqrt((err**2).mean()),
        "MAPE (%)": (err[nz].abs() / g.loc[nz, "actual"]).mean() * 100,
    }
    if {"lo", "hi"} <= set(g.columns):  # uncertainty quality
        out["Interval coverage (%)"] = ((g["actual"] >= g["lo"]) & (g["actual"] <= g["hi"])).mean() * 100
        out["Mean interval width"] = (g["hi"] - g["lo"]).mean()
    return pd.Series(out)


def metrics_by_horizon(recs: pd.DataFrame) -> pd.DataFrame:
    return recs.groupby(["model", "horizon"]).apply(_metrics).reset_index()


def metrics_overall(recs: pd.DataFrame) -> pd.DataFrame:
    return recs.groupby("model").apply(_metrics).reset_index()


# --------------------------------------------------------------------------
# 5. KPIs
# --------------------------------------------------------------------------
def forecast_accuracy(recs_m: pd.DataFrame) -> float:
    """100 - MAPE (%), floored at 0."""
    return max(0.0, 100 - _metrics(recs_m)["MAPE (%)"])


def stability_index(recs_m: pd.DataFrame) -> float:
    """1 - coefficient of variation of per-origin MAE (1 = very stable, 0 = erratic)."""
    per = recs_m.assign(e=(recs_m.pred - recs_m.actual).abs()).groupby("origin")["e"].mean()
    if len(per) < 2 or per.mean() == 0:
        return 1.0
    return float(np.clip(1 - per.std() / per.mean(), 0, 1))


def error_degradation(recs_m: pd.DataFrame) -> float:
    """Robustness: MAE at the longest horizon / MAE at the shortest (lower = more robust)."""
    mae = recs_m.assign(e=(recs_m.pred - recs_m.actual).abs()).groupby("horizon")["e"].mean()
    return float(mae.iloc[-1] / mae.iloc[0]) if mae.iloc[0] > 0 else np.nan


def breach_probability(fc: pd.DataFrame, capacity: float, level: float = 0.90) -> pd.Series:
    """P(load > capacity) per day, treating the interval as a normal approximation."""
    z = norm.ppf((1 + level) / 2)
    sigma = ((fc["hi"] - fc["lo"]) / (2 * z)).clip(lower=1e-6)
    return pd.Series(1 - norm.cdf((capacity - fc["yhat"]) / sigma), index=fc.index)


def surge_lead_time(fc: pd.DataFrame, capacity: float) -> int | None:
    """Days until the point forecast first exceeds capacity (None if it doesn't)."""
    over = np.where(fc["yhat"].values > capacity)[0]
    return int(over[0] + 1) if len(over) else None


def historical_surge_lead_time(
    df: pd.DataFrame, model: str, capacity: float, H: int = 14, level: float = 0.90,
    min_gap: int = 30, max_events: int = 5,
) -> pd.DataFrame:
    """Backtest the early warning: for past capacity crossings, how many days ahead did it fire?

    An event is a day the actual load crosses above `capacity` (at least `min_gap` days after
    the previous crossing). For each event we re-run the forecast from each of the H days before
    it and report the earliest origin whose upper interval already exceeded capacity on the event day.
    """
    y = df["in_hhs"]
    cross = np.where((y.values[1:] > capacity) & (y.values[:-1] <= capacity))[0] + 1
    events, last = [], -10**9
    for i in cross:
        if i - last >= min_gap and i > 120 + H:
            events.append(i)
            last = i
    rows = []
    for i in events[-max_events:]:
        lead = 0
        for k in range(H, 0, -1):  # earliest origin first
            fc = forecast(df.iloc[: i - k + 1], "in_hhs", k, model, level=level, horizons=[k])
            if fc["hi"].iloc[-1] > capacity:
                lead = k
                break
        rows.append(dict(event_date=df.index[i], load=y.iloc[i], lead_days=lead))
    return pd.DataFrame(rows)


def staffing_needs(load: pd.Series, ratios: dict[str, float]) -> pd.DataFrame:
    """Translate a care-load forecast into staff required: ceil(load / children-per-staff)."""
    return pd.DataFrame({role: np.ceil(load / r) for role, r in ratios.items() if r > 0}, index=load.index)


# --------------------------------------------------------------------------
# 6. Scenario engine (stock-flow: dHHS = transfers in - discharges out)
# --------------------------------------------------------------------------
def scenario_path(base_hhs: pd.DataFrame, transfers_fc: pd.DataFrame, discharge_fc: pd.DataFrame,
                  transfer_change: float, discharge_change: float) -> pd.Series:
    """Shift the baseline care-load forecast by the cumulative effect of flow changes.

    transfer_change / discharge_change are fractions (0.25 = +25 %).
    """
    delta = transfer_change * transfers_fc["yhat"] - discharge_change * discharge_fc["yhat"]
    return (base_hhs["yhat"] + delta.cumsum()).clip(lower=0)
