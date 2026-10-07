"""Streamlit interface for the Nigerian car price predictor.

    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from carprice import REFERENCE_YEAR, add_features, build_input_row, clean_raw, nice_model_name

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "car_price_model.joblib"
META_PATH = ROOT / "models" / "model_meta.json"
COMPARISON_PATH = ROOT / "models" / "model_comparison.csv"
DATA_RAW = ROOT / "data" / "autochek_cars_ng.csv"

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e0"
CONDITIONS = {"Foreign used (tokunbo)": "foreign", "Locally used (Nigerian used)": "local", "Brand new": "new"}

st.set_page_config(page_title="Naija Car Price Estimator", page_icon="🚗", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1200px;}
  .price-card {border: 1px solid #e6e5e0; border-radius: 14px; padding: 1.4rem 1.6rem; background: #fcfcfb;}
  .price-label {color: #52514e; font-size: 0.95rem; margin-bottom: 0.2rem;}
  .price-value {color: #0b0b0b; font-size: 2.6rem; font-weight: 700; line-height: 1.15;}
  .price-range {color: #52514e; font-size: 1rem; margin-top: 0.35rem;}
  .note {color: #52514e; font-size: 0.85rem;}
  @media (prefers-color-scheme: dark) {
    .price-card {background: #1a1a19; border-color: #3a3a38;}
    .price-label, .price-range, .note {color: #c3c2b7;}
    .price-value {color: #ffffff;}
  }
</style>
""", unsafe_allow_html=True)


def naira(v: float) -> str:
    if v >= 1e9:
        return f"₦{v / 1e9:,.2f}B"
    return f"₦{v / 1e6:,.1f}M"


# ---------------------------------------------------------------- data & model
@st.cache_data(show_spinner=False)
def load_data():
    df, names = clean_raw(pd.read_csv(DATA_RAW), return_display_names=True)
    return add_features(df), names


@st.cache_resource(show_spinner="Loading the price model…")
def load_model():
    """Load the saved model; if it can't be loaded (e.g. library version mismatch), retrain it."""
    try:
        model = joblib.load(MODEL_PATH)
        meta = json.loads(META_PATH.read_text())
        build_input_row(make="Toyota", make_model="Toyota CAMRY", year=2015, condition="foreign",
                        mileage_km=100_000, body_type="Sedan", cylinders=4).pipe(model.predict)
        return model, meta
    except Exception:  # noqa: BLE001
        from train import train
        return train(save=True)


df, display_names = load_data()
model, meta = load_model()
q10, q90 = meta["interval_log_ratio"]["q10"], meta["interval_log_ratio"]["q90"]


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=13), title_font=dict(color=INK, size=15),
                      hoverlabel=dict(bgcolor="white", font_color=INK), showlegend=False)
    fig.update_xaxes(showgrid=False, linecolor="#b8b7b1")
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ---------------------------------------------------------------- header
st.title("🚗 Naija Car Price Estimator")
st.caption(f"Estimate the fair asking price of a car in Nigeria. Model trained on {len(df):,} listings from "
           f"Autochek Nigeria (October 2026). Typical error ±{meta['metrics']['median_ape']:.0%}.")

tab_predict, tab_market, tab_model = st.tabs(["💰 Price estimator", "📊 Market insights", "🧠 Model performance"])

# ================================================================ TAB 1: PREDICT
with tab_predict:
    left, right = st.columns([1, 1.35], gap="large")

    with left:
        st.subheader("Describe the car")
        makes = df["make"].value_counts()
        makes = makes[makes >= 3].index.tolist()
        make = st.selectbox("Make", makes, index=makes.index("Toyota"))

        mm_counts = df.loc[df["make"] == make, "make_model"].value_counts()
        mm_options = mm_counts[mm_counts >= 3].index.tolist() or mm_counts.index.tolist()
        default_mm = f"{make} CAMRY" if f"{make} CAMRY" in mm_options else mm_options[0]
        make_model = st.selectbox("Model", mm_options, index=mm_options.index(default_mm),
                                  format_func=lambda m: f"{nice_model_name(make, m, display_names)}  ({mm_counts[m]} listings)")

        same = df[df["make_model"] == make_model]
        c1, c2 = st.columns(2)
        years = list(range(REFERENCE_YEAR, 1998, -1))
        default_year = int(same["year"].median())
        year = c1.selectbox("Year", years, index=years.index(default_year))
        condition_label = c2.selectbox("Condition", list(CONDITIONS))
        condition = CONDITIONS[condition_label]

        c3, c4 = st.columns([1.6, 1])
        unit = c4.radio("Unit", ["km", "miles"], horizontal=True)
        med_km = same["mileage_km"].median()
        med_km = df["mileage_km"].median() if pd.isna(med_km) else med_km
        default_mileage = 0 if condition == "new" else int(round(med_km / (1.60934 if unit == "miles" else 1), -3))
        mileage = c3.number_input(f"Mileage ({unit})", min_value=0, max_value=1_000_000, step=5_000, value=default_mileage,
                                  help="Enter 0 if unknown: the model will assume a typical mileage.")
        mileage_km = None if (mileage == 0 and condition != "new") else mileage * (1.60934 if unit == "miles" else 1)

        body_types = ["Sedan", "SUV", "Pickup", "Minivan", "Hatchback/Wagon", "Coupe/Sports", "Van/Bus", "Other"]
        cyl_opts = [2, 3, 4, 5, 6, 8, 10, 12]
        default_body = same["body_type"].mode().iat[0]
        default_cyl = int(same["cylinders"].mode().iat[0]) if same["cylinders"].notna().any() else 4
        c5, c6 = st.columns(2)
        body_type = c5.selectbox("Body type", body_types, index=body_types.index(default_body))
        cylinders = c6.selectbox("Engine cylinders", cyl_opts, index=cyl_opts.index(default_cyl) if default_cyl in cyl_opts else 2)

        c7, c8 = st.columns(2)
        transmission = c7.selectbox("Transmission", ["automatic", "cvt", "manual", "other"], format_func=str.capitalize)
        state = c8.selectbox("Location", ["Lagos", "Abuja", "Rivers", "Other", "Unknown"],
                             format_func=lambda s: {"Rivers": "Rivers (Port Harcourt)", "Other": "Other state", "Unknown": "Not stated"}.get(s, s))
        with st.expander("More options"):
            fuel_type = st.selectbox("Fuel", ["petrol", "non-petrol"], format_func=lambda s: "Petrol" if s == "petrol" else "Diesel / hybrid / electric")
            inspected = st.selectbox("Inspected by Autochek", ["unknown", "yes", "no"], format_func=str.capitalize)

    x = build_input_row(make=make, make_model=make_model, year=year, condition=condition, mileage_km=mileage_km,
                        body_type=body_type, cylinders=cylinders, transmission=transmission, fuel_type=fuel_type,
                        state=state, inspected=inspected)
    price = float(model.predict(x)[0])
    low, high = price * np.exp(q10), price * np.exp(q90)
    car_name = f"{year} {make} {nice_model_name(make, make_model, display_names)}"

    with right:
        st.subheader("Estimated price")
        st.markdown(
            f'<div class="price-card"><div class="price-label">{car_name} · {condition_label.split(" (")[0].lower()}</div>'
            f'<div class="price-value">{naira(price)}</div>'
            f'<div class="price-range">Likely range <b>{naira(low)} – {naira(high)}</b> '
            f'(80% of similar listings fall in this range)</div></div>', unsafe_allow_html=True)

        # Similar listings in the data
        similar = df[(df["make_model"] == make_model) & (df["year"].between(year - 1, year + 1))]
        same_cond = similar[similar["condition"] == condition]
        comp = same_cond if len(same_cond) >= 3 else similar
        st.write("")
        if len(comp) >= 3:
            k1, k2, k3 = st.columns(3)
            k1.metric("Similar listings", f"{len(comp):,}", help=f"Same model, {year - 1}–{year + 1}"
                      + ("" if comp is similar else f", {condition_label.split(' (')[0].lower()}"))
            k2.metric("Their median price", naira(comp["price_ngn"].median()))
            diff = price / comp["price_ngn"].median() - 1
            k3.metric("Estimate vs median", f"{diff:+.0%}")

            fig = go.Figure()
            jitter = np.random.default_rng(0).uniform(-0.25, 0.25, len(comp))
            fig.add_trace(go.Scatter(
                x=comp["price_ngn"], y=jitter, mode="markers",
                marker=dict(size=10, color=BLUE, opacity=0.55, line=dict(width=1, color="white")),
                customdata=np.stack([comp["year"], comp["condition"], (comp["mileage_km"].fillna(-1) / 1000).round()], axis=1),
                hovertemplate="₦%{x:,.0f}<br>%{customdata[0]} · %{customdata[1]}<br>%{customdata[2]}k km<extra></extra>"))
            fig.add_vrect(x0=low, x1=high, fillcolor=ORANGE, opacity=0.08, line_width=0)
            fig.add_vline(x=price, line_color=ORANGE, line_width=3)
            fig.add_annotation(x=price, y=0.42, text=f"Estimate {naira(price)}", showarrow=False, font=dict(color=INK, size=12),
                               bgcolor="rgba(255,255,255,0.85)")
            fig.update_yaxes(visible=False, range=[-0.5, 0.55])
            fig.update_xaxes(title="Listed price (₦)", tickformat="~s", tickprefix="₦")
            st.plotly_chart(style_fig(fig, 230).update_layout(title="Your estimate vs similar cars on the market"),
                            width="stretch")
        else:
            st.info("Few similar listings in the data for this exact model and year, so treat the estimate with extra caution.")

        # How price changes with model year (model's view)
        yrs = list(range(max(2005, year - 8), min(REFERENCE_YEAR, year + 4) + 1))
        rows = pd.concat([build_input_row(make=make, make_model=make_model, year=yy, condition=condition,
                                          mileage_km=mileage_km, body_type=body_type, cylinders=cylinders,
                                          transmission=transmission, fuel_type=fuel_type, state=state,
                                          inspected=inspected) for yy in yrs])
        curve = model.predict(rows)
        fig2 = go.Figure(go.Scatter(x=yrs, y=curve, mode="lines+markers", line=dict(color=BLUE, width=2.5),
                                    marker=dict(size=8, color=[ORANGE if yy == year else BLUE for yy in yrs]),
                                    hovertemplate="%{x}: ₦%{y:,.0f}<extra></extra>"))
        fig2.update_yaxes(title="Estimated price (₦)", tickformat="~s", tickprefix="₦")
        fig2.update_xaxes(title="Model year", dtick=1)
        st.plotly_chart(style_fig(fig2, 300).update_layout(title=f"Estimated price of this {nice_model_name(make, make_model, display_names)} by model year"),
                        width="stretch")
        st.markdown('<p class="note">Estimates reflect asking prices on Autochek in October 2026, not final sale prices. '
                    'Trim, colour, accident history and the car\'s actual condition can move the real price.</p>',
                    unsafe_allow_html=True)

# ================================================================ TAB 2: MARKET
with tab_market:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Listings analysed", f"{len(df):,}")
    k2.metric("Median asking price", naira(df["price_ngn"].median()))
    k3.metric("Foreign-used share", f"{(df['condition'] == 'foreign').mean():.0%}")
    k4.metric(f"Most-listed car ({(df['make_model'] == 'Toyota CAMRY').mean():.0%} of listings)", "Toyota Camry")

    a, b = st.columns(2, gap="large")
    with a:
        mk = df.groupby("make")["price_ngn"].agg(["median", "count"]).query("count >= 15").sort_values("median")
        fig = go.Figure(go.Bar(x=mk["median"], y=mk.index, orientation="h", marker_color=BLUE,
                               customdata=mk["count"], hovertemplate="%{y}: ₦%{x:,.0f}<br>%{customdata} listings<extra></extra>"))
        fig.update_xaxes(title="Median asking price (₦)", tickformat="~s", tickprefix="₦")
        st.plotly_chart(style_fig(fig, 460).update_layout(title="Median price by make (≥15 listings)"), width="stretch")
    with b:
        fig = go.Figure()
        for mk_name, color in zip(["Toyota", "Lexus", "Mercedes-Benz"], [BLUE, ORANGE, AQUA]):
            s = df[(df["make"] == mk_name) & (df["car_age"].between(1, 20))].groupby("year")["price_ngn"].median()
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=mk_name, mode="lines+markers",
                                     line=dict(color=color, width=2.5), marker=dict(size=6),
                                     hovertemplate=f"{mk_name} %{{x}}: ₦%{{y:,.0f}}<extra></extra>"))
        ticks = [5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8]
        fig.update_yaxes(type="log", title="Median asking price (log scale)", tickvals=ticks,
                         ticktext=[naira(t).replace(".0M", "M") for t in ticks])
        fig.update_xaxes(title="Model year")
        style_fig(fig, 460).update_layout(title="Depreciation: median price by model year", showlegend=True,
                                          legend=dict(orientation="h", y=-0.18, x=0, yanchor="top"),
                                          margin=dict(l=10, r=10, t=40, b=60))
        st.plotly_chart(fig, width="stretch")

    c, d = st.columns(2, gap="large")
    with c:
        cond = df.groupby("condition")["price_ngn"].median().reindex(["local", "foreign", "new"])
        fig = go.Figure(go.Bar(x=["Locally used", "Foreign used", "Brand new"], y=cond.values, marker_color=BLUE,
                               text=[naira(v) for v in cond.values], textposition="outside",
                               hovertemplate="%{x}: ₦%{y:,.0f}<extra></extra>"))
        fig.update_yaxes(title="Median asking price (₦)", tickformat="~s", tickprefix="₦")
        st.plotly_chart(style_fig(fig, 360).update_layout(title="Median price by condition"), width="stretch")
    with d:
        bt = df.groupby("body_type")["price_ngn"].agg(["median", "count"]).query("count >= 20").sort_values("median")
        fig = go.Figure(go.Bar(x=bt.index, y=bt["median"], marker_color=BLUE, text=[naira(v) for v in bt["median"]],
                               textposition="outside", customdata=bt["count"],
                               hovertemplate="%{x}: ₦%{y:,.0f}<br>%{customdata} listings<extra></extra>"))
        fig.update_yaxes(title="Median asking price (₦)", tickformat="~s", tickprefix="₦")
        st.plotly_chart(style_fig(fig, 360).update_layout(title="Median price by body type"), width="stretch")

    st.markdown("""
**Key findings**
- **Age is the biggest price driver.** Toyota, Honda and Hyundai lose about **12% of their value per year**; Mercedes-Benz about **19%**.
- **Foreign-used cars cost about 25% more** than locally used cars of the same model and year.
- **Toyota, Mercedes-Benz and Lexus** make up about **80%** of listings. SUVs cost about **twice** as much as sedans.
- **Mileage and location matter far less than age**, which suggests listed odometer readings are often unreliable.
""")

# ================================================================ TAB 3: MODEL
with tab_model:
    m = meta["metrics"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Model", "XGBoost")
    k2.metric("Average error (MAPE)", f"{m['mape']:.1%}")
    k3.metric("Median error", f"{m['median_ape']:.1%}")
    k4.metric("Within ±20% of actual", f"{m['within_20pct']:.0%}")
    st.caption(f"Evaluated on a held-out 30% test set ({meta['tested_on_rows']:,} listings the model never saw). "
               f"R² on log-price: {m['r2_log']:.3f}.")

    if COMPARISON_PATH.exists():
        comp = pd.read_csv(COMPARISON_PATH).sort_values("MAPE")
        fig = go.Figure(go.Bar(x=comp["MAPE"] * 100, y=comp["Model"], orientation="h",
                               marker_color=[ORANGE if i == 0 else BLUE for i in range(len(comp))],
                               text=[f"{v:.1f}%" for v in comp["MAPE"] * 100], textposition="outside",
                               hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Average % error on the test set (lower is better)")
        st.plotly_chart(style_fig(fig, 430).update_layout(title="11 algorithms compared"), width="stretch")
        show = comp[["Model", "R2 (log)", "MAPE", "Median APE", "MAE", "CV R2 (log) mean"]].copy()
        st.dataframe(show.style.format({"R2 (log)": "{:.3f}", "MAPE": "{:.1%}", "Median APE": "{:.1%}",
                                        "MAE": lambda v: naira(v), "CV R2 (log) mean": "{:.3f}"}),
                     hide_index=True, width="stretch")
    st.markdown("""
**How it works.** The model learns `log(price)` from the car's age, make, model, condition, body type, engine size,
mileage and location. Gradient-boosted trees (XGBoost, CatBoost) beat linear models, random forests, KNN and single
decision trees. **Car age is by far the most important feature**, followed by brand class, body type and condition.
See the notebook in `notebooks/` for the full analysis.
""")
