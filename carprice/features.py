"""Data cleaning and feature engineering.

Mirrors the steps in notebooks/Nigerian_Car_Price_Prediction_Autochek.ipynb so the
training script and the Streamlit app transform data exactly the same way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REFERENCE_YEAR = 2026  # data scraped October 2026; car_age = REFERENCE_YEAR - year

LUXURY_MAKES = {
    "Mercedes-Benz", "Lexus", "Land Rover", "BMW", "Porsche", "Rolls-Royce", "Lamborghini",
    "Cadillac", "Infiniti", "Acura", "Audi", "Lincoln", "Tesla", "Jaguar",
}

NUM = ["car_age", "mileage_km", "mileage_per_year", "cylinders", "grade_score",
       "has_financing", "has_warranty", "is_luxury", "listing_month"]
CAT = ["make", "make_model", "condition", "body_type", "transmission", "fuel_type", "state", "inspected"]

BODY_MAP = {8: "Sedan", 3: "SUV", 2: "Minivan", 1: "Pickup", 5: "Hatchback/Wagon", 7: "Coupe/Sports", 9: "Van/Bus"}
STATE_MAP = {"Lagos State": "Lagos", "Federal Capital Territory": "Abuja", "Port Harcourt": "Rivers",
             "Ogun State": "Ogun", "Rivers State": "Rivers"}
MAIN_STATES = ["Lagos", "Abuja", "Rivers", "Unknown"]


def _clean_make(m: str) -> str:
    m = str(m).strip()
    special = {"TOYOTA LANDCRUISER": "Toyota", "RANGE ROVER": "Land Rover", "MINI": "Mini"}
    keep_upper = {"BMW", "GMC", "BYD", "JAC", "GAC", "JMC", "MG", "RAM"}
    if m in special:
        return special[m]
    if m in keep_upper:
        return m
    return m.title() if (m.isupper() or m.islower()) else m


def model_key(model: pd.Series) -> pd.Series:
    """Normalise model spellings so 'RAV 4' == 'RAV4' and 'F 150' == 'F-150'."""
    return model.fillna("Unknown").str.strip().str.upper().str.replace(r"[\s\-]+", "", regex=True)


def clean_raw(raw: pd.DataFrame, return_display_names: bool = False):
    """Clean the scraped listings (see notebook section 3 for the reasoning behind each step)."""
    df = raw.copy()
    df = df.drop_duplicates(subset=["title", "year", "price_ngn", "mileage", "condition", "state"]).copy()

    df["model"] = df["model"].fillna("Unknown").str.strip()
    df.loc[df["make"].eq("TOYOTA LANDCRUISER"), "model"] = "Land Cruiser"
    df.loc[df["make"].eq("RANGE ROVER"), "model"] = "Range Rover " + df["model"]
    df["make"] = df["make"].map(_clean_make)
    display_model = df["model"].copy()
    df["model"] = model_key(df["model"])

    df["state"] = df["state"].str.strip().replace(STATE_MAP).fillna("Unknown")
    df.loc[~df["state"].isin(MAIN_STATES), "state"] = "Other"
    df["body_type"] = df["body_type_id"].map(BODY_MAP).fillna("Other")
    df["cylinders"] = df["engine_type"].str.extract(r"(\d+)", expand=False).astype(float)

    df["mileage_km"] = np.where(df["mileage_unit"].eq("miles"), df["mileage"] * 1.60934, df["mileage"])
    bad = (df["mileage_km"] > 1_000_000) | ((df["mileage_km"] == 0) & (df["condition"] != "new"))
    df.loc[bad, "mileage_km"] = np.nan

    df["transmission"] = df["transmission"].where(df["transmission"].isin(["automatic", "cvt", "manual"]), "other")
    df["fuel_type"] = df["fuel_type"].where(df["fuel_type"].eq("petrol"), "non-petrol")
    df["inspected"] = df["inspected"].map({True: "yes", False: "no", "True": "yes", "False": "no"}).fillna("unknown")
    df["has_financing"] = df["has_financing"].astype(int)
    df["has_warranty"] = df["has_warranty"].astype(int)
    df["listed_date"] = pd.to_datetime(df["listed_date"], utc=True)
    df["car_age"] = REFERENCE_YEAR - df["year"]
    df["make_model"] = df["make"] + " " + df["model"]

    # Remove price typos: < ₦1M, > ₦2B, or > 6x away from the median of the same make+model
    grp_median = df.groupby("make_model")["price_ngn"].transform("median")
    grp_size = df.groupby("make_model")["price_ngn"].transform("size")
    ratio = df["price_ngn"] / grp_median
    bad_price = (df["price_ngn"] < 1_000_000) | (df["price_ngn"] > 2_000_000_000) | \
                ((grp_size >= 3) & ((ratio > 6) | (ratio < 1 / 6)))
    keep = ~bad_price
    df = df[keep]

    keep_cols = ["year", "make", "model", "price_ngn", "has_financing", "condition", "transmission", "fuel_type",
                 "state", "city", "grade_score", "inspected", "has_warranty", "listed_date", "body_type",
                 "cylinders", "mileage_km", "car_age", "make_model"]
    df = df[keep_cols].reset_index(drop=True)
    if not return_display_names:
        return df
    disp = (pd.DataFrame({"make_model": (raw.loc[display_model.index, "make"].map(_clean_make) + " " +
                                          model_key(display_model)).values,
                          "name": display_model.values})
            .groupby("make_model")["name"].agg(lambda s: s.value_counts().index[0]))
    return df, disp.to_dict()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Model features derived from cleaned columns."""
    df = df.copy()
    df["mileage_per_year"] = df["mileage_km"] / df["car_age"].clip(lower=1)
    df["is_luxury"] = df["make"].isin(LUXURY_MAKES).astype(int)
    if "listing_month" not in df:
        df["listing_month"] = pd.to_datetime(df["listed_date"], utc=True).dt.month
    return df


def build_input_row(*, make: str, make_model: str, year: int, condition: str, mileage_km: float | None,
                    body_type: str, cylinders: float, transmission: str = "automatic", fuel_type: str = "petrol",
                    state: str = "Lagos", inspected: str = "unknown", grade_score: float = 3.0,
                    has_financing: int = 1, has_warranty: int = 0, listing_month: int = 10) -> pd.DataFrame:
    """One-row DataFrame in the exact format the trained pipeline expects."""
    row = pd.DataFrame([{
        "make": make, "make_model": make_model, "car_age": REFERENCE_YEAR - int(year), "condition": condition,
        "mileage_km": np.nan if mileage_km is None else float(mileage_km), "body_type": body_type,
        "cylinders": float(cylinders), "transmission": transmission, "fuel_type": fuel_type, "state": state,
        "inspected": inspected, "grade_score": float(grade_score), "has_financing": int(has_financing),
        "has_warranty": int(has_warranty), "listing_month": int(listing_month),
    }])
    row = add_features(row)
    return row[NUM + CAT]


def nice_model_name(make: str, make_model: str, display_names: dict) -> str:
    """Human-readable model name, e.g. 'Toyota RAV4' -> 'RAV 4' as most often written on the site."""
    return display_names.get(make_model, make_model[len(make) + 1:])
