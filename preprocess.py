import math
import numpy as np
import pandas as pd

# ============================================================
# Canonical 30 columns we will output for BOTH datasets
# ============================================================

CANONICAL_COLS = [
    "company_id",
    "company_name",
    "success_label",
    "status",
    "country",
    "state_code",
    "city",
    "region",
    "latitude",
    "longitude",
    "founded_at",
    "closed_at",
    "age_years",
    "primary_industry",
    "is_software",
    "is_web",
    "is_mobile",
    "is_enterprise",
    "is_biotech_or_health",
    "funding_total_usd",
    "funding_rounds",
    "first_funding_at",
    "last_funding_at",
    "age_at_first_funding_years",
    "age_at_last_funding_years",
    "has_VC",
    "has_angel",
    "avg_participants",
    "num_founders",
    "employee_count",
]

# Simple US region mapping by state_code (for startup data.csv)
US_REGION_MAP = {
    # West
    "WA": "US-West", "OR": "US-West", "CA": "US-West", "AK": "US-West", "HI": "US-West",
    "AZ": "US-West", "CO": "US-West", "ID": "US-West", "MT": "US-West", "NV": "US-West",
    "NM": "US-West", "UT": "US-West", "WY": "US-West",
    # Midwest
    "IL": "US-Midwest", "IN": "US-Midwest", "MI": "US-Midwest", "OH": "US-Midwest", "WI": "US-Midwest",
    "IA": "US-Midwest", "KS": "US-Midwest", "MN": "US-Midwest", "MO": "US-Midwest",
    "NE": "US-Midwest", "ND": "US-Midwest", "SD": "US-Midwest",
    # Northeast
    "CT": "US-Northeast", "ME": "US-Northeast", "MA": "US-Northeast", "NH": "US-Northeast",
    "RI": "US-Northeast", "VT": "US-Northeast", "NJ": "US-Northeast", "NY": "US-Northeast",
    "PA": "US-Northeast",
    # South
    "DE": "US-South", "FL": "US-South", "GA": "US-South", "MD": "US-South", "NC": "US-South",
    "SC": "US-South", "VA": "US-South", "DC": "US-South", "WV": "US-South", "AL": "US-South",
    "KY": "US-South", "MS": "US-South", "TN": "US-South", "AR": "US-South", "LA": "US-South",
    "OK": "US-South", "TX": "US-South",
}


# ============================================================
# Helper functions
# ============================================================

def compute_age(found_dt, closed_dt=None, ref_dt=None):
    """
    Age in years: from founding to either closed_dt (if present) or ref_dt.
    """
    if pd.isna(found_dt):
        return np.nan
    if pd.notna(closed_dt):
        end = closed_dt
    elif ref_dt is not None and pd.notna(ref_dt):
        end = ref_dt
    else:
        return np.nan
    return (end - found_dt).days / 365.25


def map_region_from_state_or_continent(state_code, continent):
    """
    Map a US state to a US region; otherwise fall back on continent string.
    """
    if isinstance(state_code, str) and state_code in US_REGION_MAP:
        return US_REGION_MAP[state_code]

    if isinstance(continent, str):
        cont = continent.lower()
        if "europe" in cont:
            return "EU"
        if "asia" in cont:
            return "APAC"
        if "america" in cont:
            return "North America"
        if "africa" in cont:
            return "Africa"
        if "oceania" in cont:
            return "Oceania"

    return np.nan


def flag_from_text(series, keyword):
    return series.astype(str).str.contains(keyword, case=False, na=False).astype(int)


# ============================================================
# Preprocess: Crunchbase-like dataset ("startup data.csv")
# ============================================================

def preprocess_crunchbase(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    out = pd.DataFrame(index=df.index)

    # --- Identity ---
    out["company_id"] = df["id"].fillna(df.get("object_id"))
    out["company_name"] = df["name"]

    # --- Status & success label ---
    status = df["status"].astype(str).str.lower()
    out["status"] = status
    # You can tweak this rule, but this is a reasonable start
    out["success_label"] = np.where(status.isin(["acquired", "ipo", "operating"]), 1, 0)

    # --- Geography ---
    out["country"] = "USA"  # this dataset is US-only
    out["state_code"] = df["state_code"]
    out["city"] = df["city"]
    out["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    out["region"] = out["state_code"].map(US_REGION_MAP)

    # --- Dates / ages ---
    out["founded_at"] = pd.to_datetime(df["founded_at"], errors="coerce")
    out["closed_at"] = pd.to_datetime(df["closed_at"], errors="coerce")
    out["first_funding_at"] = pd.to_datetime(df["first_funding_at"], errors="coerce")
    out["last_funding_at"] = pd.to_datetime(df["last_funding_at"], errors="coerce")

    age_list = []
    for f, c, lf in zip(out["founded_at"], out["closed_at"], out["last_funding_at"]):
        age_list.append(compute_age(f, closed_dt=c, ref_dt=lf))
    out["age_years"] = age_list

    out["age_at_first_funding_years"] = pd.to_numeric(
        df["age_first_funding_year"], errors="coerce"
    )
    out["age_at_last_funding_years"] = pd.to_numeric(
        df["age_last_funding_year"], errors="coerce"
    )

    # --- Industry / product features ---
    out["primary_industry"] = df["category_code"]
    out["is_software"] = df["is_software"].fillna(0).astype(int)
    out["is_web"] = df["is_web"].fillna(0).astype(int)
    out["is_mobile"] = df["is_mobile"].fillna(0).astype(int)
    out["is_enterprise"] = df["is_enterprise"].fillna(0).astype(int)
    out["is_biotech_or_health"] = df["is_biotech"].fillna(0).astype(int)

    # --- Funding ---
    out["funding_total_usd"] = pd.to_numeric(df["funding_total_usd"], errors="coerce")
    out["funding_rounds"] = pd.to_numeric(df["funding_rounds"], errors="coerce")

    vc_counts = pd.to_numeric(df["has_VC"], errors="coerce")
    out["has_VC"] = vc_counts.fillna(0).gt(0).astype(int)

    angel_counts = pd.to_numeric(df["has_angel"], errors="coerce")
    out["has_angel"] = angel_counts.fillna(0).gt(0).astype(int)

    out["avg_participants"] = pd.to_numeric(df["avg_participants"], errors="coerce")

    # --- Team / network ---
    # Not exactly “num_founders” but relationships = founders/advisors/investors
    out["num_founders"] = pd.to_numeric(df["relationships"], errors="coerce")

    # No employee count in this dataset
    out["employee_count"] = np.nan

    # Return in canonical column order
    return out[CANONICAL_COLS]


# ============================================================
# Preprocess: CAX dataset ("CAX_Startup_Data.csv")
# ============================================================

def preprocess_cax(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    out = pd.DataFrame(index=df.index)

    # --- Identity ---
    out["company_id"] = df["Company_Name"]
    out["company_name"] = df["Company_Name"]

    # --- Status & success label ---
    status_raw = df["Dependent-Company Status"].astype(str).str.lower()
    # Map to something closer to Crunchbase
    out["status"] = status_raw.replace(
        {"success": "operating", "failed": "closed"}
    )
    out["success_label"] = np.where(status_raw == "success", 1, 0)

    # --- Geography ---
    out["country"] = df["Country of company"]
    out["state_code"] = np.nan
    out["city"] = np.nan
    out["latitude"] = np.nan
    out["longitude"] = np.nan

    continent = df.get("Continent of company")
    out["region"] = continent.apply(
        lambda x: map_region_from_state_or_continent(None, x)
    )

    # --- Dates / ages ---
    # Founding date: use Est. Founding Date, backfill with year of founding
    est_found_dt = pd.to_datetime(df["Est. Founding Date"], errors="coerce")
    out["founded_at"] = est_found_dt

    year_series = df["year of founding"].replace("No Info", np.nan)
    year_series = pd.to_numeric(year_series, errors="coerce")
    missing_mask = out["founded_at"].isna() & year_series.notna()

    # Approximate founding as July 1 of founding year when only year is known
    out.loc[missing_mask, "founded_at"] = pd.to_datetime(
        year_series[missing_mask].astype(int).astype(str) + "-07-01",
        errors="coerce",
    )

    # CAX doesn’t give explicit closed date
    out["closed_at"] = pd.NaT

    out["first_funding_at"] = pd.to_datetime(
        df.get("Date of 1st Investment"), errors="coerce"
    )
    out["last_funding_at"] = pd.to_datetime(
        df.get("Last Funding Date"), errors="coerce"
    )

    # Age in years: prefer directly given "Age of company in years"
    age_col = pd.to_numeric(
        df["Age of company in years"].replace("No Info", np.nan),
        errors="coerce",
    )

    age_list = []
    for idx, f in out["founded_at"].items():
        if not (idx >= len(age_col)) and not math.isnan(age_col.iloc[idx]):
            age_list.append(age_col.iloc[idx])
        else:
            lf = out["last_funding_at"].iloc[idx] if idx < len(out["last_funding_at"]) else pd.NaT
            age_list.append(compute_age(f, ref_dt=lf))
    out["age_years"] = age_list

    # Time to investment in months → years
    out["age_at_first_funding_years"] = pd.to_numeric(
        df.get("Time to 1st investment (in months)"), errors="coerce"
    ) / 12.0

    out["age_at_last_funding_years"] = pd.to_numeric(
        df.get(
            "Avg time to investment - average across all rounds, measured from previous investment"
        ),
        errors="coerce",
    ) / 12.0

    # --- Industry / product ---
    out["primary_industry"] = df["Industry of company"]

    out["is_software"] = flag_from_text(df["Industry of company"], "software")
    out["is_web"] = (
        flag_from_text(df["Industry of company"], "internet")
        | flag_from_text(df["Industry of company"], "web")
    )
    out["is_mobile"] = flag_from_text(df["Industry of company"], "mobile")

    focus = df.get("Focus functions of company").astype(str)
    out["is_enterprise"] = focus.str.contains("enterprise", case=False, na=False).astype(int)

    regulated_keywords = [
        "health", "medical", "pharma", "biotech",
        "finance", "bank", "insurance", "energy"
    ]
    reg_flag = pd.Series(False, index=df.index)
    industry_lower = df["Industry of company"].astype(str).str.lower()
    for kw in regulated_keywords:
        reg_flag = reg_flag | industry_lower.str.contains(kw, na=False)
    out["is_biotech_or_health"] = reg_flag.astype(int)

    # --- Funding ---
    out["funding_total_usd"] = pd.to_numeric(
        df.get("Last Funding Amount"), errors="coerce"
    )

    # Many CAX files don’t have explicit number-of-rounds column;
    # if yours does, this will use it; otherwise it will be NaN
    out["funding_rounds"] = pd.to_numeric(
        df.get("Number of funding rounds"), errors="coerce"
    )

    vc_n = pd.to_numeric(
        df.get("Number of Investors in Angel and or VC"), errors="coerce"
    )
    out["has_VC"] = vc_n.fillna(0).gt(0).astype(int)

    angel_n = pd.to_numeric(
        df.get("Number of Investors in Seed"), errors="coerce"
    )
    out["has_angel"] = angel_n.fillna(0).gt(0).astype(int)

    out["avg_participants"] = pd.to_numeric(
        df.get("Average number of  investors across rounds"), errors="coerce"
    )

    # --- Team / network / scale ---
    out["num_founders"] = pd.to_numeric(
        df.get("Number of Co-founders"), errors="coerce"
    )
    out["employee_count"] = pd.to_numeric(
        df.get("Employee Count"), errors="coerce"
    )

    return out[CANONICAL_COLS]


# ============================================================
# Main entry point: run both preprocessors
# ============================================================

if __name__ == "__main__":
    # Adjust paths if needed
    cax_path = "./data/CAX_Startup_Data.csv"
    cb_path = "./data/startup data.csv"

    cax_raw = pd.read_csv(cax_path)
    cb_raw = pd.read_csv(cb_path)

    cax_clean = preprocess_cax(cax_raw)
    cb_clean = preprocess_crunchbase(cb_raw)

    cax_clean.to_csv("cax_preprocessed_30cols.csv", index=False)
    cb_clean.to_csv("crunchbase_preprocessed_30cols.csv", index=False)

    print("Saved cax_preprocessed_30cols.csv and crunchbase_preprocessed_30cols.csv")
