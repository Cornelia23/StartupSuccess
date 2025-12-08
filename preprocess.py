import pandas as pd
import numpy as np

df = pd.read_csv("startup data.csv")

# --- A. Core identity & lifecycle ---
df['founded_at'] = pd.to_datetime(df.get('founded_at'))
df['closed_at'] = pd.to_datetime(df.get('closed_at'))
df['acquired_at'] = pd.to_datetime(df.get('acquired_at'))
df['ipo_at'] = pd.to_datetime(df.get('ipo_at'))

today = pd.Timestamp("2025-01-01")

def compute_age(row):
    if pd.notnull(row['closed_at']):
        return (row['closed_at'] - row['founded_at']).days / 365
    if pd.notnull(row['acquired_at']):
        return (row['acquired_at'] - row['founded_at']).days / 365
    if pd.notnull(row['founded_at']):
        return (today - row['founded_at']).days / 365
    return np.nan

df['age'] = df.apply(compute_age, axis=1)

# --- B. Geography ---
region_map = {
    "CA": "US-West", "WA": "US-West", "OR": "US-West",
    "NY": "US-East", "MA": "US-East", "NJ": "US-East",
    "TX": "US-South", "FL": "US-South", "GA": "US-South"
}
df['region'] = df['state_code'].map(region_map).fillna("Other")
df['country'] = "USA"

# --- C. Industry (placeholder since dataset lacks details) ---
df['primary_industry'] = np.nan
df['is_software'] = 1
df['is_web'] = 1
df['is_mobile'] = 0
df['is_b2b'] = np.nan
df['regulated'] = 0

# --- D. Funding ---
round_cols = ['has_roundA','has_roundB','has_roundC','has_roundD']
df['funding_stage'] = df[round_cols].idxmax(axis=1).str.replace("has_","")
df['num_rounds'] = df[round_cols].sum(axis=1)
df['has_vc_backing'] = df['has_VC'].astype(int)

# funding totals missing: placeholder
df['funding_total_usd'] = np.nan
df['age_first_funding'] = np.nan
df['age_last_funding'] = np.nan
df['lead_investor_tier'] = df['is_top500'].map({1:"Top",0:"Other"})

# --- E. Team & Network ---
df['num_founders'] = df.get('founder_count')
df['repeat_founder'] = np.nan
df['accelerator'] = np.nan
df['avg_investor_participation'] = df['avg_participants']

# --- F. Traction ---
df['revenue_bucket'] = np.nan
df['growth_indicator'] = np.nan

# --- Final selection of features ---
cols = [
    "id","status","founded_at","closed_at","acquired_at","ipo_at","age",
    "city","country","region",
    "primary_industry","is_software","is_web","is_mobile","is_b2b","regulated",
    "funding_total_usd","funding_stage","num_rounds","age_first_funding","age_last_funding",
    "lead_investor_tier","has_vc_backing",
    "num_founders","repeat_founder","accelerator","avg_investor_participation",
    "revenue_bucket","growth_indicator"
]

clean = df[cols]

# Save to CSV
clean.to_csv('cleaned_startup_data.csv', index=False)

print(f"Cleaned data saved! Shape: {clean.shape}")
clean.head()