# data_utils.py
import pandas as pd
import numpy as np
import tensorflow as tf


# ------------------------------------------------------------
# Feature configuration for the 30 canonical columns
# ------------------------------------------------------------


# Categorical features for the transformer branch
CATEGORICAL_FEATURES = [
   "country",
   "state_code",
   "city",
   "region",
   "primary_industry",
]


# Numeric features for the MLP branch
NUMERIC_FEATURES = [
   "age_years",
   "is_software",
   "is_web",
   "is_mobile",
   "is_enterprise",
   "is_biotech_or_health",
   "funding_total_usd",
   "funding_rounds",
   "age_at_first_funding_years",
   "age_at_last_funding_years",
   "has_VC",
   "has_angel",
   "avg_participants",
   "num_founders",
   "employee_count",
   "latitude",
   "longitude",
   # Date columns as numeric features (after conversion to ordinal)
   "founded_at",
   "closed_at",
   "first_funding_at",
   "last_funding_at",
]


# Heavy-tailed numeric features to log-transform
HEAVY_TAILED = [
   "funding_total_usd",
   "employee_count",
   "num_founders",
   "avg_participants",
]


# Date columns that we'll convert to numeric (ordinal days)
DATE_COLUMNS = [
   "founded_at",
   "closed_at",
   "first_funding_at",
   "last_funding_at",
]


# We train on `status` (multi-class: acquired / closed / operating / ...)
LABEL_COLUMN = "status"




def load_and_clean_csv(path: str):
   """
   Load the combined 30-column CSV and:
     - normalize and encode `status` as labels,
     - convert date columns to numeric,
     - clean numeric features (impute, clip, log-transform, standardize),
     - clean categorical features (strings with 'UNKNOWN' for missing).


   Returns:
     df: cleaned DataFrame with a 'label_id' column
     status_to_id: dict mapping status string -> integer id
   """
   df = pd.read_csv(path)


   # --- Label: status ---
   df = df.dropna(subset=[LABEL_COLUMN])
   df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.lower()


   unique_statuses = sorted(df[LABEL_COLUMN].unique())
   status_to_id = {s: i for i, s in enumerate(unique_statuses)}
   df["label_id"] = df[LABEL_COLUMN].map(status_to_id)


   # --- Convert date columns to numeric (ordinal days) ---
   for col in DATE_COLUMNS:
       if col in df.columns:
           dt = pd.to_datetime(df[col], errors="coerce")
           df[col] = dt.map(lambda x: x.toordinal() if pd.notnull(x) else np.nan)


   # --- Numeric: convert to float + median impute ---
   for col in NUMERIC_FEATURES:
       if col not in df.columns:
           continue
       df[col] = pd.to_numeric(df[col], errors="coerce")
       median = df[col].median()
       if pd.isna(median):
           median = 0.0
       df[col] = df[col].fillna(median)


   # --- Clip weird ages (no negative years) ---
   for col in ["age_years", "age_at_first_funding_years", "age_at_last_funding_years"]:
       if col in df.columns:
           df[col] = df[col].clip(lower=0.0)


   # --- Log-transform heavy-tailed features ---
   for col in HEAVY_TAILED:
       if col in df.columns:
           df[col] = df[col].clip(lower=0.0)
           df[col] = np.log1p(df[col])


   # --- Standardize all numeric features (mean 0, std 1) ---
   for col in NUMERIC_FEATURES:
       if col not in df.columns:
           continue
       mean = df[col].mean()
       std = df[col].std()
       if std == 0 or np.isnan(std):
           std = 1.0
       df[col] = (df[col] - mean) / std


   # --- Categorical: strings with 'UNKNOWN' for missing ---
   for col in CATEGORICAL_FEATURES:
       if col not in df.columns:
           continue
       df[col] = df[col].fillna("UNKNOWN").astype(str)


   return df, status_to_id




def encode_categorical(df: pd.DataFrame):
   """
   Encode each categorical column into integer IDs with its own vocabulary.


   Returns:
       cat_array: (N, num_cat_features) int32
       vocab_sizes: list[int]
       lookups: {col_name: {category_string: int_id}}
   """
   cat_ids = []
   vocab_sizes = []
   lookups = {}


   for col in CATEGORICAL_FEATURES:
       if col not in df.columns:
           # stub if column missing (shouldn't happen with canonical 30, but safe)
           cat_ids.append(np.zeros((len(df),), dtype="int32"))
           vocab_sizes.append(1)
           lookups[col] = {"UNKNOWN": 0}
           continue


       values = df[col].astype(str).values
       unique_vals = sorted(pd.unique(values))


       # Reserve 0 for "UNKNOWN"/OOV; start real tokens at 1
       vocab = {v: i + 1 for i, v in enumerate(unique_vals)}
       vocab["UNKNOWN"] = 0


       encoded = np.array([vocab.get(v, 0) for v in values], dtype="int32")


       cat_ids.append(encoded)
       vocab_sizes.append(len(vocab))
       lookups[col] = vocab


   cat_array = np.stack(cat_ids, axis=1)  # (N, num_cat_features)
   return cat_array, vocab_sizes, lookups




def extract_numeric(df: pd.DataFrame):
   """
   Collect numeric features into a single array of shape (N, num_numeric_features).
   """
   num_data = []
   for col in NUMERIC_FEATURES:
       if col not in df.columns:
           num_data.append(np.zeros((len(df),), dtype="float32"))
       else:
           num_data.append(df[col].astype("float32").values)
   num_array = np.stack(num_data, axis=1)
   return num_array




def make_splits(cat_array, num_array, labels, train_frac=0.7, val_frac=0.15, seed=42):
   """
   Random train/val/test split with consistent shuffling across all arrays.
   """
   N = cat_array.shape[0]
   rng = np.random.default_rng(seed)
   indices = np.arange(N)
   rng.shuffle(indices)


   train_end = int(train_frac * N)
   val_end = int((train_frac + val_frac) * N)


   train_idx = indices[:train_end]
   val_idx = indices[train_end:val_end]
   test_idx = indices[val_end:]


   def split(arr):
       return arr[train_idx], arr[val_idx], arr[test_idx]


   cat_train, cat_val, cat_test = split(cat_array)
   num_train, num_val, num_test = split(num_array)
   y_train, y_val, y_test = split(labels)


   return (cat_train, num_train, y_train), (cat_val, num_val, y_val), (cat_test, num_test, y_test)




def make_dataset(cat_array, num_array, labels, batch_size=64, shuffle=True):
   """
   Wrap arrays into a tf.data.Dataset that matches the transformer input signature.
   """
   x = {
       "categorical_inputs": cat_array.astype("int32"),
       "numeric_inputs": num_array.astype("float32"),
   }
   y = labels.astype("int32")


   ds = tf.data.Dataset.from_tensor_slices((x, y))
   if shuffle:
       ds = ds.shuffle(buffer_size=len(y), seed=42)
   ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
   return ds



