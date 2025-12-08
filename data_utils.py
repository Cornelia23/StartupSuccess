# data_utils.py
import pandas as pd
import numpy as np
import tensorflow as tf

CATEGORICAL_FEATURES = [
    "city",
    "country",
    "region",
    "primary_industry",
    "funding_stage",
    "lead_investor_tier",
    "has_vc_backing",
    "accelerator",
    "repeat_founder",
]

NUMERIC_FEATURES = [
    "age",
    "funding_total_usd",
    "num_rounds",
    "age_first_funding",
    "age_last_funding",
    "avg_investor_participation",
]

LABEL_COLUMN = "status"


def load_and_clean_csv(path: str):
    df = pd.read_csv(path)

    # Drop rows without a status
    df = df.dropna(subset=[LABEL_COLUMN])

    # Map status to integer labels
    unique_statuses = sorted(df[LABEL_COLUMN].unique())
    status_to_id = {s: i for i, s in enumerate(unique_statuses)}
    df["label_id"] = df[LABEL_COLUMN].map(status_to_id)

    # Clean/convert numeric features to float
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

        median = df[col].median()
        # If column is all NaN, median will be NaN -> fall back to 0.0
        if pd.isna(median):
            median = 0.0
        df[col] = df[col].fillna(median)


    # Ensure categorical features are strings and fill missing with "UNKNOWN"
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("UNKNOWN").astype(str)

    return df, status_to_id


def encode_categorical(df: pd.DataFrame):
    """
    Build vocabularies for each categorical column and encode them to integer IDs.
    Returns:
        cat_array: shape (N, num_cat_features), int32
        vocab_sizes: list[int]
        lookups: dict[col_name] -> {category_string: int_id}
    """
    cat_ids = []
    vocab_sizes = []
    lookups = {}

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            # stub column if missing
            cat_ids.append(np.zeros((len(df),), dtype="int32"))
            vocab_sizes.append(1)
            lookups[col] = {"UNKNOWN": 0}
            continue

        values = df[col].astype(str).values
        unique_vals = sorted(pd.unique(values))
        # Reserve 0 for "UNKNOWN/OOV"; start vocab at 1
        vocab = {v: i + 1 for i, v in enumerate(unique_vals)}
        vocab["UNKNOWN"] = 0

        encoded = np.array([vocab.get(v, 0) for v in values], dtype="int32")

        cat_ids.append(encoded)
        vocab_sizes.append(len(vocab))
        lookups[col] = vocab

    cat_array = np.stack(cat_ids, axis=1)  # (N, num_cat_features)
    return cat_array, vocab_sizes, lookups


def extract_numeric(df: pd.DataFrame):
    num_data = []
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            # Use zeros for missing numeric columns
            num_data.append(np.zeros((len(df),), dtype="float32"))
        else:
            num_data.append(df[col].astype("float32").values)
    num_array = np.stack(num_data, axis=1)  # (N, num_numeric_features)
    return num_array


def make_splits(cat_array, num_array, labels, train_frac=0.7, val_frac=0.15, seed=42):
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
