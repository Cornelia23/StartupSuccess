# train_transformer.py
import tensorflow as tf
from data_utils import (
    load_and_clean_csv,
    encode_categorical,
    extract_numeric,
    make_splits,
    make_dataset,
)
from startup_transformer import StartupTransformer

CSV_PATH = "./data/cleaned_startup_data.csv"   # TODO: update this


def main():
    # 1. Load and clean CSV
    df, status_to_id = load_and_clean_csv(CSV_PATH)
    print("Statuses:", status_to_id)

    # 2. Encode features
    cat_array, vocab_sizes, lookups = encode_categorical(df)
    num_array = extract_numeric(df)
    labels = df["label_id"].values

    num_cat_features = cat_array.shape[1]
    num_numeric_features = num_array.shape[1]
    num_classes = len(status_to_id)

    print("num_cat_features:", num_cat_features)
    print("num_numeric_features:", num_numeric_features)
    print("num_classes:", num_classes)

    # 3. Train/val/test split
    (train_cat, train_num, y_train), (val_cat, val_num, y_val), (test_cat, test_num, y_test) = make_splits(
        cat_array, num_array, labels
    )

    train_ds = make_dataset(train_cat, train_num, y_train, batch_size=64, shuffle=True)
    val_ds = make_dataset(val_cat, val_num, y_val, batch_size=64, shuffle=False)
    test_ds = make_dataset(test_cat, test_num, y_test, batch_size=64, shuffle=False)

    # 4. Build model
    model = StartupTransformer(
        num_cat_features=num_cat_features,
        num_numeric_features=num_numeric_features,
        vocab_sizes=vocab_sizes,
        num_classes=num_classes,
        d_model=64,        
        num_heads=4,
        num_layers=2,
        d_ff=128,
        dropout_rate=0.1,
    )

    # 5. Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # 6. Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
    )

    # 7. Evaluate on test set
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # Optional: save model & label mapping
    model.save("startup_transformer_model")
    import json


if __name__ == "__main__":
    main()
