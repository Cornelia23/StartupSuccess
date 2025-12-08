# train_transformer_new.py
import tensorflow as tf
from data_utils import (
    load_and_clean_csv,
    encode_categorical,
    extract_numeric,
    make_splits,
    make_dataset,
)
from startup_transformer import StartupTransformer

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np                                                   
import matplotlib.pyplot as plt   

CSV_PATH = "./data/cleaned/combined_startups_30cols.csv"   

def plot_confusion_matrix(cm, class_names, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues, filename="confusion_matrix.png"):
    """
    Save a nicely formatted confusion matrix as a PNG.

    cm: 2D numpy array (confusion matrix)
    class_names: list of class label strings in the same order as cm axes
    normalize: if True, shows percentages per row
    filename: output PNG file name
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Show all ticks and label them with the class names
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)

    # Print numbers inside the squares
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


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

    # 5. Compile (keep LR that was working well)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # 6. Train for fixed epochs (no early stopping)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
    )

    # 7. Evaluate on test set
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # --- NEW: save transformer history + test accuracy for comparison ---
    np.savez(
        "transformer_history_and_metrics.npz",
        train_accuracy=np.array(history.history.get("accuracy", [])),
        val_accuracy=np.array(history.history.get("val_accuracy", [])),
        test_accuracy=np.array([test_acc]),
    )

    # 8. Collect predictions for confusion matrix
    all_true = []
    all_pred = []

    for batch_x, batch_y in test_ds:
        logits = model(batch_x, training=False)
        preds = tf.argmax(logits, axis=-1)

        all_true.extend(batch_y.numpy())
        all_pred.extend(preds.numpy())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # Map ids back to human-readable statuses
    id_to_status = {v: k for k, v in status_to_id.items()}
    target_names = [id_to_status[i] for i in range(len(id_to_status))]

    # 9. Text confusion matrix + report in terminal
    cm = confusion_matrix(all_true, all_pred)
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(cm)

    print("\nClassification report:")
    print(classification_report(all_true, all_pred, target_names=target_names))

    # 10. Nice PNG confusion matrix
    plot_confusion_matrix(
        cm,
        class_names=target_names,
        normalize=False,  # set True if you want row-normalized percentages
        title="Startup status confusion matrix",
        filename="Visuals/confusion_matrix_status.png",
    )

    # 11. Save model (optional)
    model.save("startup_transformer_model")


if __name__ == "__main__":
    main()
