from matplotlib import pyplot as plt
from cnn import StartupCNN
from sklearn.preprocessing import StandardScaler
import os
import tensorflow as tf
import numpy as np
import random
import math
from data_utils import (
   load_and_clean_csv,
   encode_categorical,
   extract_numeric,
   make_dataset,
   make_splits
)

CSV_PATH = "data/cleaned/combined_startups_30cols.csv"


# -------------------------------
# PLOTTING FUNCTIONS
# -------------------------------
def visualize_loss(history):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('Visuals/loss.png')
    plt.show()


def visualize_accuracy(history):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('Visuals/accuracy.png')
    plt.show()


# --- NEW: comparison plots ---

def plot_accuracy_comparison(filename="Visuals/accuracy_comparison_transformer_vs_cnn.png"):
    """
    Plot train/val accuracy curves for Transformer vs CNN and save as PNG.
    Uses blue/green color palette for clarity.
    """
    try:
        t_data = np.load("transformer_history_and_metrics.npz", allow_pickle=True)
        c_data = np.load("cnn_history_and_metrics.npz", allow_pickle=True)
    except FileNotFoundError as e:
        print("Could not load history files for comparison:", e)
        print("Make sure you have run train_transformer.py first.")
        return

    t_train = t_data["train_accuracy"]
    t_val = t_data["val_accuracy"]
    c_train = c_data["train_accuracy"]
    c_val = c_data["val_accuracy"]

    epochs_t = np.arange(1, len(t_train) + 1)
    epochs_c = np.arange(1, len(c_train) + 1)

    # --- Blue-Green Palette ---
    colors = {
        "t_train": "#1f77b4",   # medium blue
        "t_val": "#6baed6",     # light blue
        "c_train": "#2ca25f",   # green
        "c_val": "#7bccc4",     # aqua/teal
    }

    plt.figure(figsize=(9, 5))

    # Transformer curves
    plt.plot(epochs_t, t_train, label="Transformer Train", color=colors["t_train"], linewidth=2)
    plt.plot(epochs_t, t_val, label="Transformer Val", color=colors["t_val"], linestyle="--", linewidth=2)

    # CNN curves
    plt.plot(epochs_c, c_train, label="CNN Train", color=colors["c_train"], linewidth=2)
    plt.plot(epochs_c, c_val, label="CNN Val", color=colors["c_val"], linestyle="--", linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Transformer vs CNN: Accuracy per Epoch", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved accuracy comparison plot to {filename}")


def plot_test_accuracy_bar(filename="Visuals/test_accuracy_bar_transformer_vs_cnn.png"):
    """
    Bar chart comparing final test accuracy of Transformer vs CNN.
    Requires:
      - transformer_history_and_metrics.npz
      - cnn_history_and_metrics.npz
    """
    try:
        t_data = np.load("transformer_history_and_metrics.npz", allow_pickle=True)
        c_data = np.load("cnn_history_and_metrics.npz", allow_pickle=True)
    except FileNotFoundError as e:
        print("Could not load history files for comparison:", e)
        print("Make sure you have run train_transformer.py first.")
        return

    t_test = float(t_data["test_accuracy"][0])
    c_test = float(c_data["test_accuracy"][0])

    models = ["Transformer", "CNN"]
    accuracies = [t_test, c_test]

    plt.figure(figsize=(5, 4))
    bars = plt.bar(models, accuracies)

    # Add text labels above bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{acc:.2f}",
            ha="center",
            va="bottom"
        )

    plt.ylim(0.0, 1.0)
    plt.ylabel("Test Accuracy")
    plt.title("Final Test Accuracy: Transformer vs CNN")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved test accuracy bar chart to {filename}")


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main() -> None:
    print("Preprocessing data:")
    df, status_to_id = load_and_clean_csv(CSV_PATH)

    print("Formatting data:")
    categorical_array, vocab_sizes, lookups = encode_categorical(df)
    num_array = extract_numeric(df)
    labels = df['label_id'].values

    (train_cat, train_num, y_train), (val_cat, val_num, y_val), (test_cat, test_num, y_test) = make_splits(
         categorical_array, num_array, labels
    )

    # --- Dataset size printouts ---
    print("Train rows:", len(train_cat))
    print("Val rows:", len(val_cat))
    print("Test rows:", len(test_cat))
    print("Total rows after split:", len(train_cat) + len(val_cat) + len(test_cat))
    print("Original rows:", len(df))

    train_ds = make_dataset(train_cat, train_num, y_train, batch_size=64, shuffle=True)
    val_ds = make_dataset(val_cat, val_num, y_val, batch_size=64, shuffle=False)
    test_ds = make_dataset(test_cat, test_num, y_test, batch_size=64, shuffle=False)

    cnn = StartupCNN(
        vocab_sizes=vocab_sizes,
        num_numeric_features=num_array.shape[1],
        embedding_dim=8,
        num_classes=len(status_to_id),
    )

    cnn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )

    history = cnn.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
    )

    # ---- Existing plots ----
    visualize_loss(history)
    visualize_accuracy(history)

    test_loss, test_acc = cnn.evaluate(test_ds)
    predictions = cnn.predict(test_ds)

    print(predictions)
    print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

    # save CNN history + test accuracy for comparison ---
    np.savez(
        "cnn_history_and_metrics.npz",
        train_accuracy=np.array(history.history.get("accuracy", [])),
        val_accuracy=np.array(history.history.get("val_accuracy", [])),
        test_accuracy=np.array([test_acc]),
    )

    # generate comparison plots (requires transformer npz) ---
    plot_accuracy_comparison()
    plot_test_accuracy_bar()


if __name__ == '__main__':
    main()
