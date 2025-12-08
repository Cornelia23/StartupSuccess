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
    plt.savefig('loss.png')
    plt.show()


def visualize_accuracy(history):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy.png')
    plt.show()


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

    # ---- NEW: Plot results ----
    visualize_loss(history)
    visualize_accuracy(history)

    test_loss, test_acc = cnn.evaluate(test_ds)
    predictions = cnn.predict(test_ds)

    print(predictions)
    print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")


if __name__ == '__main__':
    main()
