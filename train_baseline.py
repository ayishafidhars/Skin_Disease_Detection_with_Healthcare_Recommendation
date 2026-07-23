"""
Baseline training pipeline for skin disease detection.

This script rebuilds a stratified split from the full dataset, trains a
MobileNetV2 baseline with matching preprocessing, and writes evaluation
artifacts that are easier to compare across runs.
"""

import json
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10
IMG_SIZE = 224
HEAD_BATCH_SIZE = 32
FINE_TUNE_BATCH_SIZE = 24
HEAD_EPOCHS = 20              # Fix 1: was 10, more time to learn the head
FINE_TUNE_EPOCHS = 50         # Fix 1: was 20, full convergence needs more epochs
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-4  # Fix 2: was 1e-5 (100x drop), now 10x drop only
LABEL_SMOOTHING = 0.05
UNFREEZE_LAYERS = 70          # Fix 5: was 40, unfreeze more layers for better features

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(SCRIPT_DIR, "SkinDisease")
OUTPUT_MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_LOGS_DIR = os.path.join(BASE_DIR, "logs")

MODEL_BEST_PATH = os.path.join(OUTPUT_MODELS_DIR, "skin_disease_model_baseline_best.keras")
MODEL_FINAL_PATH = os.path.join(OUTPUT_MODELS_DIR, "skin_disease_model_baseline_final.keras")


def configure_runtime():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("Using CPU")
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print(f"Could not enable memory growth for {gpu}: {exc}")
    print(f"Using GPU: {len(gpus)} device(s)")


def ensure_output_dirs():
    os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LOGS_DIR, exist_ok=True)


def collect_image_records(data_root):
    records = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for split_name in ("train", "val", "test"):
        split_dir = os.path.join(data_root, split_name)
        if not os.path.isdir(split_dir):
            continue

        for label in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, label)
            if not os.path.isdir(class_dir):
                continue

            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                if os.path.isfile(image_path) and os.path.splitext(image_name)[1].lower() in valid_exts:
                    records.append({"filepath": image_path, "label": label})

    if not records:
        raise FileNotFoundError(f"No images found under {data_root}")

    dataframe = pd.DataFrame(records)
    dataframe = dataframe.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return dataframe


def build_stratified_splits(dataframe):
    train_val_df, test_df = train_test_split(
        dataframe,
        test_size=TEST_RATIO,
        stratify=dataframe["label"],
        random_state=SEED,
    )

    val_share_of_remaining = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_share_of_remaining,
        stratify=train_val_df["label"],
        random_state=SEED,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_split_manifests(train_df, val_df, test_df, timestamp):
    split_dir = os.path.join(OUTPUT_RESULTS_DIR, f"baseline_split_{timestamp}")
    os.makedirs(split_dir, exist_ok=True)
    train_df.to_csv(os.path.join(split_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(split_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(split_dir, "test.csv"), index=False)
    return split_dir


def create_generators(train_df, val_df, test_df):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=12,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.10,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=HEAD_BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )
    val_gen = eval_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=HEAD_BATCH_SIZE,
        shuffle=False,
    )
    test_gen = eval_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=HEAD_BATCH_SIZE,
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


def create_fine_tune_generator(train_df):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=10,
        width_shift_range=0.06,
        height_shift_range=0.06,
        zoom_range=0.08,
        horizontal_flip=True,
        brightness_range=[0.92, 1.08],
        fill_mode="nearest",
    )
    return train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=FINE_TUNE_BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )


def compute_training_class_weights(train_df, class_indices):
    ordered_labels = sorted(class_indices, key=class_indices.get)
    encoded = np.array([class_indices[label] for label in train_df["label"]])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(ordered_labels)),
        y=encoded,
    )
    return {index: float(weight) for index, weight in enumerate(weights)}


def build_model(num_classes):
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False

    features = base_model.output
    features = GlobalAveragePooling2D()(features)
    features = Dropout(0.30)(features)
    features = Dense(512, activation="relu")(features)  # Fix 4: added wider first layer
    features = Dropout(0.30)(features)
    features = Dense(256, activation="relu")(features)
    features = Dropout(0.20)(features)
    outputs = Dense(num_classes, activation="softmax")(features)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )


def create_callbacks(timestamp):
    return [
        ModelCheckpoint(
            filepath=MODEL_BEST_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=10,  # Fix 3: was 6, give more time before stopping
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        TensorBoard(
            log_dir=os.path.join(OUTPUT_LOGS_DIR, f"baseline_{timestamp}"),
            histogram_freq=0,
        ),
    ]


def unfreeze_top_layers(base_model):
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False
    for layer in base_model.layers[-UNFREEZE_LAYERS:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def merge_histories(first_history, second_history):
    merged = {}
    for history in (first_history.history, second_history.history):
        for key, values in history.items():
            merged.setdefault(key, []).extend(values)
    return merged


def save_class_names(class_indices):
    ordered = sorted(class_indices, key=class_indices.get)
    class_names_path = os.path.join(OUTPUT_MODELS_DIR, "class_names.txt")
    with open(class_names_path, "w", encoding="utf-8") as handle:
        for class_name in ordered:
            handle.write(f"{class_name}\n")
    return ordered


def save_model_metadata(model_path, class_names):
    metadata = {
        "architecture": "MobileNetV2",
        "image_size": IMG_SIZE,
        "preprocessing": "mobilenet_v2",
        "class_names": class_names,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
    }
    metadata_path = os.path.splitext(model_path)[0] + ".meta.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def plot_training_curves(history, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["accuracy"], label="train")
    axes[0].plot(history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["loss"], label="train")
    axes[1].plot(history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def evaluate_model(model, test_gen, class_names):
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    predicted_indices = np.argmax(predictions, axis=1)
    true_indices = test_gen.classes
    report = classification_report(
        true_indices,
        predicted_indices,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    macro_f1 = f1_score(true_indices, predicted_indices, average="macro")
    balanced_acc = balanced_accuracy_score(true_indices, predicted_indices)
    return report, macro_f1, balanced_acc


def write_report(report_path, training_summary):
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(training_summary)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("Skin Disease Baseline Training")
    print("=" * 70)

    configure_runtime()
    ensure_output_dirs()

    dataset_df = collect_image_records(DATA_ROOT)
    train_df, val_df, test_df = build_stratified_splits(dataset_df)
    manifest_dir = save_split_manifests(train_df, val_df, test_df, timestamp)

    print(f"Total images: {len(dataset_df)}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Split manifests: {manifest_dir}")

    train_gen, val_gen, test_gen = create_generators(train_df, val_df, test_df)
    num_classes = len(train_gen.class_indices)
    model, base_model = build_model(num_classes=num_classes)
    compile_model(model, HEAD_LEARNING_RATE)

    class_names = save_class_names(train_gen.class_indices)
    class_weight_dict = compute_training_class_weights(train_df, train_gen.class_indices)
    callbacks = create_callbacks(timestamp)

    print("\nTraining head...")
    head_history = model.fit(
        train_gen,
        epochs=HEAD_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )

    print("\nFine-tuning top layers...")
    fine_tune_train_gen = create_fine_tune_generator(train_df)
    unfreeze_top_layers(base_model)
    compile_model(model, FINE_TUNE_LEARNING_RATE)
    fine_tune_history = model.fit(
        fine_tune_train_gen,
        initial_epoch=len(head_history.history["loss"]),
        epochs=len(head_history.history["loss"]) + FINE_TUNE_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )

    full_history = merge_histories(head_history, fine_tune_history)

    print("\nEvaluating best checkpoint on test split...")
    best_model = tf.keras.models.load_model(MODEL_BEST_PATH)
    test_loss, test_accuracy = best_model.evaluate(test_gen, verbose=1)
    report, macro_f1, balanced_acc = evaluate_model(best_model, test_gen, class_names)

    best_model.save(MODEL_FINAL_PATH)
    save_model_metadata(MODEL_BEST_PATH, class_names)
    save_model_metadata(MODEL_FINAL_PATH, class_names)

    history_plot_path = os.path.join(OUTPUT_RESULTS_DIR, f"baseline_training_history_{timestamp}.png")
    plot_training_curves(full_history, history_plot_path)

    summary_lines = [
        "=" * 70,
        "BASELINE TRAINING REPORT - Skin Disease Detection",
        "=" * 70,
        "",
        f"Timestamp: {timestamp}",
        "Architecture: MobileNetV2",
        f"Image Size: {IMG_SIZE}x{IMG_SIZE}",
        f"Head Batch Size: {HEAD_BATCH_SIZE}",
        f"Fine-tune Batch Size: {FINE_TUNE_BATCH_SIZE}",
        f"Head Epochs: {HEAD_EPOCHS}",
        f"Fine-tune Epochs: {FINE_TUNE_EPOCHS}",
        f"Head Learning Rate: {HEAD_LEARNING_RATE}",
        f"Fine-tune Learning Rate: {FINE_TUNE_LEARNING_RATE}",
        f"Label Smoothing: {LABEL_SMOOTHING}",
        f"Train/Val/Test Ratio: {TRAIN_RATIO:.2f}/{VAL_RATIO:.2f}/{TEST_RATIO:.2f}",
        "Preprocessing: tensorflow.keras.applications.mobilenet_v2.preprocess_input",
        "Augmentation: light rotation/translation/zoom/horizontal flip only",
        "Imbalance Handling: class weights only",
        "",
        f"Train Samples: {len(train_df)}",
        f"Validation Samples: {len(val_df)}",
        f"Test Samples: {len(test_df)}",
        "",
        f"Best Test Accuracy: {test_accuracy * 100:.2f}%",
        f"Best Test Loss: {test_loss:.4f}",
        f"Macro F1: {macro_f1:.4f}",
        f"Balanced Accuracy: {balanced_acc:.4f}",
        f"Best Validation Accuracy: {max(full_history['val_accuracy']) * 100:.2f}%",
        "",
        "Classification Report",
        "-" * 70,
        report,
        "",
        "Artifacts",
        "-" * 70,
        f"Best Model: {MODEL_BEST_PATH}",
        f"Final Model: {MODEL_FINAL_PATH}",
        f"History Plot: {history_plot_path}",
        f"Split Manifests: {manifest_dir}",
    ]
    summary_text = "\n".join(summary_lines)

    report_path = os.path.join(OUTPUT_RESULTS_DIR, f"baseline_training_report_{timestamp}.txt")
    write_report(report_path, summary_text)

    print("\n" + summary_text)


if __name__ == "__main__":
    main()
