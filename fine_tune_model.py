import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, LearningRateScheduler
from tensorflow.keras import mixed_precision
from datetime import datetime
import math

print("=" * 70)
print("Skin Disease Detection - Fine-Tuning Model with GPU Optimization")
print("=" * 70)

# GPU Configuration
print("\n" + "=" * 70)
print("GPU Configuration")
print("=" * 70)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ Found {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
    
    try:
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU memory growth enabled")
        
        # Optional: Set GPU as the default device
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"✓ Using GPU: {gpus[0].name}")
        
    except RuntimeError as e:
        print(f"✗ GPU configuration error: {e}")
else:
    print("✗ No GPU found. Training will use CPU (slower).")
    print("  To use GPU, ensure you have:")
    print("  - CUDA-compatible GPU")
    print("  - CUDA Toolkit installed")
    print("  - tensorflow-gpu or TensorFlow 2.x with GPU support")

# Enable mixed precision for better GPU performance
print("\nEnabling mixed precision training for better GPU performance...")
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f"✓ Mixed precision policy: {policy.name}")
    print("  Compute dtype: float16")
    print("  Variable dtype: float32")
except Exception as e:
    print(f"✗ Could not enable mixed precision: {e}")
    print("  Continuing with default float32 precision")

# Display TensorFlow build info
print(f"\nTensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"GPU available: {tf.test.is_gpu_available()}")

# Configuration - Maximum Accuracy with GPU & Adam Optimizer
IMG_SIZE = 224  # High quality images (224x224) - NO reduction
BATCH_SIZE = 128 if gpus else 32  # Large batch for GPU speed
FINE_TUNE_EPOCHS = 15  # Sufficient for convergence with early stopping
NUM_CLASSES = 22

# Fine-tuning configuration - Deep unfreezing for maximum accuracy
FINE_TUNE_AT = 80  # Unfreeze last 80 layers (deep fine-tuning)
INITIAL_LEARNING_RATE = 2e-4  # Optimal for Adam optimizer
MIN_LEARNING_RATE = 1e-7

# TTA Configuration
TTA_AUGMENTATIONS = 5  # Multiple predictions for higher accuracy

# Get the script's directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Parent directory (workspace root)

# Data directories
TRAIN_DIR = os.path.join(SCRIPT_DIR, 'SkinDisease/train')
VAL_DIR = os.path.join(SCRIPT_DIR, 'SkinDisease/val')
TEST_DIR = os.path.join(SCRIPT_DIR, 'SkinDisease/test')

# Model paths - Baseline-first model priority
PRETRAINED_MODEL_PATH = os.path.join(BASE_DIR, 'models/skin_disease_model_baseline_best.keras')
FINETUNED_MODEL_PATH = os.path.join(BASE_DIR, 'models/skin_disease_model_finetuned.keras')

# Check for alternative models if main one doesn't exist
if not os.path.exists(PRETRAINED_MODEL_PATH):
    alternative_paths = [
        os.path.join(BASE_DIR, 'models/skin_disease_model_baseline_final.keras'),
        os.path.join(BASE_DIR, 'models/skin_disease_model_best.keras'),
        os.path.join(BASE_DIR, 'models/skin_disease_model_cpu_best.keras'),
        os.path.join(BASE_DIR, 'models/skin_disease_model_final.keras')
    ]
    for alt_path in alternative_paths:
        if os.path.exists(alt_path):
            PRETRAINED_MODEL_PATH = alt_path
            print(f"Using alternative model: {alt_path}")
            break


def load_model_metadata(model_path):
    metadata_path = os.path.splitext(model_path)[0] + '.meta.json'
    if not os.path.exists(metadata_path):
        return {}
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def get_preprocess_fn(metadata):
    if metadata.get('preprocessing') == 'mobilenet_v2':
        return mobilenet_v2_preprocess_input
    return None

# Create output directories
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)

print("\n" + "=" * 70)
print("Data Loading")
print("=" * 70)

model_metadata = load_model_metadata(PRETRAINED_MODEL_PATH)
if model_metadata.get('image_size'):
    IMG_SIZE = int(model_metadata['image_size'])
    print(f"Using metadata image size: {IMG_SIZE}")

preprocess_fn = get_preprocess_fn(model_metadata)

# Strong Data Augmentation for Maximum Accuracy
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_fn,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.85, 1.15],
    fill_mode='reflect'
)

# No augmentation for validation and test
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

print("\nLoading training data...")
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

print("Loading validation data...")
val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Loading test data...")
test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Get class names
class_names = list(train_data.class_indices.keys())
print(f"\nDataset Summary:")
print(f"  Training samples:   {train_data.samples}")
print(f"  Validation samples: {val_data.samples}")
print(f"  Test samples:       {test_data.samples}")
print(f"  Number of classes:  {len(class_names)}")

# Calculate class weights for balanced training
from sklearn.utils.class_weight import compute_class_weight
print("\nCalculating class weights...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights computed for {len(class_weight_dict)} classes")

print("\n" + "=" * 70)
print("Loading Pre-trained Model")
print("=" * 70)

# Load the pre-trained model
if not os.path.exists(PRETRAINED_MODEL_PATH):
    print(f"✗ Error: Pre-trained model not found at {PRETRAINED_MODEL_PATH}")
    print("  Please train the base model first using train_baseline.py")
    exit(1)

print(f"Loading model from: {PRETRAINED_MODEL_PATH}")
model = load_model(PRETRAINED_MODEL_PATH)
print("✓ Model loaded successfully!")

# Display model info (skip full summary for speed)
print(f"\nModel: {len(model.layers)} total layers")
print("Skipping model.summary() for faster startup...")

# Find the base model (EfficientNet or MobileNet) for fine-tuning
base_model = None
for layer in model.layers:
    if any(x in layer.name.lower() for x in ['efficientnet', 'mobilenet']) or hasattr(layer, 'layers'):
        base_model = layer
        break

if base_model is None or not hasattr(base_model, 'layers'):
    print("\n✗ Error: Could not find EfficientNet/MobileNet base model in the loaded model.")
    print("  This script requires a model with a transfer learning base.")
    print("  The model will be fine-tuned by unfreezing all layers instead.")
    
    # Fine-tune all layers
    print("\n" + "=" * 70)
    print("Fine-Tuning Configuration (All Layers)")
    print("=" * 70)
    
    # Count current trainable layers
    total_layers = len(model.layers)
    print(f"\nTotal model layers: {total_layers}")
    
    # Unfreeze all layers
    for layer in model.layers:
        layer.trainable = True
    
    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    print(f"All layers set to trainable: {trainable_count}/{total_layers}")
    
else:
    print(f"\nBase model found: {base_model.name}")
    print(f"Total layers in base model: {len(base_model.layers)}")

    print("\n" + "=" * 70)
    print("Fine-Tuning Configuration")
    print("=" * 70)

    # Unfreeze the base model layers for fine-tuning
    base_model.trainable = True

    # Freeze layers up to FINE_TUNE_AT
    freeze_until = min(FINE_TUNE_AT, len(base_model.layers))
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    # Count trainable and non-trainable parameters
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    non_trainable_count = len(base_model.layers) - trainable_count

    print(f"\nUnfreezing layers from layer {freeze_until} onwards")
    print(f"Trainable layers: {trainable_count}")
    print(f"Frozen layers: {freeze_until}")
    print(f"Total base model layers: {len(base_model.layers)}")

# Focal Loss for better class imbalance handling
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss focuses more on hard-to-classify examples.
    Better than categorical crossentropy for imbalanced datasets.
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return focal_loss_fixed

# Cosine Annealing with Warmup for smooth learning
def cosine_decay_with_warmup(epoch, lr, warmup_epochs=2, total_epochs=15):
    """Cosine annealing learning rate schedule with warmup"""
    if epoch < warmup_epochs:
        return INITIAL_LEARNING_RATE * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return INITIAL_LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * progress))

# Compile model with focal loss and Adam optimizer
print(f"\nCompiling model with Adam optimizer...")
print(f"  Learning rate: {INITIAL_LEARNING_RATE}")
print(f"  Loss function: Focal Loss (gamma=2.0, alpha=0.25)")
print(f"  Metrics: accuracy")

optimizer = Adam(learning_rate=INITIAL_LEARNING_RATE)

model.compile(
    optimizer=optimizer,
    loss=focal_loss(gamma=2.0, alpha=0.25),  # Focal loss for better accuracy
    metrics=['accuracy']
)

print("✓ Model compiled successfully!")

# Callbacks
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

callbacks = [
    # Save the best fine-tuned model
    ModelCheckpoint(
        filepath=FINETUNED_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    
    # Early stopping for optimal convergence
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,  # Balanced patience
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    
    # Reduce learning rate on plateau
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        min_lr=MIN_LEARNING_RATE,
        verbose=1,
        mode='max'
    ),
    
    # Cosine annealing learning rate schedule
    LearningRateScheduler(
        lambda epoch: cosine_decay_with_warmup(epoch, INITIAL_LEARNING_RATE, 
                                               warmup_epochs=3, 
                                               total_epochs=FINE_TUNE_EPOCHS),
        verbose=0
    ),
    
    # TensorBoard logging (optimized for speed)
    TensorBoard(
        log_dir=os.path.join(BASE_DIR, f'logs/finetune_{timestamp}'),
        histogram_freq=0,  # Disabled for speed (was 1)
        write_graph=False,  # Disabled for speed
        update_freq='epoch'
    )
]

print("\nCallbacks configured:")
print("  ✓ ModelCheckpoint - Save best model (val_accuracy)")
print("  ✓ EarlyStopping - Patience: 12 epochs")
print("  ✓ ReduceLROnPlateau - Reduce LR by 30% after 6 epochs")
print("  ✓ LearningRateScheduler - Cosine annealing with warmup")
print("  ✓ TensorBoard - Log training metrics")

print("\n" + "=" * 70)
print("Fine-Tuning Model")
print("=" * 70)
print(f"\nStarting fine-tuning for {FINE_TUNE_EPOCHS} epochs...")
print(f"Batch size: {BATCH_SIZE}")
print(f"Initial learning rate: {INITIAL_LEARNING_RATE}")
print(f"Training on: {'GPU' if len(gpus) > 0 else 'CPU'}")
print(f"Using class weights: Yes")

# Fine-tune the model with class weights
history_obj = model.fit(
    train_data,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=val_data,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

history = history_obj.history

print("\n✓ Fine-tuning completed!")

print("\n" + "=" * 70)
print("Model Evaluation")
print("=" * 70)

# Evaluate on validation set
print("\nEvaluating on validation set...")
val_loss, val_accuracy = model.evaluate(val_data, verbose=1)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

# Evaluate on test set (regular)
print("\nEvaluating on test set (regular)...")
test_loss, test_accuracy = model.evaluate(test_data, verbose=1)
print(f"Test Accuracy (Regular): {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Test-Time Augmentation (TTA) for Maximum Accuracy
print("\n" + "=" * 70)
print("Test-Time Augmentation (TTA) - Enhanced Prediction Accuracy")
print("=" * 70)

def predict_with_tta(model, data_dir, num_augmentations=10):
    """Use TTA to get more robust and accurate predictions"""
    predictions_list = []
    
    print(f"\nRunning {num_augmentations} augmented predictions...")
    for i in range(num_augmentations):
        if i == 0:
            # First pass without augmentation
            temp_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
        else:
            # Subsequent passes with light augmentation
            temp_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_fn,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True if i % 2 == 0 else False,
                brightness_range=[0.9, 1.1]
            )
        
        temp_generator = temp_datagen.flow_from_directory(
            data_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        preds = model.predict(temp_generator, verbose=0)
        predictions_list.append(preds)
        
        if (i + 1) % 2 == 0:
            print(f"  Progress: {i + 1}/{num_augmentations} augmentations complete")
    
    # Average all predictions
    final_predictions = np.mean(predictions_list, axis=0)
    print(f"\n✓ TTA complete - averaged {num_augmentations} predictions")
    return final_predictions

print(f"\nPerforming TTA with {TTA_AUGMENTATIONS} augmentations...")
tta_predictions = predict_with_tta(model, TEST_DIR, num_augmentations=TTA_AUGMENTATIONS)
tta_predicted_classes = np.argmax(tta_predictions, axis=1)
true_classes_tta = test_data.classes

tta_accuracy = np.mean(tta_predicted_classes == true_classes_tta)
print(f"\nTest Accuracy with TTA: {tta_accuracy * 100:.2f}%")
print(f"Accuracy Improvement: +{(tta_accuracy - test_accuracy) * 100:.2f}%")
print(f"\n{'='*70}")

# Save final fine-tuned model
final_finetuned_path = os.path.join(BASE_DIR, 'models/skin_disease_model_finetuned_final.keras')
model.save(final_finetuned_path)
print(f"\n✓ Final fine-tuned model saved to: {final_finetuned_path}")

print("\n" + "=" * 70)
print("Generating Plots and Reports")
print("=" * 70)

# Plot training history
def plot_finetuning_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    axes[0].plot(history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
    axes[0].set_title('Fine-Tuning: Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history['loss'], label='Training Loss', color='blue', linewidth=2)
    axes[1].plot(history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    axes[1].set_title('Fine-Tuning: Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved to: {save_path}")
    plt.close()

plot_path = os.path.join(BASE_DIR, f'results/finetuning_history_{timestamp}.png')
plot_finetuning_history(history, plot_path)

# Generate classification report (using TTA predictions for better accuracy)
print("\nGenerating classification report with TTA predictions...")
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Use TTA predictions for final report
predicted_classes = tta_predicted_classes
true_classes = true_classes_tta

# Classification report
report = classification_report(true_classes, predicted_classes, target_names=class_names, digits=4)
print("\nClassification Report:")
print(report)

# Save classification report
report_path = os.path.join(BASE_DIR, f'results/finetuned_classification_report_{timestamp}.txt')
with open(report_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("Skin Disease Detection - Fine-Tuned Model Report\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: {FINETUNED_MODEL_PATH}\n")
    f.write(f"Fine-tuning epochs: {FINE_TUNE_EPOCHS}\n")
    f.write(f"Initial learning rate: {INITIAL_LEARNING_RATE}\n")
    f.write(f"Layers unfrozen from: {FINE_TUNE_AT}\n")
    f.write(f"GPU used: {'Yes' if len(gpus) > 0 else 'No'}\n\n")
    f.write(f"Validation Accuracy: {val_accuracy * 100:.2f}%\n")
    f.write(f"Validation Loss: {val_loss:.4f}\n\n")
    f.write(f"Test Accuracy (Regular): {test_accuracy * 100:.2f}%\n")
    f.write(f"Test Accuracy (with TTA): {tta_accuracy * 100:.2f}%\n")
    f.write(f"TTA Improvement: +{(tta_accuracy - test_accuracy) * 100:.2f}%\n")
    f.write(f"Test Loss: {test_loss:.4f}\n\n")
    f.write("=" * 70 + "\n")
    f.write("Classification Report\n")
    f.write("=" * 70 + "\n")
    f.write(report)

print(f"✓ Classification report saved to: {report_path}")

# Confusion matrix
print("\nGenerating confusion matrix...")
plt.figure(figsize=(20, 16))
cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=class_names, 
    yticklabels=class_names,
    cbar_kws={'label': 'Count'}
)
plt.title('Fine-Tuned Model - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

cm_path = os.path.join(BASE_DIR, f'results/finetuned_confusion_matrix_{timestamp}.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrix saved to: {cm_path}")
plt.close()

# Calculate per-class accuracy
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

# Plot per-class accuracy comparison
plt.figure(figsize=(14, 8))
x = range(len(class_names))
bars = plt.bar(x, per_class_accuracy * 100, color='steelblue', alpha=0.8, edgecolor='black')

# Highlight best and worst classes
best_idx = np.argmax(per_class_accuracy)
worst_idx = np.argmin(per_class_accuracy)
bars[best_idx].set_color('green')
bars[worst_idx].set_color('red')

plt.axhline(y=test_accuracy*100, color='red', linestyle='--', linewidth=2, 
            label=f'Overall Accuracy: {test_accuracy*100:.2f}%')
plt.xlabel('Disease Class', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Fine-Tuned Model - Per-Class Accuracy', fontsize=14, fontweight='bold', pad=20)
plt.xticks(x, class_names, rotation=45, ha='right')
plt.ylim(0, 105)
plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

acc_path = os.path.join(BASE_DIR, f'results/finetuned_per_class_accuracy_{timestamp}.png')
plt.savefig(acc_path, dpi=300, bbox_inches='tight')
print(f"✓ Per-class accuracy plot saved to: {acc_path}")
plt.close()

print("\n" + "=" * 70)
print("Fine-Tuning Complete!")
print("=" * 70)

print(f"\nSummary:")
print(f"  GPU Used: {'Yes' if len(gpus) > 0 else 'No'}")
print(f"  Mixed Precision: Enabled (float16)")
print(f"  Loss Function: Focal Loss (better for imbalanced classes)")
print(f"  Optimizer: Adam with Cosine Annealing")
print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Fine-tuning Epochs: {FINE_TUNE_EPOCHS}")
print(f"  Layers Unfrozen: From layer {FINE_TUNE_AT}")
print(f"  Test Accuracy (Regular): {test_accuracy * 100:.2f}%")
print(f"  Test Accuracy (with TTA): {tta_accuracy * 100:.2f}% (+{(tta_accuracy - test_accuracy) * 100:.2f}%)")
print(f"  Best performing class: {class_names[best_idx]} ({per_class_accuracy[best_idx]*100:.2f}%)")
print(f"  Worst performing class: {class_names[worst_idx]} ({per_class_accuracy[worst_idx]*100:.2f}%)")

print(f"\nModels saved:")
print(f"  ✓ Best model: {FINETUNED_MODEL_PATH}")
print(f"  ✓ Final model: {final_finetuned_path}")

print(f"\nResults saved:")
print(f"  ✓ Training history: {plot_path}")
print(f"  ✓ Classification report: {report_path}")
print(f"  ✓ Confusion matrix: {cm_path}")
print(f"  ✓ Per-class accuracy: {acc_path}")

print("\n" + "=" * 70)
