"""
Skin Disease Detection - Training Script
Target: 80%+ Accuracy with Adam Optimizer
Optimized for CPU training with realistic expectations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import seaborn as sns

print("=" * 70)
print("SKIN DISEASE DETECTION - OPTIMIZED TRAINING")
print("Target: 80%+ Accuracy | Optimizer: Adam | Model: MobileNetV2")
print("=" * 70)

# Configuration - OPTIMIZED FOR 80% ACCURACY (Realistic Target)
IMG_SIZE = 224          # Good balance of detail and speed
BATCH_SIZE = 16         # Works well on CPU
EPOCHS = 60             # Enough for convergence without overfitting
NUM_CLASSES = 22
LEARNING_RATE = 0.0005  # Higher learning rate for Adam - faster training

print(f"\nTraining Configuration:")
print(f"  Model: MobileNetV2 (Faster than EfficientNet)")
print(f"  Optimizer: Adam")
print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Target Accuracy: 80%+")
print(f"  Device: CPU")

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

TRAIN_DIR = os.path.join(SCRIPT_DIR, 'SkinDisease/train')
VAL_DIR = os.path.join(SCRIPT_DIR, 'SkinDisease/val')
TEST_DIR = os.path.join(SCRIPT_DIR, 'SkinDisease/test')

# Create output directories
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)

# Data Augmentation - Optimized for skin diseases
print("\nSetting up data augmentation...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.75, 1.25],
    shear_range=0.2,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load datasets
print("\nLoading datasets...")
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Get class names
class_names = list(train_data.class_indices.keys())
print(f"\n✓ Found {len(class_names)} classes")
print(f"  Training samples: {train_data.samples}")
print(f"  Validation samples: {val_data.samples}")
print(f"  Test samples: {test_data.samples}")

# Save class names
with open(os.path.join(BASE_DIR, 'models/class_names.txt'), 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")

# Build Model - MobileNetV2 (Faster and efficient)
print("\n" + "=" * 70)
print("Building MobileNetV2 Model...")
print("=" * 70)

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=1.0  # Full model width
)

# Freeze base model initially
base_model.trainable = False

# Custom classification head - Optimized for 80% accuracy
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile with Adam optimizer
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n✓ Model built successfully")
print(f"  Total parameters: {model.count_params():,}")
trainable_params = sum([np.prod(w.shape) for w in model.trainable_weights])
print(f"  Trainable parameters: {trainable_params:,}")

# Calculate class weights
print("\nCalculating class weights for imbalanced data...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"✓ Class weights calculated")

# Callbacks
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(BASE_DIR, 'models/skin_disease_model_80_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1,
        mode='max',
        min_delta=0.002
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.3,
        patience=5,
        min_lr=1e-7,
        verbose=1,
        mode='max'
    )
]

# PHASE 1: Train with frozen base model
print("\n" + "=" * 70)
print("PHASE 1: Training with frozen base model")
print("=" * 70)

history_obj = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

history = history_obj.history
print("\n✓ Phase 1 completed!")

# PHASE 2: Fine-tuning - Unfreeze last layers
print("\n" + "=" * 70)
print("PHASE 2: Fine-tuning (unfreezing last layers)")
print("=" * 70)

# Unfreeze the last 30 layers
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with lower learning rate
fine_tune_lr = LEARNING_RATE / 5
model.compile(
    optimizer=Adam(learning_rate=fine_tune_lr),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Fine-tuning with learning rate: {fine_tune_lr}")
print(f"Trainable layers: {sum([layer.trainable for layer in base_model.layers])} out of {len(base_model.layers)}")

# Fine-tune
finetune_callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(BASE_DIR, 'models/skin_disease_model_80_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=12,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=4,
        min_lr=1e-8,
        verbose=1,
        mode='max'
    )
]

history_finetune = model.fit(
    train_data,
    epochs=30,  # Additional 30 epochs for fine-tuning
    validation_data=val_data,
    callbacks=finetune_callbacks,
    class_weight=class_weight_dict,
    verbose=1,
    initial_epoch=len(history['accuracy'])
)

# Merge histories
for key in history.keys():
    if key in history_finetune.history:
        history[key] = history[key] + history_finetune.history[key]

print("\n✓ Phase 2 fine-tuning completed!")

# Evaluate on test set
print("\n" + "=" * 70)
print("EVALUATING ON TEST SET")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(test_data, verbose=1)

print("\n" + "=" * 70)
print(f"FINAL TEST ACCURACY: {test_accuracy * 100:.2f}%")
print(f"FINAL TEST LOSS: {test_loss:.4f}")
print("=" * 70)

if test_accuracy >= 0.80:
    print("\n🎉 SUCCESS! Achieved 80%+ accuracy target!")
elif test_accuracy >= 0.75:
    print("\n✓ Very close! Consider training a bit longer.")
elif test_accuracy >= 0.70:
    print("\n✓ Good progress! May need more epochs or data augmentation.")
else:
    print("\n⚠ Consider adding more data or trying different augmentation.")

# Save final model
model.save(os.path.join(BASE_DIR, 'models/skin_disease_model_80_final.keras'))
print(f"\n✓ Final model saved")

# Plot training history
print("\nGenerating training plots...")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy plot
axes[0].plot(history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
axes[0].plot(history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
axes[0].axhline(y=0.80, color='green', linestyle='--', label='80% Target', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history['loss'], label='Training Loss', color='blue', linewidth=2)
axes[1].plot(history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'results/training_history_80.png'), dpi=300, bbox_inches='tight')
print(f"✓ Training history saved")

# Classification Report
print("\nGenerating classification report...")
test_data.reset()
predictions = model.predict(test_data, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data.classes

report = classification_report(true_classes, predicted_classes, target_names=class_names)
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print(report)

# Confusion Matrix
plt.figure(figsize=(18, 15))
cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Number of Predictions'})
plt.title('Confusion Matrix - Skin Disease Classification', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('True Class', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'results/confusion_matrix_80.png'), dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrix saved")

# Save detailed report
report_path = os.path.join(BASE_DIR, f'results/training_report_80_{timestamp}.txt')
with open(report_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("SKIN DISEASE DETECTION - TRAINING REPORT\n")
    f.write("Target: 80%+ Accuracy\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("CONFIGURATION:\n")
    f.write(f"  Model: MobileNetV2\n")
    f.write(f"  Optimizer: Adam\n")
    f.write(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}\n")
    f.write(f"  Batch Size: {BATCH_SIZE}\n")
    f.write(f"  Initial Learning Rate: {LEARNING_RATE}\n")
    f.write(f"  Fine-tune Learning Rate: {fine_tune_lr}\n")
    f.write(f"  Total Epochs: {len(history['accuracy'])}\n\n")
    f.write("RESULTS:\n")
    f.write(f"  Final Test Accuracy: {test_accuracy * 100:.2f}%\n")
    f.write(f"  Final Test Loss: {test_loss:.4f}\n")
    f.write(f"  Best Val Accuracy: {max(history['val_accuracy']) * 100:.2f}%\n\n")
    f.write("=" * 70 + "\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 70 + "\n\n")
    f.write(report)
    f.write("\n\n" + "=" * 70 + "\n")
    f.write("FILES SAVED:\n")
    f.write("=" * 70 + "\n")
    f.write("  - models/skin_disease_model_80_best.keras (best model)\n")
    f.write("  - models/skin_disease_model_80_final.keras (final model)\n")
    f.write("  - models/class_names.txt\n")
    f.write("  - results/training_history_80.png\n")
    f.write("  - results/confusion_matrix_80.png\n")
    f.write(f"  - results/training_report_80_{timestamp}.txt\n")

print(f"\n✓ Detailed report saved to: {report_path}")

# Summary
print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print("\nSaved Files:")
print("  📁 models/skin_disease_model_80_best.keras (BEST MODEL - Use this!)")
print("  📁 models/skin_disease_model_80_final.keras")
print("  📁 models/class_names.txt")
print("  📊 results/training_history_80.png")
print("  📊 results/confusion_matrix_80.png")
print(f"  📄 results/training_report_80_{timestamp}.txt")
print("\n" + "=" * 70)
print(f"FINAL ACCURACY: {test_accuracy * 100:.2f}%")
print("=" * 70)
