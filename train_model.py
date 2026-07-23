import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from datetime import datetime

print("=" * 50)
print("Skin Disease Detection - Model Training")
print("=" * 50)

# Configuration
IMG_SIZE = 160  # Balanced for speed and accuracy
BATCH_SIZE = 64  # Increased for better throughput
EPOCHS = 30  # Reduced epochs
NUM_CLASSES = 22
LEARNING_RATE = 0.001

# Get the script's directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Parent directory (workspace root)

# Data directories (relative to script location)
TRAIN_DIR = os.path.join(SCRIPT_DIR, 'SkinDisease/train')
VAL_DIR = os.path.join(SCRIPT_DIR, 'SkinDisease/val')
TEST_DIR = os.path.join(SCRIPT_DIR, 'SkinDisease/test')

# Create output directories in workspace root
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)

# Data Augmentation for training (simplified for speed)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# No augmentation for validation and test - only rescaling
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

print("\nLoading training data...")
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

print("\nLoading validation data...")
val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("\nLoading test data...")
test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Get class names
class_names = list(train_data.class_indices.keys())
print(f"\nNumber of classes: {len(class_names)}")
print(f"Classes: {class_names}")

# Save class names for later use
with open(os.path.join(BASE_DIR, 'models/class_names.txt'), 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")

# Build the model using Transfer Learning (MobileNetV2)
print("\nBuilding model with MobileNetV2 backbone...")

# Load pre-trained MobileNetV2 without top layers
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base model layers initially
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

callbacks = [
    # Save the best model
    ModelCheckpoint(
        filepath=os.path.join(BASE_DIR, 'models/skin_disease_model_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    # Early stopping
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate on plateau
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    # TensorBoard logging
    TensorBoard(
        log_dir=os.path.join(BASE_DIR, f'logs/fit_{timestamp}'),
        histogram_freq=1
    )
]

# Train the model
print("\n" + "=" * 50)
print("Training model (base model frozen)")
print("=" * 50)

history_obj = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=callbacks,
    verbose=1
)

history = history_obj.history

# Evaluate on test set
print("\n" + "=" * 50)
print("Evaluating on test set...")
print("=" * 50)

test_loss, test_accuracy = model.evaluate(test_data, verbose=1)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Save the final model
model.save(os.path.join(BASE_DIR, 'models/skin_disease_model_final.keras'))
print("\nModel saved to 'models/skin_disease_model_final.keras'")

# Plot training history
def plot_training_history(history, save_path=os.path.join(BASE_DIR, 'results/training_history.png')):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
     
     
    # Plot accuracy
    axes[0].plot(history['accuracy'], label='Training Accuracy', color='blue')
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy', color='orange')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc='lower right')
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history['loss'], label='Training Loss', color='blue')
    axes[1].plot(history['val_loss'], label='Validation Loss', color='orange')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nTraining history plot saved to '{save_path}'")

plot_training_history(history)

# Generate classification report
print("\n" + "=" * 50)
print("Generating Classification Report...")
print("=" * 50)

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Get predictions
test_data.reset()
predictions = model.predict(test_data, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data.classes

# Classification report
report = classification_report(true_classes, predicted_classes, target_names=class_names)
print("\nClassification Report:")
print(report)

# Save classification report
with open(os.path.join(BASE_DIR, 'results/classification_report.txt'), 'w') as f:
    f.write("Skin Disease Detection - Classification Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")
    f.write(f"Test Loss: {test_loss:.4f}\n\n")
    f.write(report)

# Confusion matrix
plt.figure(figsize=(20, 16))
cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'results/confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.show()
print("\nConfusion matrix saved to 'results/confusion_matrix.png'")

print("\n" + "=" * 50)
print("Training Complete!")
print("=" * 50)
print(f"\nSummary:")
print(f"  - Best model saved to: models/skin_disease_model_best.keras")
print(f"  - Final model saved to: models/skin_disease_model_final.keras")
print(f"  - Class names saved to: models/class_names.txt")
print(f"  - Training plots saved to: results/training_history.png")
print(f"  - Confusion matrix saved to: results/confusion_matrix.png")
print(f"  - Classification report saved to: results/classification_report.txt")
print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")
