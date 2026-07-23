import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from datetime import datetime

print("=" * 50)
print("Skin Disease Detection - Model Validation")
print("=" * 50)

# Configuration - Updated for new models
IMG_SIZE = 224  # Updated to match new training size
BATCH_SIZE = 32
NUM_CLASSES = 22

# Get the script's directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Parent directory (workspace root)

# Paths - Support multiple model types
VAL_DIR = os.path.join(SCRIPT_DIR, 'SkinDisease/val')
TEST_DIR = os.path.join(SCRIPT_DIR, 'SkinDisease/test')
MODEL_PATH = os.path.join(BASE_DIR, 'models/skin_disease_model_baseline_best.keras')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'models/class_names.txt')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Check for alternative models if main one doesn't exist
if not os.path.exists(MODEL_PATH):
    alternative_models = [
        'skin_disease_model_baseline_final.keras',
        'skin_disease_model_best.keras',
        'skin_disease_model_final.keras',
        'skin_disease_model_finetuned.keras',
        'skin_disease_model_cpu_best.keras',
        'skin_disease_model_cpu_final.keras',
        'skin_disease_model_fast.keras'
    ]
    for alt_model in alternative_models:
        alt_path = os.path.join(BASE_DIR, f'models/{alt_model}')
        if os.path.exists(alt_path):
            MODEL_PATH = alt_path
            print(f"Using alternative model: {alt_model}")
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


def build_eval_datagen(metadata):
    preprocessing = metadata.get('preprocessing', 'legacy_rescale')
    if preprocessing == 'mobilenet_v2':
        print("Using MobileNetV2 preprocessing for validation")
        return ImageDataGenerator(preprocessing_function=mobilenet_v2_preprocess_input)
    print("Using legacy rescale preprocessing for validation")
    return ImageDataGenerator(rescale=1./255)

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load class names
print("\nLoading class names...")
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
print(f"Classes: {class_names}")

# Load trained model
print(f"\nLoading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

metadata = load_model_metadata(MODEL_PATH)
if metadata.get('image_size'):
    IMG_SIZE = int(metadata['image_size'])
    print(f"Using metadata image size: {IMG_SIZE}")

# Data generator for validation (no augmentation)
val_datagen = build_eval_datagen(metadata)

print("\nLoading validation data...")
val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Important: don't shuffle for evaluation
)

print(f"\nValidation samples: {val_data.samples}")
print(f"Batches: {len(val_data)}")

# Evaluate model on validation data
print("\n" + "=" * 50)
print("Evaluating model on validation dataset...")
print("=" * 50)

val_loss, val_accuracy = model.evaluate(val_data, verbose=1)
print(f"\nValidation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

# Get predictions
print("\nGenerating predictions...")
val_data.reset()
predictions = model.predict(val_data, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_data.classes
class_labels = list(val_data.class_indices.keys())

# Calculate detailed metrics
print("\n" + "=" * 50)
print("Detailed Metrics")
print("=" * 50)

# Overall metrics
accuracy = accuracy_score(true_classes, predicted_classes)
precision, recall, f1_score, support = precision_recall_fscore_support(
    true_classes, predicted_classes, average='weighted'
)

print(f"\nOverall Metrics:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1_score:.4f}")

# Per-class metrics
print("\n" + "=" * 50)
print("Classification Report")
print("=" * 50)
report = classification_report(
    true_classes, 
    predicted_classes, 
    target_names=class_labels,
    digits=4
)
print(report)

# Save classification report
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = os.path.join(RESULTS_DIR, f'validation_report_{timestamp}.txt')
with open(report_path, 'w') as f:
    f.write("=" * 50 + "\n")
    f.write("Skin Disease Detection - Validation Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Validation samples: {val_data.samples}\n\n")
    f.write(f"Validation Loss: {val_loss:.4f}\n")
    f.write(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)\n\n")
    f.write(f"Overall Metrics:\n")
    f.write(f"  Accuracy:  {accuracy:.4f}\n")
    f.write(f"  Precision: {precision:.4f}\n")
    f.write(f"  Recall:    {recall:.4f}\n")
    f.write(f"  F1-Score:  {f1_score:.4f}\n\n")
    f.write("=" * 50 + "\n")
    f.write("Classification Report\n")
    f.write("=" * 50 + "\n")
    f.write(report)

print(f"\nValidation report saved to: {report_path}")

# Generate confusion matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(20, 16))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels,
    cbar_kws={'label': 'Count'}
)
plt.title('Confusion Matrix - Validation Set', fontsize=16, pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

cm_path = os.path.join(RESULTS_DIR, f'confusion_matrix_{timestamp}.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"Confusion matrix saved to: {cm_path}")
plt.close()

# Plot normalized confusion matrix
print("\nGenerating normalized confusion matrix...")
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(20, 16))
sns.heatmap(
    cm_normalized, 
    annot=True, 
    fmt='.2f', 
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels,
    cbar_kws={'label': 'Proportion'}
)
plt.title('Normalized Confusion Matrix - Validation Set', fontsize=16, pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

cm_norm_path = os.path.join(RESULTS_DIR, f'confusion_matrix_normalized_{timestamp}.png')
plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
print(f"Normalized confusion matrix saved to: {cm_norm_path}")
plt.close()

# Calculate per-class accuracy
print("\n" + "=" * 50)
print("Per-Class Accuracy")
print("=" * 50)
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
for i, class_name in enumerate(class_labels):
    print(f"{class_name:25s}: {per_class_accuracy[i]:.4f} ({per_class_accuracy[i]*100:.2f}%)")

# Plot per-class accuracy
plt.figure(figsize=(14, 8))
bars = plt.bar(range(len(class_labels)), per_class_accuracy * 100, color='steelblue', alpha=0.8)
plt.axhline(y=accuracy*100, color='red', linestyle='--', linewidth=2, label=f'Overall Accuracy: {accuracy*100:.2f}%')
plt.xlabel('Disease Class', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Per-Class Accuracy on Validation Set', fontsize=14, pad=20)
plt.xticks(range(len(class_labels)), class_labels, rotation=45, ha='right')
plt.ylim(0, 105)
plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

acc_path = os.path.join(RESULTS_DIR, f'per_class_accuracy_{timestamp}.png')
plt.savefig(acc_path, dpi=300, bbox_inches='tight')
print(f"\nPer-class accuracy plot saved to: {acc_path}")
plt.close()

# Top-k accuracy analysis
print("\n" + "=" * 50)
print("Top-K Accuracy Analysis")
print("=" * 50)

for k in [1, 3, 5]:
    top_k_predictions = np.argsort(predictions, axis=1)[:, -k:]
    correct = sum([true_class in top_k_pred for true_class, top_k_pred in zip(true_classes, top_k_predictions)])
    top_k_accuracy = correct / len(true_classes)
    print(f"Top-{k} Accuracy: {top_k_accuracy:.4f} ({top_k_accuracy*100:.2f}%)")

# Analyze misclassifications
print("\n" + "=" * 50)
print("Misclassification Analysis")
print("=" * 50)

misclassified_indices = np.where(predicted_classes != true_classes)[0]
print(f"Total misclassified samples: {len(misclassified_indices)} out of {len(true_classes)} ({len(misclassified_indices)/len(true_classes)*100:.2f}%)")

# Find most confused pairs
print("\nMost confused class pairs (Top 10):")
confused_pairs = []
for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        if i != j and cm[i, j] > 0:
            confused_pairs.append((class_labels[i], class_labels[j], cm[i, j]))

confused_pairs.sort(key=lambda x: x[2], reverse=True)
for i, (true_label, pred_label, count) in enumerate(confused_pairs[:10], 1):
    print(f"{i:2d}. {true_label:25s} -> {pred_label:25s}: {count} times")

# Summary statistics
print("\n" + "=" * 50)
print("Summary")
print("=" * 50)
print(f"\nValidation completed successfully!")
print(f"Total validation samples: {val_data.samples}")
print(f"Overall accuracy: {val_accuracy*100:.2f}%")
print(f"Best performing class: {class_labels[np.argmax(per_class_accuracy)]} ({per_class_accuracy.max()*100:.2f}%)")
print(f"Worst performing class: {class_labels[np.argmin(per_class_accuracy)]} ({per_class_accuracy.min()*100:.2f}%)")
print(f"\nAll results saved to: {RESULTS_DIR}")
print("\n" + "=" * 50)
print("Validation Complete!")
print("=" * 50)
