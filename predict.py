import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
import matplotlib.pyplot as plt

# Get the script's directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Parent directory (workspace root)

# Default paths - Auto-select best available model
DEFAULT_CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'models/class_names.txt')

# Try to find the best available model
model_priority = [
    'models/skin_disease_model_baseline_best.keras',  # New baseline with corrected preprocessing
    'models/skin_disease_model_baseline_final.keras',
    'models/skin_disease_model_best.keras',          # Kaggle GPU best checkpoint
    'models/skin_disease_model_final.keras',         # Legacy final model
    'models/skin_disease_model_finetuned.keras',     # Fine-tuned
    'models/skin_disease_model_cpu_best.keras',      # CPU-optimized model
    'models/skin_disease_model_cpu_final.keras',     # CPU final
    'models/skin_disease_model_fast.keras',          # Fast model (MobileNetV2) - lowest accuracy
]

DEFAULT_MODEL_PATH = None
for model_file in model_priority:
    model_path = os.path.join(BASE_DIR, model_file)
    if os.path.exists(model_path):
        DEFAULT_MODEL_PATH = model_path
        break

if DEFAULT_MODEL_PATH is None:
    DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, 'models/skin_disease_model_best.keras')


def _metadata_path_for_model(model_path):
    return os.path.splitext(model_path)[0] + '.meta.json'


def load_model_metadata(model_path):
    metadata_path = _metadata_path_for_model(model_path)
    if not os.path.exists(metadata_path):
        return {}

    try:
        with open(metadata_path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def preprocess_image_array(img_array, model_path, metadata=None):
    if metadata is None:
        metadata = load_model_metadata(model_path)

    preprocessing = metadata.get('preprocessing', 'legacy_rescale')
    if preprocessing == 'mobilenet_v2':
        return mobilenet_v2_preprocess_input(img_array)
    return img_array / 255.0

def load_class_names(path=None):
    """Load class names from file."""
    if path is None:
        path = DEFAULT_CLASS_NAMES_PATH
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def predict_skin_disease(img_path, model_path=None, img_size=None):
    """
    Predict skin disease from an image.
    
    Args:
        img_path: Path to the image file
        model_path: Path to the trained model (auto-detects if None)
        img_size: Image size (auto-detected based on model if None)
        
    Returns:
        Predicted class name and confidence scores
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    metadata = load_model_metadata(model_path)
    
    # Auto-detect image size based on model name if not explicitly provided
    if img_size is None:
        if metadata.get('image_size'):
            img_size = int(metadata['image_size'])
        elif 'fast' in model_path.lower():
            img_size = 160  # Fast models use 160x160
        elif 'cpu' in model_path.lower():
            img_size = 192  # CPU models use 192x192
        else:
            img_size = 224  # Default to 224x224
    
    print(f"Using model: {os.path.basename(model_path)}")
    print(f"Image size: {img_size}x{img_size}")
    
    # Load model
    model = load_model(model_path)
    
    # Load class names
    class_names = load_class_names()
    
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_image_array(img_array, model_path, metadata)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Get top 5 predictions
    top_5_idx = np.argsort(predictions[0])[-5:][::-1]
    top_5_predictions = [(class_names[i], predictions[0][i] * 100) for i in top_5_idx]
    
    return {
        'predicted_class': class_names[predicted_class_idx],
        'confidence': confidence,
        'top_5_predictions': top_5_predictions,
        'all_predictions': dict(zip(class_names, predictions[0] * 100))
    }

def visualize_prediction(img_path, model_path=None, img_size=None):
    """Visualize the prediction with the image and bar chart."""
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    
    # Auto-detect image size if not provided
    if img_size is None:
        if 'fast' in model_path.lower():
            img_size = 160
        elif 'cpu' in model_path.lower():
            img_size = 192
        elif 'enhanced' in model_path.lower():
            img_size = 224
        else:
            img_size = 224
    
    result = predict_skin_disease(img_path, model_path, img_size)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Show the image
    img = image.load_img(img_path, target_size=(img_size, img_size))
    axes[0].imshow(img)
    axes[0].set_title(f"Predicted: {result['predicted_class']}\nConfidence: {result['confidence']:.2f}%")
    axes[0].axis('off')
    
    # Show top 5 predictions as bar chart
    classes = [p[0] for p in result['top_5_predictions']]
    confidences = [p[1] for p in result['top_5_predictions']]
    colors = ['green' if i == 0 else 'steelblue' for i in range(5)]
    
    bars = axes[1].barh(classes[::-1], confidences[::-1], color=colors[::-1])
    axes[1].set_xlabel('Confidence (%)')
    axes[1].set_title('Top 5 Predictions')
    axes[1].set_xlim(0, 100)
    
    # Add percentage labels on bars
    for bar, conf in zip(bars, confidences[::-1]):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{conf:.1f}%', va='center')
    
    plt.tight_layout()
    
    return result, fig

# Health recommendations based on skin disease
HEALTH_RECOMMENDATIONS = {
    'Acne': 'Keep skin clean, avoid touching face, use non-comedogenic products. Consult a dermatologist for persistent acne.',
    'Actinic_Keratosis': 'Protect skin from sun exposure, use SPF 30+ sunscreen. Consult a dermatologist - this may need treatment.',
    'Benign_tumors': 'Monitor for changes in size, shape, or color. Regular skin check-ups recommended.',
    'Bullous': 'Avoid scratching, keep affected areas clean. Seek medical attention for proper diagnosis and treatment.',
    'Candidiasis': 'Keep affected area dry, wear loose clothing. Antifungal treatment may be needed.',
    'DrugEruption': 'Stop suspected medication and consult healthcare provider immediately.',
    'Eczema': 'Moisturize regularly, avoid triggers, use mild soaps. Consider consulting an allergist.',
    'Infestations_Bites': 'Clean the affected area, apply anti-itch cream. Seek medical care if infection develops.',
    'Lichen': 'Avoid scratching, use prescribed corticosteroid creams. Manage stress levels.',
    'Lupus': 'Protect from sun, manage stress, follow prescribed treatment. Regular rheumatologist visits needed.',
    'Moles': 'Monitor for ABCDE signs (Asymmetry, Border, Color, Diameter, Evolution). Annual skin exams recommended.',
    'Psoriasis': 'Moisturize, avoid triggers, consider phototherapy. Consult dermatologist for treatment options.',
    'Rosacea': 'Identify and avoid triggers, use gentle skincare, protect from sun. Consult dermatologist.',
    'Seborrh_Keratoses': 'Usually benign, can be removed for cosmetic reasons. Monitor for changes.',
    'SkinCancer': 'URGENT: Seek immediate medical attention. Early treatment is crucial.',
    'Sun_Sunlight_Damage': 'Use SPF 30+ sunscreen, wear protective clothing, avoid peak sun hours.',
    'Tinea': 'Keep area clean and dry, use antifungal treatments. Avoid sharing personal items.',
    'Unknown_Normal': 'Skin appears normal. Maintain good skincare routine and regular check-ups.',
    'Vascular_Tumors': 'Monitor for changes, consult dermatologist for evaluation and treatment options.',
    'Vasculitis': 'Seek medical attention for proper diagnosis. May require systemic treatment.',
    'Vitiligo': 'Protect affected areas from sun, consider phototherapy. Consult dermatologist.',
    'Warts': 'Avoid picking, consider OTC treatments or see dermatologist for removal options.'
}

def get_health_recommendation(disease_name):
    """Get health recommendation for a detected skin condition."""
    return HEALTH_RECOMMENDATIONS.get(disease_name, 'Please consult a dermatologist for proper diagnosis and treatment.')

if __name__ == '__main__':
    import sys
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Predict skin disease from an image')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--no-viz', help='Skip visualization, only show text output', action='store_true')
    parser.add_argument('--save-only', help='Save graph without displaying', action='store_true')
    
    # Handle old-style single argument (backwards compatibility)
    if len(sys.argv) == 2 and not sys.argv[1].startswith('--'):
        args = argparse.Namespace(image_path=sys.argv[1], no_viz=False, save_only=False)
    else:
        args = parser.parse_args()
    
    img_path = args.image_path
    if os.path.exists(img_path):
        print("\n" + "=" * 50)
        print("Skin Disease Prediction")
        print("=" * 50)
        
        if args.no_viz:
            # Just predict without visualization
            result = predict_skin_disease(img_path)
        else:
            # Create visualization
            result, fig = visualize_prediction(img_path)
            
            # Auto-save the graph
            output_file = os.path.splitext(os.path.basename(img_path))[0] + '_prediction.png'
            results_dir = os.path.join(BASE_DIR, 'results')
            os.makedirs(results_dir, exist_ok=True)
            output_path = os.path.join(results_dir, output_file)
            
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n📊 Visualization saved to: {output_path}")
            
            # Show the graph automatically (unless save-only mode)
            if not args.save_only:
                print("📊 Displaying graph... (close the window to continue)")
                plt.show()
            else:
                plt.close()
        
        print(f"\nPredicted Condition: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        
        print("\nTop 5 Predictions:")
        for i, (cls, conf) in enumerate(result['top_5_predictions'], 1):
            print(f"  {i}. {cls}: {conf:.2f}%")
        
        print("\n" + "=" * 50)
        print("Health Recommendation:")
        print("=" * 50)
        print(get_health_recommendation(result['predicted_class']))
        print("\nNote: This is an AI-based prediction. Always consult a healthcare professional for accurate diagnosis.")
    else:
        print(f"Error: Image file not found: {img_path}")
        sys.exit(1)
