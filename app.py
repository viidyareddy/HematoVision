import os
import json
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, jsonify, url_for
import numpy as np

# Disable TensorFlow optimization warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("TensorFlow not available. Running in demo mode.")

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model if available
model = None
if MODEL_AVAILABLE:
    try:
        model = load_model('waste_classifier_vgg16.h5')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        MODEL_AVAILABLE = False

# Define class labels and descriptions
class_info = {
    'Biodegradable': {
        'description': 'This appears to be organic waste that can decompose naturally. Consider composting this item to reduce environmental impact.',
        'className': 'biodegradable',
        'icon': '🌱'
    },
    'Recyclable': {
        'description': 'This item can be recycled! Please clean it and place it in the appropriate recycling bin to give it a new life.',
        'className': 'recyclable',
        'icon': '♻️'
    },
    'Trash': {
        'description': 'This item should be disposed of in regular trash. Unfortunately, it cannot be recycled or composted safely.',
        'className': 'trash',
        'icon': '🗑️'
    }
}

class_labels = list(class_info.keys())

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Preprocess image for model prediction"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def simulate_prediction():
    """Simulate model prediction for demo purposes"""
    import random
    class_index = random.randint(0, len(class_labels) - 1)
    confidence = round(random.uniform(0.7, 0.99), 3)
    
    # Create fake prediction array
    prediction = np.zeros(len(class_labels))
    prediction[class_index] = confidence
    
    return prediction

@app.route('/')
def home():
    """Home route - show upload form"""
    return render_template('index.html', model_available=MODEL_AVAILABLE)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction route - handle file upload and classification"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded.'
            })

        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected.'
            })

        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload an image file.'
            })

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Make prediction
        if MODEL_AVAILABLE and model is not None:
            try:
                # Preprocess and predict
                img_array = preprocess_image(filepath)
                prediction = model.predict(img_array)
            except Exception as e:
                print(f"Model prediction error: {e}")
                prediction = simulate_prediction()
        else:
            # Use simulation if model not available
            prediction = simulate_prediction()

        # Get prediction results
        class_index = np.argmax(prediction)
        predicted_class = class_labels[class_index]
        confidence = float(prediction[0][class_index])

        # Get class information
        result_info = class_info[predicted_class].copy()
        result_info['label'] = predicted_class
        result_info['confidence'] = round(confidence * 100, 1)
        result_info['image_path'] = url_for('static', filename=f'uploads/{filename}')

        return jsonify({
            'success': True,
            'result': result_info,
            'all_predictions': {
                class_labels[i]: round(float(prediction[0][i]) * 100, 1) 
                for i in range(len(class_labels))
            }
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': f'An error occurred during prediction: {str(e)}'
        })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_available': MODEL_AVAILABLE,
        'model_loaded': model is not None
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error occurred.'
    }), 500

if __name__ == '__main__':
    print(f"Starting Waste Classifier App...")
    print(f"Model Available: {MODEL_AVAILABLE}")
    print(f"Upload Directory: {UPLOAD_FOLDER}")
    
    # Create templates directory structure info
    print("\nMake sure you have the following directory structure:")
    print("├── app.py")
    print("├── waste_classifier_vgg16.h5 (optional)")
    print("├── templates/")
    print("│   └── index.html")
    print("└── static/")
    print("    └── uploads/ (will be created automatically)")
    
    app.run(debug=True, host='0.0.0.0', port=5000)