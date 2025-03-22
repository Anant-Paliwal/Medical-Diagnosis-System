# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from PIL import Image
import json
import os
import logging
import traceback
import io

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
symptom_model = None
xray_model = None
Training = None
X = None
y = None
y_encoder = None
scaler = None
medical_dialog = []

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Disease descriptions dictionary (from original code)
disease_descriptions = {
    "fungal_infection": {
        "description": "A fungal infection is a condition caused by harmful fungi that can affect various parts of the body.",
        "symptoms_detail": {
            "itching": "Persistent itching sensation on affected areas",
            "skin_rash": "Visible rash or discoloration on skin",
            "nodal_skin_eruptions": "Raised bumps or lesions on skin"
        },
        "precautions": [
            "Keep affected areas clean and dry",
            "Avoid sharing personal items",
            "Use antifungal medications as prescribed",
            "Maintain good personal hygiene"
        ],
        "treatment": [
            "Topical antifungal medications",
            "Oral antifungal medications in severe cases",
            "Regular cleaning and dressing of affected areas",
            "Lifestyle modifications to prevent recurrence"
        ],
        "severity": "Mild to Moderate",
        "recovery_time": "2-4 weeks with proper treatment",
        "common_age_groups": "All age groups",
        "risk_factors": [
            "Weakened immune system",
            "Excessive sweating",
            "Poor hygiene",
            "Tight clothing"
        ]
    }
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_initial_data():
    """Load all required data and models with error handling"""
    global symptom_model, xray_model, Training, X, y, y_encoder, scaler, medical_dialog
    
    try:
        # Load the symptom prediction model
        if os.path.exists('disease_prediction_model_advanced.h5'):
            symptom_model = load_model('disease_prediction_model_advanced.h5')
            logger.info("Symptom prediction model loaded successfully")
        else:
            raise FileNotFoundError("Symptom prediction model file not found")

        # Load the X-ray model
        if os.path.exists('covid19_xray_model.h5'):
            xray_model = load_model('covid19_xray_model.h5')
            logger.info("X-ray model loaded successfully")
        else:
            raise FileNotFoundError("X-ray model file not found")

        # Load training data
        training_file = "Training.csv"
        if os.path.exists(training_file):
            Training = pd.read_csv(training_file)
            X = Training.iloc[:, :-1]
            y = Training.iloc[:, -1]
            logger.info("Training data loaded successfully")
        else:
            raise FileNotFoundError("Training dataset not found")

        # Initialize encoders and scalers
        y_encoder = LabelEncoder()
        y_encoder.fit(y)
        scaler = StandardScaler()
        scaler.fit(X)

        # Load medical dialogue dataset
        medical_dialog_file = "en_medical_dialog.json"
        if os.path.exists(medical_dialog_file):
            with open(medical_dialog_file, 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        medical_dialog.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipped malformed JSON line: {e}")

    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def get_disease_details(disease_name):
    """Get detailed information about a disease"""
    try:
        disease_key = disease_name.lower().replace(' ', '_')
        if disease_key in disease_descriptions:
            return disease_descriptions[disease_key]
        else:
            return {
                "description": f"Details for {disease_name} are being updated.",
                "symptoms_detail": {},
                "precautions": ["Consult a healthcare professional for specific advice"],
                "treatment": ["Seek medical attention for proper diagnosis and treatment"],
                "severity": "Varies",
                "recovery_time": "Depends on various factors",
                "common_age_groups": "All age groups",
                "risk_factors": ["Various factors may contribute to this condition"]
            }
    except Exception as e:
        logger.error(f"Error getting disease details: {str(e)}")
        return None

def enhance_symptoms(symptoms):
    """Enhance symptoms with error handling"""
    try:
        enhanced_symptoms = set(symptoms)
        for dialog in medical_dialog:
            if isinstance(dialog, dict) and 'symptoms' in dialog:
                for symptom in symptoms:
                    if symptom in dialog['symptoms']:
                        enhanced_symptoms.update(dialog['symptoms'])
        return list(enhanced_symptoms)
    except Exception as e:
        logger.error(f"Error in enhance_symptoms: {str(e)}")
        return symptoms

def get_symptom_severity(symptom):
    """Get severity information for a symptom"""
    severity_levels = {
        'high': ['chest_pain', 'breathlessness', 'unconsciousness'],
        'medium': ['fever', 'vomiting', 'fatigue'],
        'low': ['headache', 'skin_rash', 'back_pain']
    }
    
    for level, symptoms in severity_levels.items():
        if symptom in symptoms:
            return level
    return 'unknown'

def predict_disease(symptoms):
    """Make prediction with detailed error handling"""
    try:
        logger.info(f"Making prediction for symptoms: {symptoms}")
        
        if not symptoms or not isinstance(symptoms, list):
            raise ValueError("Invalid symptoms input")

        # Validate symptoms against training data columns
        valid_symptoms = [s for s in symptoms if s in Training.columns]
        if not valid_symptoms:
            raise ValueError("No valid symptoms found. Please select symptoms from the provided list.")

        symptoms = enhance_symptoms(valid_symptoms)
        
        input_data = np.zeros((1, X.shape[1]))
        symptom_details = {}
        
        for symptom in symptoms:
            if symptom in Training.columns:
                input_data[0, Training.columns.get_loc(symptom)] = 1
                symptom_details[symptom] = get_symptom_severity(symptom)

        input_scaled = scaler.transform(input_data)
        prediction = symptom_model.predict(input_scaled)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        predictions = []
        
        for idx in top_3_indices:
            disease = y_encoder.inverse_transform([idx])[0]
            confidence = float(prediction[0][idx] * 100)
            disease_info = get_disease_details(disease)
            
            predictions.append({
                'disease': disease,
                'confidence': round(confidence, 2),
                'details': disease_info
            })

        return predictions, symptom_details

    except Exception as e:
        logger.error(f"Error in predict_disease: {str(e)}")
        logger.error(traceback.format_exc())
        raise
def preprocess_image(img):
    """Preprocess image for X-ray model"""
    try:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize((256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        logger.info("Image preprocessed successfully")
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

# Routes
@app.route('/')
def home():
    """Render home page"""
    try:
        available_symptoms = list(X.columns)
        return render_template('index.html', symptoms=available_symptoms)
    except Exception as e:
        logger.error(f"Error in home route: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict_symptoms', methods=['POST'])
def predict_symptoms():
    """Handle symptom-based prediction requests"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        symptoms = data.get('symptoms', [])
        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400

        logger.info(f"Received prediction request for symptoms: {symptoms}")
        
        symptoms = [symptom.strip().lower() for symptom in symptoms if isinstance(symptom, str)]
        predictions, symptom_details = predict_disease(symptoms)
        
        return jsonify({
            'predictions': predictions,
            'symptoms': symptoms,
            'symptom_details': symptom_details
        })

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'An error occurred during prediction'}), 500

@app.route('/predict_xray', methods=['POST'])
def predict_xray():
    """Handle X-ray image prediction requests"""
    try:
        if xray_model is None:
            return jsonify({'error': 'X-ray model not loaded. Please check server logs.'}), 500

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a PNG, JPG, or JPEG image.'}), 400
        
        try:
            img = Image.open(file.stream)
            logger.info(f"Image opened successfully: {file.filename}")
            
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            img.save(filename)
            logger.info(f"Image saved to {filename}")
            
            processed_img = preprocess_image(img)
            prediction = xray_model.predict(processed_img)
            predicted_prob = float(prediction[0][0])
            predicted_class = 'PNEUMONIA' if predicted_prob >= 0.5 else 'NORMAL'
            confidence = predicted_prob if predicted_class == 'PNEUMONIA' else 1 - predicted_prob
            
            logger.info(f"Prediction made successfully: {predicted_class} with confidence {confidence}")
            
            return jsonify({
                'prediction': predicted_class,
                'confidence': float(confidence * 100),
                'image_path': filename
            })
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/disease/<disease_name>')
def disease_info(disease_name):
    """Get detailed information about a specific disease"""
    try:
        details = get_disease_details(disease_name)
        if details:
            return jsonify(details)
        return jsonify({'error': 'Disease information not found'}), 404
    except Exception as e:
        logger.error(f"Error getting disease info: {str(e)}")
        return jsonify({'error': 'An error occurred'}), 500
    

@app.route('/get_symptoms')
def get_symptoms():
    try:
        available_symptoms = list(X.columns)
        return jsonify({
            'symptoms': [{'value': s, 'label': s.replace('_', ' ').title()} 
                        for s in available_symptoms]
        })
    except Exception as e:
        logger.error(f"Error getting symptoms: {str(e)}")
        return jsonify({'error': 'Failed to get symptoms list'}), 500    

if __name__ == '__main__':
    load_initial_data()
    app.run(debug=True)