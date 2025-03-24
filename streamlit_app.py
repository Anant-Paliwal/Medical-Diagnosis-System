import streamlit as st
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

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
symptom_model = None
xray_model = None
Training = None
X = None
y = None
y_encoder = None
scaler = None
medical_dialog = []

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
            st.error("Symptom prediction model file not found")
            return False

        # Load the X-ray model
        if os.path.exists('covid19_xray_model_compressed.h5'):
            xray_model = load_model('covid19_xray_model_compressed.h5')
            logger.info("X-ray model loaded successfully")
        else:
            st.error("X-ray model file not found")
            return False

        # Load training data
        training_file = "Training.csv"
        if os.path.exists(training_file):
            Training = pd.read_csv(training_file)
            X = Training.iloc[:, :-1]
            y = Training.iloc[:, -1]
            logger.info("Training data loaded successfully")
        else:
            st.error("Training dataset not found")
            return False

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
        
        return True

    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Initialization error: {str(e)}")
        return False

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

def main():
    # Set page title and favicon
    st.set_page_config(
        page_title="Medical Diagnosis Assistant",
        page_icon="🏥",
        layout="wide"
    )
    
    # Header
    st.title("🏥 Medical Diagnosis Assistant")
    st.markdown("### Symptom-based Disease Prediction & X-ray Analysis")
    
    # Initialize data
    success = load_initial_data()
    if not success:
        st.error("Failed to initialize the application. Please check if all required files are available.")
        return
    
    # Create sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose the diagnosis method:", 
                               ["Symptom-based Diagnosis", "X-ray Analysis"])
    
    # Display chosen mode
    if app_mode == "Symptom-based Diagnosis":
        symptom_based_diagnosis()
    else:
        xray_analysis()

def symptom_based_diagnosis():
    st.header("Symptom-based Disease Prediction")
    st.write("Select the symptoms you are experiencing for a preliminary diagnosis.")
    
    # Get available symptoms and sort them alphabetically
    available_symptoms = sorted([s.replace('_', ' ').title() for s in X.columns])
    
    # Create a multiselect for symptoms
    selected_symptoms = st.multiselect(
        "Select your symptoms:",
        available_symptoms
    )
    
    # Convert selected symptoms back to the format needed for prediction
    formatted_symptoms = [s.lower().replace(' ', '_') for s in selected_symptoms]
    
    # Add a predict button
    if st.button("Predict Disease"):
        if not formatted_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            try:
                with st.spinner("Analyzing symptoms..."):
                    predictions, symptom_details = predict_disease(formatted_symptoms)
                
                # Display prediction results
                st.subheader("Prediction Results")
                
                # Create three columns for top predictions
                cols = st.columns(min(3, len(predictions)))
                
                for i, (col, prediction) in enumerate(zip(cols, predictions)):
                    with col:
                        disease = prediction['disease']
                        confidence = prediction['confidence']
                        details = prediction['details']
                        
                        # Create a colored box based on confidence
                        color = "green" if confidence > 80 else "orange" if confidence > 50 else "red"
                        st.markdown(f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: {color}30;">
                            <h3>{disease.replace('_', ' ').title()}</h3>
                            <h4>Confidence: {confidence}%</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("View Details"):
                            st.markdown(f"**Description**: {details['description']}")
                            
                            st.markdown("**Precautions**:")
                            for precaution in details['precautions']:
                                st.markdown(f"- {precaution}")
                            
                            st.markdown("**Treatment**:")
                            for treatment in details['treatment']:
                                st.markdown(f"- {treatment}")
                            
                            st.markdown(f"**Severity**: {details['severity']}")
                            st.markdown(f"**Recovery Time**: {details['recovery_time']}")
                
                # Display symptom severity information
                st.subheader("Symptom Severity Analysis")
                severity_colors = {
                    'high': 'red',
                    'medium': 'orange',
                    'low': 'green',
                    'unknown': 'gray'
                }
                
                for symptom, severity in symptom_details.items():
                    color = severity_colors.get(severity, 'gray')
                    st.markdown(f"""
                    <div style="padding: 10px; margin: 5px; border-radius: 5px; background-color: {color}30;">
                        <b>{symptom.replace('_', ' ').title()}</b>: {severity.title()} severity
                    </div>
                    """, unsafe_allow_html=True)
                
                st.warning("⚠️ This is not a substitute for professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment.")
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                logger.error(f"Prediction error: {str(e)}")
                logger.error(traceback.format_exc())

def xray_analysis():
    st.header("X-ray Image Analysis")
    st.write("Upload a chest X-ray image for pneumonia detection.")
    
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image_col, result_col = st.columns(2)
            
            with image_col:
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded X-ray Image", use_column_width=True)
            
            # Add a button to analyze the image
            if st.button("Analyze X-ray"):
                with st.spinner("Analyzing X-ray..."):
                    # Save image to file
                    filename = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                    img.save(filename)
                    
                    # Process and predict
                    processed_img = preprocess_image(img)
                    prediction = xray_model.predict(processed_img)
                    predicted_prob = float(prediction[0][0])
                    predicted_class = 'PNEUMONIA' if predicted_prob >= 0.5 else 'NORMAL'
                    confidence = predicted_prob if predicted_class == 'PNEUMONIA' else 1 - predicted_prob
                    confidence_percentage = round(float(confidence * 100), 2)
                
                # Display results
                with result_col:
                    st.subheader("Analysis Results")
                    
                    # Color based on predicted class
                    result_color = "red" if predicted_class == 'PNEUMONIA' else "green"
                    
                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: {result_color}30;">
                        <h2>Prediction: {predicted_class}</h2>
                        <h3>Confidence: {confidence_percentage}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if predicted_class == 'PNEUMONIA':
                        st.warning("""
                        **Pneumonia Detected**
                        
                        Please consult a healthcare professional for proper diagnosis and treatment.
                        Common symptoms of pneumonia include:
                        - Chest pain when breathing or coughing
                        - Confusion or changes in mental awareness (in adults age 65 and older)
                        - Cough, which may produce phlegm
                        - Fatigue
                        - Fever, sweating and shaking chills
                        - Lower than normal body temperature (in adults older than age 65 and people with weak immune systems)
                        - Nausea, vomiting or diarrhea
                        - Shortness of breath
                        """)
                    else:
                        st.success("No signs of pneumonia detected in the X-ray image.")
                    
                    st.warning("⚠️ This is not a substitute for professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment.")
        
        except Exception as e:
            st.error(f"An error occurred during image analysis: {str(e)}")
            logger.error(f"X-ray analysis error: {str(e)}")
            logger.error(traceback.format_exc())

if __name__ == '__main__':
    main()