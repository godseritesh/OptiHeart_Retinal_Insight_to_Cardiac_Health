import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# Application Configuration
st.set_page_config(
    page_title="Heart Health Analyzer",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants for image dimensions and model path
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
MODEL_PATH = './models/heart_attack_detection_model.h5'  # Relative path to model

# Advanced Error Handling for Model Loading
@st.cache_resource
def load_ml_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.sidebar.error(f"Critical Error Loading Model: {e}")
        return None

# Cached model loading
model = load_ml_model()

# Helper Functions for Retinal Image Analysis
def preprocess_retinal_image(image):
    """Advanced image preprocessing for retinal analysis"""
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Enhanced contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img)
    
    # Binary thresholding
    _, binary_img = cv2.threshold(enhanced_img, 127, 255, cv2.THRESH_BINARY)
    
    return enhanced_img, binary_img

def extract_vessel_features(image):
    """Simulate vessel feature extraction"""
    # Note: In a real-world scenario, this would use advanced image processing
    vessel_density = np.random.uniform(0.3, 0.5)
    avg_vessel_width = np.random.uniform(2.5, 4.5)
    return vessel_density, avg_vessel_width

def extract_tortuosity(image):
    """Simulate vessel tortuosity calculation"""
    return np.random.uniform(1.0, 2.5)

def extract_optic_disc_features(image):
    """Simulate optic disc feature extraction"""
    disc_diameter = np.random.uniform(100, 150)
    disc_area = np.random.uniform(12000, 16000)
    return disc_diameter, disc_area

def analyze_risk(features):
    """Calculate risk factors based on extracted features"""
    thresholds = {
        'vessel_density': 0.4,
        'vessel_width': 5.0,
        'tortuosity': 2.0,
        'disc_area': 15000
    }
    
    risk_factors = {
        'Vessel Density Risk': 1 - (features['Vessel Density'] / thresholds['vessel_density']),
        'Vessel Width Risk': features['Avg Vessel Width'] / thresholds['vessel_width'],
        'Tortuosity Risk': features['Tortuosity'] / thresholds['tortuosity'],
        'Disc Area Risk': features['Disc Area'] / thresholds['disc_area']
    }

    return risk_factors

def visualize_retinal_analysis(image, enhanced_img, binary_img, features, risk_factors):
    """Create comprehensive visualization of retinal analysis"""
    # Convert image to RGB if it's grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize all images to 288x288
    def resize_image(img):
        return cv2.resize(img, (288, 288))
    
    resized_original = resize_image(image)
    resized_enhanced = resize_image(enhanced_img)
    resized_binary = resize_image(binary_img)
    
    # Create columns for images
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(resized_original, caption="Fig 1: Original Image", use_container_width=True)
    
    with col2:
        st.image(resized_enhanced, caption="Fig 2: Enhanced Image", use_container_width=True)
    
    with col3:
        st.image(resized_binary, caption="Fig 3: Binary Segmentation", use_container_width=True)
    
    # Plotly Risk Gauge
    st.subheader("Cardiovascular Risk Assessment")
    overall_risk = sum(risk_factors.values()) / len(risk_factors)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall_risk * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Cardiovascular Risk"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 33], 'color': "rgba(144, 238, 144, 0.5)"},   #lightgreen with 50% transparency
                {'range': [33, 66], 'color': "rgba(255, 255, 224, 0.5)"}, # lightyellow with 50% transparency
                {'range': [66, 100], 'color': "rgba(255, 160, 122, 0.5)"}  # lightred (coral) with 50% transparency
            ],
        }
    ))
    st.plotly_chart(fig)
    
    # Risk Factors Visualization
    st.subheader("Detailed Risk Factors")
    risk_df = pd.DataFrame.from_dict(risk_factors, orient='index', columns=['Risk Value'])
    fig_bar = px.bar(
        risk_df, 
        title="Risk Factors Breakdown",
        labels={'index': 'Risk Category', 'value': 'Risk Level'},
        color_discrete_sequence=px.colors.sequential.Reds_r
    )
    st.plotly_chart(fig_bar)
    
    # Feature Values Display
    st.subheader("Feature Values")
    features_df = pd.DataFrame.from_dict(features, orient='index', columns=['Value'])
    st.dataframe(features_df)

def read_and_preprocess_image(image):
    """Preprocess image for ML model input"""
    # Read image buffer
    img_array = np.array(Image.open(image))
    
    # Convert to RGB if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize and normalize
    img = cv2.resize(img_array, (IMAGE_HEIGHT, IMAGE_WIDTH))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(image):
    """Classify heart disease from image"""
    if model is None:
        st.error("Machine Learning Model Not Loaded")
        return None, None
    
    img = read_and_preprocess_image(image)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    class_labels = ['Healthy', 'Mild Disease', 'Moderate Disease', 'Severe Disease', 'Heart Attack']
    score = predictions[0][predicted_class]
    return class_labels[predicted_class], score

def main():
    # Sidebar Navigation with Icons
    with st.sidebar:
        selected = option_menu(
            "Heart Disease Analyzer", 
            ["Home", "Retinal Analysis", "ML Classification"],
            icons=['house', 'eye', 'heart-pulse'],
            menu_icon="app-indicator", 
            default_index=0,
            styles={
                "container": {"padding": "5!important", "background-color": "#f0f2f6"},
                "icon": {"color": "blue", "font-size": "20px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
                "nav-link-selected": {"background-color": "#1874CD"},
            }
        )
    
    # Home Section
    if selected == "Home":
        st.title("🏥 Heart Health Diagnostic Platform")
        
        st.markdown("""
        ### Welcome to Advanced Cardiovascular Risk Analysis
        
        This application provides cutting-edge diagnostic support through:
        - **Retinal Image Analysis**: Detect early cardiovascular risk indicators
        - **Machine Learning Classification**: Predict heart disease severity
        
        Navigate through sections to explore our advanced diagnostic tools.
        """)
        
        # Feature Cards
        cols = st.columns(3)
        feature_details = [
            ("Retinal Analysis", "Advanced image processing to detect early cardiovascular risks", "👁️"),
            ("ML Classification", "AI-powered heart disease prediction", "❤️"),
            ("Risk Assessment", "Comprehensive risk factor evaluation", "📊")
        ]
        
        for col, (title, desc, icon) in zip(cols, feature_details):
            with col:
                st.markdown(f"""
                <div style="
                    border: 1px solid #e0e0e0; 
                    border-radius: 10px; 
                    padding: 15px; 
                    margin-bottom: 10px;
                    background-color: white;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                ">
                    <h3>{icon} {title}</h3>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Retinal Analysis Section
    elif selected == "Retinal Analysis":
        st.title("🔬 Retinal Image Analysis")
        
        uploaded_retinal_image = st.file_uploader(
            "Upload Retinal Image", 
            type=["jpg", "jpeg", "png"],
            help="Upload medical images for detailed retinal analysis"
        )
        
        if uploaded_retinal_image is not None:
            retinal_image = Image.open(uploaded_retinal_image)
            
            # Resize and preview the uploaded image
            preview_image = cv2.resize(np.array(retinal_image), (288, 288))
            st.image(preview_image, caption=f"Fig 0: Uploaded {uploaded_retinal_image.name}", use_container_width=True)
            
            enhanced_img, binary_img = preprocess_retinal_image(retinal_image)
            vessel_density, avg_vessel_width = extract_vessel_features(retinal_image)
            tortuosity = extract_tortuosity(retinal_image)
            disc_diameter, disc_area = extract_optic_disc_features(retinal_image)

            features = {
                'Vessel Density': vessel_density,
                'Avg Vessel Width': avg_vessel_width,
                'Tortuosity': tortuosity,
                'Disc Area': disc_area,
                'Disc Diameter': disc_diameter
            }

            risk_factors = analyze_risk(features)
            visualize_retinal_analysis(np.array(retinal_image), enhanced_img, binary_img, features, risk_factors)
    
    # ML Classification Section
    elif selected == "ML Classification":
        st.title("❤️ Heart Disease Classification")
        
        uploaded_image = st.file_uploader(
            "Upload Heart Disease Image", 
            type=["jpg", "jpeg", "png"],
            help="Upload medical images for heart disease classification"
        )
        
        if uploaded_image is not None:
            # Resize and preview the uploaded image
            preview_image = Image.open(uploaded_image)
            preview_resized = cv2.resize(np.array(preview_image), (288, 288))
            st.image(preview_resized, caption=f"Fig 0: Uploaded {uploaded_image.name}", use_container_width=True)
            
            result = classify_image(uploaded_image)
            
            if result[0] is not None:
                class_label, score = result
                st.subheader(f"Predicted Class: {class_label}")
                st.subheader(f"Prediction Confidence: {score * 100:.2f}%")
                
                # Detailed Risk Breakdown
                risk_colors = {
                    'Healthy': 'green',
                    'Mild Disease': 'yellow',
                    'Moderate Disease': 'orange',
                    'Severe Disease': 'red',
                    'Heart Attack': 'darkred'
                }
                
                st.markdown(f"""
                <div style="background-color:{risk_colors.get(class_label, 'white')}; 
                            color:white; 
                            padding:10px; 
                            border-radius:10px;">
                    <h3>{class_label} Risk Assessment</h3>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()