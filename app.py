import os
import numpy as np
import tensorflow as tf
import cv2
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Constants for image dimensions and model path
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
MODEL_PATH = './models/heart_attack_detection_model.h5'  # Relative path to model

# Load model with error handling
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.sidebar.success("Heart Disease Model loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# Helper functions for Retinal Image Analysis
def preprocess_retinal_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img)
    _, binary_img = cv2.threshold(enhanced_img, 127, 255, cv2.THRESH_BINARY)
    return enhanced_img, binary_img

def extract_vessel_features(image):
    vessel_density = 0.35  # Placeholder value
    avg_vessel_width = 3.2  # Placeholder value
    return vessel_density, avg_vessel_width

def extract_tortuosity(image):
    tortuosity = 1.5  # Placeholder value
    return tortuosity

def extract_optic_disc_features(image):
    disc_diameter = 120  # Placeholder value
    disc_area = 14000  # Placeholder value
    return disc_diameter, disc_area

def analyze_risk(features):
    thresholds = {
        'vessel_density': 0.4,
        'vessel_width': 5.0,
        'tortuosity': 100,
        'disc_area': 15000
    }
    
    risk_factors = {
        'vessel_density_risk': 1 - (features['Vessel Density'] / thresholds['vessel_density']),
        'vessel_width_risk': features['Avg Vessel Width'] / thresholds['vessel_width'],
        'tortuosity_risk': features['Tortuosity'] / thresholds['tortuosity'],
        'disc_risk': features['Disc Area'] / thresholds['disc_area']
    }

    return risk_factors

def visualize_retinal_analysis(image, enhanced_img, binary_img, features, risk_factors):
    st.image(image, caption="Original Image", use_column_width=True)
    st.image(enhanced_img, caption="Enhanced Image (Contrast)", use_column_width=True)
    st.image(binary_img, caption="Binary Image (Vessel Segmentation)", use_column_width=True)
    
    st.subheader("Retinal Feature Metrics")
    feature_names = list(features.keys())
    feature_values = list(features.values())

    fig, ax = plt.subplots()
    sns.barplot(x=feature_values, y=feature_names, ax=ax, palette="Blues_d")
    ax.set_xlabel('Feature Value')
    ax.set_title('Extracted Retinal Features')
    st.pyplot(fig)

    st.subheader("Risk Factors Analysis")
    risk_factor_names = list(risk_factors.keys())
    risk_factor_values = list(risk_factors.values())

    fig2, ax2 = plt.subplots()
    sns.barplot(x=risk_factor_values, y=risk_factor_names, ax=ax2, palette="Reds_d")
    ax2.set_xlabel('Risk Factor Value')
    ax2.set_title('Risk Factors for Retinal Health')
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.hist(feature_values, bins=10, color='lightblue', edgecolor='black')
    ax3.set_title('Distribution of Extracted Features')
    ax3.set_xlabel('Feature Value')
    ax3.set_ylabel('Frequency')
    st.pyplot(fig3)

# Helper functions for Heart Disease Classification
def read_and_preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(image):
    img = read_and_preprocess_image(image)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    class_labels = ['Healthy', 'Mild Disease', 'Moderate Disease', 'Severe Disease', 'Heart Attack']
    score = predictions[0][predicted_class]
    return class_labels[predicted_class], score

# Streamlit Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choose a section:", ["Home", "Static Analysis", "ML Analysis"])

# Button for toggling light and dark mode
light_mode_button = st.sidebar.button("Switch to Light Mode")

# Set theme based on button press
if light_mode_button:
    st.markdown(
        """
        <style>
        body {background-color: white; color: black;}
        .sidebar .sidebar-content {background-color: #f4f4f4;}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Home Section
if section == "Home":
    st.title("Welcome to the Retinal and Heart Disease Analysis App")
    st.write(
        """
        This app provides advanced analysis for:
        - **Retinal image analysis** for early signs of heart disease.
        - **Image-based heart disease classification** using machine learning models.
        
        Navigate to the desired section using the sidebar to explore the functionalities.
        """
    )

# Static Analysis Section
elif section == "Static Analysis":
    st.title("Static Analysis: Retinal Image")
    uploaded_retinal_image = st.file_uploader("Upload a retinal image", type=["jpg", "jpeg", "png"])

    if uploaded_retinal_image:
        retinal_image = Image.open(uploaded_retinal_image)
        st.image(retinal_image, caption="Uploaded Retinal Image", use_column_width=True)
        
        enhanced_img, binary_img = preprocess_retinal_image(retinal_image)
        vessel_density, avg_vessel_width = extract_vessel_features(retinal_image)
        tortuosity = extract_tortuosity(retinal_image)
        disc_diameter, disc_area = extract_optic_disc_features(retinal_image)

        features = {
            'Vessel Density': vessel_density,
            'Avg Vessel Width': avg_vessel_width,
            'Tortuosity': tortuosity,
            'Disc Area': disc_area
        }

        risk_factors = analyze_risk(features)
        visualize_retinal_analysis(retinal_image, enhanced_img, binary_img, features, risk_factors)

# ML Analysis Section
elif section == "ML Analysis":
    st.title("Heart Disease Classification")
    uploaded_image = st.file_uploader("Upload a heart disease image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        class_label, score = classify_image(uploaded_image)
        st.subheader(f"Predicted Class: {class_label}")
        st.subheader(f"Prediction Confidence: {score * 100:.2f}%")
