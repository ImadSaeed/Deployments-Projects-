```python
"""
PlantVillage Disease Classifier - Streamlit App
Uses Tiny NN trained on EfficientNetV2B0 features
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

# ============================================
# FIX: Get absolute path to model file
# ============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "tiny_nn_final.h5")

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="\U0001F33F",  # Unicode for Herb
    layout="wide"
)

# ============================================
# CACHE MODELS (Load once at startup)
# ============================================
@st.cache_resource
def load_models():
    """Load EfficientNet feature extractor and Tiny NN model"""
    
    # Load feature extractor (EfficientNetV2B0 - frozen)
    feature_extractor = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        pooling='avg',  # Outputs 1280-dim features
        weights='imagenet'
    )
    feature_extractor.trainable = False
    
    # Load your trained Tiny NN model using absolute path
    tiny_nn = tf.keras.models.load_model(MODEL_PATH)
    
    return feature_extractor, tiny_nn

# ============================================
# CLASS NAMES (38 classes from PlantVillage)
# ============================================
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ============================================
# PREPROCESSING FUNCTION (CLAHE + Resize)
# ============================================
def preprocess_image(uploaded_file, img_size=224):
    """
    Apply CLAHE preprocessing (matches training pipeline)
    """
    # Read image
    image = Image.open(uploaded_file)
    
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    img = np.array(image)
    
    # Apply CLAHE in LAB space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge((l_enhanced, a, b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Resize to 224x224
    img = cv2.resize(img, (img_size, img_size))
    
    # Normalize to [0, 1]
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

# ============================================
# MAIN APP UI
# ============================================
st.title("\U0001F33F Plant Disease Classification")
st.markdown("""
    **98% Accuracy** | Tiny Neural Network + EfficientNetV2B0 Features
    
    Upload a leaf image to identify the plant disease or confirm it's healthy.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("\U0001F4E4 Upload Leaf Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a clear photo of a plant leaf"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf", use_container_width=True)

with col2:
    st.subheader("\U0001F52C Diagnosis Result")
    
    if uploaded_file is not None:
        with st.spinner("Analyzing leaf image..."):
            # Load models (cached after first load)
            feature_extractor, tiny_nn = load_models()
            
            # Preprocess image
            processed_img = preprocess_image(uploaded_file)
            
            # Extract 1280-dim features using EfficientNet
            features = feature_extractor.predict(processed_img, verbose=0)
            
            # Predict using Tiny NN
            predictions = tiny_nn.predict(features, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx] * 100
            
            predicted_class = CLASS_NAMES[predicted_idx]
        
        # Display results
        if confidence > 80:
            st.success(f"### \U0001F3AF {predicted_class}")
            st.metric("Confidence", f"{confidence:.2f}%")
        elif confidence > 60:
            st.warning(f"### \U000026A0 {predicted_class}")
            st.metric("Confidence", f"{confidence:.2f}%")
        else:
            st.error("### \U0000274C Low confidence prediction")
            st.markdown(f"Confidence: {confidence:.1f}%")
        
        # Show top 3 predictions
        with st.expander("\U0001F4CA Top 3 Predictions"):
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            for idx in top_3_idx:
                class_name = CLASS_NAMES[idx]
                prob = predictions[0][idx] * 100
                st.progress(int(prob), text=f"{class_name}: {prob:.1f}%")
    
    else:
        st.info("\U0001F448 Upload a leaf image to see diagnosis")

st.divider()
st.markdown("""
    <div style="text-align: center; color: gray;">
        <small>
        \U0001F331 PlantVillage Disease Classifier | Model Accuracy: 98% | 38 Classes<br>
        Preprocessing: CLAHE in LAB space | Feature extraction: EfficientNetV2B0
        </small>
    </div>
    """, unsafe_allow_html=True)
```
