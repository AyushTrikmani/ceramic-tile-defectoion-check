import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Tile Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .non-defective {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .defective {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained Keras model"""
    try:
        model = keras.models.load_model('keras_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    # Resize image to 224x224 (standard size for most teachable machine models)
    img = image.resize((224, 224))
    
    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Normalize pixel values to [0, 1] range
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_tile_defect(model, image):
    """Make prediction on the processed image"""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get the predicted class (0: Non defected, 1: Defected)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Class labels
        class_labels = {0: "Non Defected", 1: "Defected"}
        
        return class_labels[predicted_class], confidence, predictions[0]
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def main():
    # Main header
    st.markdown('<h1 class="main-header">🔍 Tile Defect Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📋 Instructions")
    st.sidebar.markdown("""
    1. Upload an image of a tile
    2. Wait for the model to process
    3. View the prediction results
    4. Check confidence scores
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.info("**Supported formats:** JPG, JPEG, PNG")
    
    # Load model
    with st.spinner("Loading AI model..."):
        model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please ensure 'keras_model.h5' is in the correct directory.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Tile Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a tile for defect detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add image info
            st.info(f"**Image Details:**\n- Size: {image.size}\n- Mode: {image.mode}\n- Format: {uploaded_file.type}")
    
    with col2:
        st.header("🔍 Prediction Results")
        
        if uploaded_file is not None:
            with st.spinner("Analyzing image..."):
                # Make prediction
                prediction, confidence, raw_predictions = predict_tile_defect(model, image)
                
                if prediction is not None:
                    # Display prediction result
                    if prediction == "Non Defected":
                        st.markdown(f"""
                        <div class="prediction-box non-defective">
                            ✅ {prediction}<br>
                            Confidence: {confidence:.2%}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box defective">
                            ❌ {prediction}<br>
                            Confidence: {confidence:.2%}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show detailed predictions
                    st.subheader("📊 Detailed Analysis")
                    
                    # Create a bar chart for predictions
                    pred_data = {
                        'Class': ['Non Defected', 'Defected'],
                        'Probability': [float(raw_predictions[0]), float(raw_predictions[1])]
                    }
                    
                    st.bar_chart(pred_data, x='Class', y='Probability')
                    
                    # Show raw probabilities
                    st.write("**Raw Prediction Scores:**")
                    st.write(f"- Non Defected: {raw_predictions[0]:.4f}")
                    st.write(f"- Defected: {raw_predictions[1]:.4f}")
                    
                else:
                    st.error("Failed to make prediction. Please try with a different image.")
        else:
            st.markdown("""
            <div class="info-box">
                <strong>📝 How it works:</strong><br>
                • Upload an image of a tile<br>
                • Our AI model analyzes the image<br>
                • Get instant defect detection results<br>
                • View confidence scores and detailed analysis
            </div>
            """, unsafe_allow_html=True)
    
    # Additional information section
    st.markdown("---")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Model Type", "TensorFlow Keras")
    
    with col4:
        st.metric("Classes", "2 (Defected/Non-Defected)")
    
    with col5:
        st.metric("Input Size", "224x224 pixels")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏗️ Tile Defect Detection System | Built with Streamlit & TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()