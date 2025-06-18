import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import time

# Configure page
st.set_page_config(
    page_title="Ceramic Tiles Defect Detector",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('keras_model.h5')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

# Load labels
@st.cache_resource
def load_labels():
    try:
        with open('labels.txt', 'r') as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        st.error(f"Failed to load labels: {str(e)}")
        st.stop()

# Preprocess image
def preprocess_image(image, size=(224, 224)):
    try:
        img = np.array(image)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[..., :3]
        img = cv2.resize(img, size)
        img = (img.astype(np.float32) / 127.5) - 1
        return np.expand_dims(img, axis=0)
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

def main():
    st.title("Ceramic Tiles Defect Detection")
    st.write("Upload an image to check for defects")
    
    # Load resources
    model = load_model()
    labels = load_labels()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Maximum file size: 20MB"
    )
    
    if uploaded_file is not None:
        try:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process and predict
            with st.spinner("Analyzing image..."):
                processed_img = preprocess_image(image)
                if processed_img is None:
                    return
                
                start_time = time.time()
                prediction = model.predict(processed_img)
                elapsed = time.time() - start_time
                
                # Get results
                class_idx = np.argmax(prediction[0])
                confidence = prediction[0][class_idx]
                label = labels[class_idx]
                
                # Display results
                st.subheader("Results")
                if class_idx == 0:
                    st.success(f"✅ {label} (Confidence: {confidence:.2%})")
                else:
                    st.error(f"❌ {label} (Confidence: {confidence:.2%})")
                
                st.write(f"Processing time: {elapsed:.2f} seconds")
                
                # Confidence meter
                st.progress(float(confidence))
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
