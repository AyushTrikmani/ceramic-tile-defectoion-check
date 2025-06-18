import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Set page config
st.set_page_config(
    page_title="Ceramic Tiles Defect Detector",
    page_icon="🖼️",
    layout="centered"
)

# Function to load the model and labels
@st.cache_resource
def load_model_and_labels():
    try:
        model = load_model('keras_model.h5', compile=False)
        with open('labels.txt', 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return model, labels
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, None

def preprocess_image(image, target_size=(224, 224)):
    try:
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        img_resized = cv2.resize(img_array, target_size)
        img_normalized = (img_resized.astype(np.float32) / 127.5) - 1
        return np.expand_dims(img_normalized, axis=0)
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

def main():
    st.title("Ceramic Tiles Defect Detection")
    
    model, labels = load_model_and_labels()
    if model is None:
        return

    uploaded_file = st.file_uploader(
        "Upload tile image", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
            processed_img = preprocess_image(image)
            if processed_img is None:
                return
                
            with st.spinner("Analyzing..."):
                pred = model.predict(processed_img)
                class_idx = np.argmax(pred[0])
                confidence = pred[0][class_idx]
                
            if class_idx == 0:
                st.success(f"✅ Non-defected ({confidence:.2%} confidence)")
            else:
                st.error(f"❌ Defected ({confidence:.2%} confidence)")
                
            st.progress(float(confidence))
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    main()
