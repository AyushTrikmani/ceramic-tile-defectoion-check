import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import time

# Configure app
st.set_page_config(
    page_title="Tile Defect Inspector",
    page_icon="🔍",
    layout="centered"
)

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('keras_model.h5')
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

@st.cache_resource
def load_labels():
    try:
        with open('labels.txt') as f:
            return [x.strip() for x in f.readlines()]
    except Exception as e:
        st.error(f"Label loading failed: {str(e)}")
        st.stop()

def preprocess_image(image):
    try:
        img = np.array(image)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        img = cv2.resize(img, (224, 224))
        img = (img.astype(np.float32) / 127.5) - 1
        return np.expand_dims(img, axis=0)
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

def main():
    st.title("🔍 Tile Defect Inspector")
    st.write("Upload a ceramic tile image for defect detection")
    
    model = load_model()
    labels = load_labels()
    
    uploaded = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded:
        try:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image", width=300)
            
            with st.spinner("Analyzing..."):
                processed = preprocess_image(img)
                if processed is None:
                    return
                
                start = time.time()
                pred = model.predict(processed)
                elapsed = time.time() - start
                
                result = np.argmax(pred[0])
                confidence = pred[0][result]
                
                st.subheader("Results")
                if result == 0:
                    st.success(f"✅ {labels[result]} ({(confidence*100):.1f}% confidence)")
                else:
                    st.error(f"❌ {labels[result]} ({(confidence*100):.1f}% confidence)")
                
                st.progress(float(confidence))
                st.caption(f"Analysis took {elapsed:.2f} seconds")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
