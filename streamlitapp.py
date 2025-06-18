import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2


# Configure app
st.set_page_config(
    page_title="Tile Defect Detector",
    layout="wide",
    initial_sidebar_state="expanded"
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

def preprocess(img):
    img = np.array(img)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    img = cv2.resize(img, (224, 224))
    return (img.astype(np.float32) / 127.5) - 1

def main():
    st.title("🔍 Tile Defect Inspector")
    st.write("Upload a tile image for defect detection")
    
    model = load_model()
    labels = load_labels()
    
    uploaded = st.file_uploader("Choose image...", type=["jpg", "jpeg", "png"])
    
    if uploaded:
        try:
            img = Image.open(uploaded)
            st.image(img, caption="Your Tile", width=300)
            
            with st.spinner("Analyzing..."):
                start = time.time()
                processed = np.expand_dims(preprocess(img), axis=0)
                pred = model.predict(processed)
                time_taken = time.time() - start
                
            result = np.argmax(pred[0])
            confidence = pred[0][result]
            
            st.subheader("🔬 Results")
            if result == 0:
                st.success(f"✅ Non-Defected ({(confidence*100):.1f}% confidence)")
            else:
                st.error(f"❌ Defected ({(confidence*100):.1f}% confidence)")
            
            st.progress(float(confidence))
            st.caption(f"Analysis took {time_taken:.2f} seconds")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
