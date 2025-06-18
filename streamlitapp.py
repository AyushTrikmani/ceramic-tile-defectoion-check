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
    layout="centered",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model('keras_model.h5')
        with open('labels.txt') as f:
            labels = [x.strip() for x in f.readlines()]
        return model, labels
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
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
    st.title("🔍 Ceramic Tile Defect Inspector")
    st.write("Upload an image to detect manufacturing defects")
    
    model, labels = load_assets()
    
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
                
                st.subheader("🔬 Results")
                col1, col2 = st.columns(2)
                with col1:
                    if result == 0:
                        st.success(f"✅ {labels[result]}")
                    else:
                        st.error(f"❌ {labels[result]}")
                with col2:
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                
                st.progress(float(confidence))
                st.caption(f"Analysis time: {elapsed:.2f}s")
                
        except Exception as e:
            st.error(f"Processing error: {str(e)}")

if __name__ == "__main__":
    main()
