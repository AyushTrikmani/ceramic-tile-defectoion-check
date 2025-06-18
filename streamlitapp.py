import streamlit as st
from PIL import Image
import numpy as np
import keras
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
        # Load the model using keras (compatible with TensorFlow 2.12.0)
        model = keras.models.load_model('keras_model.h5', compile=False)
        
        # Load labels
        with open('labels.txt', 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        
        return model, labels
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to RGB if not already
        if len(img_array.shape) == 3 and img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        
        # Resize the image
        img_resized = cv2.resize(img_array, target_size)
        
        # Normalize the image (assuming model expects [-1, 1] range)
        img_normalized = (img_resized.astype(np.float32) / 127.5) - 1
        
        # Add batch dimension
        img_expanded = np.expand_dims(img_normalized, axis=0)
        
        return img_expanded
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def main():
    st.title("Ceramic Tiles Defect Detection")
    st.write("Upload an image of a ceramic tile to check for defects")
    
    # Load model and labels
    model, labels = load_model_and_labels()
    if model is None:
        st.error("Failed to load model. Please check the model file.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a ceramic tile"
    )
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess the image
            processed_image = preprocess_image(image)
            if processed_image is None:
                return
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                prediction = model.predict(processed_image)
            
            # Get the predicted class and confidence
            predicted_class_index = np.argmax(prediction[0])
            predicted_class = labels[predicted_class_index]
            confidence = prediction[0][predicted_class_index]
            
            # Display results
            st.subheader("Prediction Result")
            
            if predicted_class_index == 0:
                st.success(f"✅ Non-defected (Confidence: {confidence:.2%})")
            else:
                st.error(f"❌ Defected (Confidence: {confidence:.2%})")
            
            # Show confidence for both classes
            st.write("\nConfidence levels:")
            for i, label in enumerate(labels):
                st.write(f"{label}: {prediction[0][i]:.2%}")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
