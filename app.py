import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Page Configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

# Custom Styling
st.markdown("""
    <style>
    .result-real {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        color: #155724;
        text-align: center;
        font-weight: bold;
        font-size: 22px;
    }
    .result-fake {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8d7da;
        color: #721c24;
        text-align: center;
        font-weight: bold;
        font-size: 22px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Header
st.title("📰 Fake News Detection System")
st.markdown("Enter a news article below to check its authenticity using your trained **Logistic Regression** or **CNN** models.")

# Sidebar for Model Selection
st.sidebar.header("⚙️ Model Configuration")
selected_model_name = st.sidebar.selectbox(
    "Choose Trained Model:",
    ["Logistic Regression", "CNN Model"]
)

# Main Input Section
news_text = st.text_area("✍️ Paste News Article Here:", height=150, placeholder="Type or paste news content here...")

# Check Button
if st.button("🔍 Check Authenticity", use_container_width=True):
    if not news_text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner(f"Analyzing text using {selected_model_name}..."):
            
            # --- 1. Logistic Regression Prediction ---
            if selected_model_name == "Logistic Regression":
                try:
                    model = pickle.load(open("models/logistic_regression_model.pkl", "rb"))
                    vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
                    
                    # Transform and Predict
                    transformed_text = vectorizer.transform([news_text])
                    prediction = model.predict(transformed_text)[0]
                    
                    st.markdown("---")
                    st.subheader("📊 Prediction Result")
                    
                    # ඔබේ ඩේටාසෙට් එකේ ලේබල් වලට අනුව (0/1 හෝ Fake/Real) මෙහි ප්‍රතිඵලය පෙන්වනු ඇත
                    if prediction == 0 or str(prediction).lower() in ['fake', '0']:
                        st.markdown('<div class="result-fake">🚨 Prediction: FAKE NEWS</div>', unsafe_allow_html=True)
                        st.error("The trained Logistic Regression model classifies this article as **Fake/Misleading**.")
                    else:
                        st.markdown('<div class="result-real">✅ Prediction: REAL NEWS</div>', unsafe_allow_html=True)
                        st.success("The trained Logistic Regression model classifies this article as **Real/Reliable**.")
                        
                except Exception as e:
                    st.error(f"⚠️ Error loading Logistic Regression model: {e}")
                    st.info("💡 Please ensure `vectorizer.pkl` and `logistic_regression_model.pkl` are inside the 'models/' folder.")
            
            # --- 2. CNN Model Prediction ---
            elif selected_model_name == "CNN Model":
                try:
                    cnn_model = load_model("models/cnn_model.keras")
                    
                    # මෙතැනට CNN මෝඩලයට අදාළ preprocessing / tokenization කෝඩ් එක එකතු කළ යුතුය
                    # උදාහරණයක් ලෙස: padded_sequence = tokenizer.texts_to_sequences([news_text]) ...
                    
                    st.markdown("---")
                    st.subheader("📊 Prediction Result")
                    st.success("CNN model loaded successfully from `models/cnn_model.keras`!")
                    st.info("ℹ️ Connect your CNN text tokenization pipeline here to get the final Real/Fake output.")
                    
                except Exception as e:
                    st.error(f"⚠️ Error loading CNN model: {e}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2026 NLP Group Project | SLTC Data Science Undergraduates</p>", unsafe_allow_html=True)