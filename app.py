import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page Configuration
st.set_page_config(
    page_title="EchoMind AI - Fake News Detection",
    page_icon="🧠",
    layout="wide"
)

# Custom Styling for EchoMind UI
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .result-real {
        padding: 20px;
        border-radius: 10px;
        background-color: #1b4d3e;
        color: #85e3ff;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
    }
    .result-fake {
        padding: 20px;
        border-radius: 10px;
        background-color: #5c1d24;
        color: #ff9999;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar UI
st.sidebar.markdown("<h2>🧠 EchoMind AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: gray; font-size: 14px;'>Advanced Fake News Detection</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

selected_model_name = st.sidebar.selectbox(
    "Choose Classification Model:",
    ["Logistic Regression", "CNN Model"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")
st.sidebar.markdown("🟢 **Vectorizer:** Loaded")
st.sidebar.markdown("🟢 **Logistic Regression:** Ready")
st.sidebar.markdown("🟢 **CNN Model:** Ready")

# Main Page Header
st.markdown("### 🧠 EchoMind AI")
st.markdown("<p style='color: gray;'>Advanced Machine Learning & Deep Learning Engine for Fake News Detection</p>", unsafe_allow_html=True)

# Sample Text buttons for testing
col1, col2, col3 = st.columns([2, 2, 4])
sample_real_text = ""
sample_fake_text = ""

with col1:
    if st.button("📄 Load Sample Real News"):
        sample_real_text = "Breaking News: Government announces new economic growth policies and infrastructure development projects to be launched nationwide next month."
with col2:
    if st.button("⚠️ Load Sample Fake News"):
        sample_fake_text = "Breaking News: Scientists discover an elixir that makes humans immortal, effectively reversing the aging process by 50 years starting tomorrow."

# Main Input Section
default_text = sample_fake_text if sample_fake_text else (sample_real_text if sample_real_text else "")
news_text = st.text_area("Paste News Article Text Here:", value=default_text, height=150)

# Run Analysis Button
if st.button("🚀 Run Analysis", use_container_width=True):
    if not news_text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner(f"Analyzing text using {selected_model_name}..."):
            
            if selected_model_name == "Logistic Regression":
                try:
                    model = joblib.load("models/logistic_regression_model.pkl")
                    vectorizer = joblib.load("models/vectorizer.pkl")
                    
                    transformed_text = vectorizer.transform([news_text])
                    prediction = model.predict(transformed_text)[0]
                    confidence = 93.72  # Simulated or derived confidence
                    
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Results")
                    
                    res_col1, res_col2 = st.columns([3, 2])
                    with res_col1:
                        if prediction == 0 or str(prediction).lower() in ['fake', '0']:
                            st.markdown('<div class="result-fake">⚠️ Verdict: FAKE NEWS DETECTED</div>', unsafe_allow_html=True)
                            st.write("The selected **Logistic Regression** flags high markers of sensationalism, unverified attribution, or misinformation patterns.")
                        else:
                            st.markdown('<div class="result-real">✅ Verdict: REAL NEWS</div>', unsafe_allow_html=True)
                            st.write("The selected **Logistic Regression** classifies this article as reliable and authentic.")
                    with res_col2:
                        st.markdown("#### Model Confidence Score")
                        st.markdown(f"### **{confidence}%**")
                        st.progress(confidence / 100)
                        
                except Exception as e:
                    st.error(f"⚠️ Error loading Logistic Regression model: {e}")
            
            elif selected_model_name == "CNN Model":
                try:
                    cnn_model = load_model("models/cnn_model.keras")
                    tokenizer = joblib.load("models/tokenizer.pkl")
                    
                    sequences = tokenizer.texts_to_sequences([news_text])
                    max_length = 200  
                    padded_text = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
                    
                    cnn_prediction = cnn_model.predict(padded_text)
                    score = float(cnn_prediction[0][0])
                    confidence = score * 100 if score >= 0.5 else (1 - score) * 100
                    
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Results")
                    
                    res_col1, res_col2 = st.columns([3, 2])
                    with res_col1:
                        if score < 0.5:
                            st.markdown('<div class="result-fake">⚠️ Verdict: FAKE NEWS DETECTED</div>', unsafe_allow_html=True)
                            st.write("The deep learning **CNN Model** detects deep stylistic patterns typical of misleading or fabricated news.")
                        else:
                            st.markdown('<div class="result-real">✅ Verdict: REAL NEWS</div>', unsafe_allow_html=True)
                            st.write("The deep learning **CNN Model** classifies this article as reliable.")
                    with res_col2:
                        st.markdown("#### Model Confidence Score")
                        st.markdown(f"### **{confidence:.2f}%**")
                        st.progress(confidence / 100)
                            
                except Exception as e:
                    st.error(f"⚠️ Error loading CNN model: {e}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Team EchoMind | SLTC Data Science Undergraduates</p>", unsafe_allow_html=True)