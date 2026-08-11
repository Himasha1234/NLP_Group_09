import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page Configuration
st.set_page_config(
    page_title="EchoMind AI - Fake News Detection",
    page_icon="🤖",
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

# Load Models and Vectorizers (Caching for better performance)
@st.cache_resource
def load_all_models():
    models = {}
    try:
        # Load Machine Learning Models & Vectorizers
        models['naive_bayes'] = joblib.load('models/naive_bayes_model.pkl')
        models['tfidf'] = joblib.load('models/tfidf_vectorizer.pkl')
        
        # Load Deep Learning Models & Tokenizers
        models['lstm'] = load_model('models/lstm_model.keras')
        models['lstm_tokenizer'] = joblib.load('models/lstm_tokenizer.pkl')
        
        # Optional: Load CNN if available in models folder
        try:
            models['cnn'] = load_model('cnn_model.keras')
        except:
            pass
            
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

models_dict = load_all_models()

# App Header
st.title("🛡️ EchoMind AI: Fake News Detection System")
st.write("Enter a news article or headline below to verify its authenticity using our advanced AI models.")

# Sidebar for Model Selection
st.sidebar.header("Configuration")
selected_model = st.sidebar.selectbox(
    "Choose Detection Model",
    ["Ensemble (All Models)", "Naive Bayes", "LSTM Deep Learning", "CNN Model"]
)

# Main Input Section
news_text = st.text_area("News Content / Headline:", placeholder="Paste your news text here...", height=150)

if st.button("Analyze News", type="primary"):
    if not news_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text authenticity..."):
            
            prediction_score = 0.5 # Default neutral
            
            # 1. Naive Bayes Prediction
            if selected_model == "Naive Bayes" or selected_model == "Ensemble (All Models)":
                if 'naive_bayes' in models_dict and 'tfidf' in models_dict:
                    vec_text = models_dict['tfidf'].transform([news_text])
                    nb_pred = models_dict['naive_bayes'].predict_proba(vec_text)[0][1] # Probability of being fake/real based on training
                    prediction_score = nb_pred

            # 2. LSTM Prediction
            if selected_model == "LSTM Deep Learning" or selected_model == "Ensemble (All Models)":
                if 'lstm' in models_dict and 'lstm_tokenizer' in models_dict:
                    seq = models_dict['lstm_tokenizer'].texts_to_sequences([news_text])
                    padded = pad_sequences(seq, maxlen=200) # Adjust maxlen according to your training setup
                    lstm_pred = models_dict['lstm'].predict(padded)[0][0]
                    if selected_model == "LSTM Deep Learning":
                        prediction_score = lstm_pred

            # Display Results
            st.markdown("---")
            st.subheader("Analysis Results")
            
            # Threshold logic (Assuming > 0.5 is Fake or Real depending on label encoding)
            # Adjust the condition based on your dataset labels (e.g., 1 for Fake, 0 for Real)
            is_fake = prediction_score > 0.5 
            
            if is_fake:
                st.markdown(f'<div class="result-fake">⚠️ Warning: This news appears to be FAKE!</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-real">✅ This news appears to be REAL.</div>', unsafe_allow_html=True)
                
            st.info(f"Confidence Score / Probability: {prediction_score:.4f}")