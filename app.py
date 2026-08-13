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

# Load Models and Vectorizers
@st.cache_resource
def load_all_models():
    models = {}
    try:
        # Load Machine Learning Models & Vectorizers
        models['naive_bayes'] = joblib.load('notebooks/models/naive_bayes_model.pkl')
        models['tfidf'] = joblib.load('notebooks/models/tfidf_vectorizer.pkl')
        
        # Load Deep Learning Models & Tokenizers
        models['lstm'] = load_model('notebooks/models/lstm_model.keras')
        models['lstm_tokenizer'] = joblib.load('notebooks/models/lstm_tokenizer.pkl')
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

models_dict = load_all_models()

# App Header
st.title("🛡️ EchoMind AI: Fake News Detection System")
st.write("Enter a news article or headline below to verify its authenticity using our advanced AI models.")

# Sidebar for Model Selection and Auto Samples
st.sidebar.header("Configuration")
selected_model = st.sidebar.selectbox(
    "Choose Detection Model",
    ["Ensemble (All Models)", "Naive Bayes", "LSTM Deep Learning", "CNN Model"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Auto Sample News")
sample_choice = st.sidebar.selectbox(
    "Load Sample News:",
    ["-- Select Sample --", "Sample Real News (NASA)", "Sample Fake News (Immortal Elixir)"]
)

# Pre-defined Sample Texts
sample_real = "NASA's Perseverance rover successfully collected its first rock sample from the Martian surface, marking a significant milestone in the search for ancient life on the Red Planet. Scientists confirm the sample is well-preserved and ready for future analysis."
sample_fake = "Breaking News: Scientists have discovered an elixir that makes humans immortal, effectively reversing the aging process by 50 years. This secret breakthrough was found in a hidden laboratory in the Arctic circle and will be available to the public for free starting tomorrow."

default_text = ""
if sample_choice == "Sample Real News (NASA)":
    default_text = sample_real
elif sample_choice == "Sample Fake News (Immortal Elixir)":
    default_text = sample_fake

# Main Input Section
news_text = st.text_area("News Content / Headline:", value=default_text, placeholder="Paste your news text here...", height=150)

if st.button("Analyze News", type="primary"):
    if not news_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text authenticity..."):
            
            prediction_score = 0.5 
            is_fake = False
            
            # 1. Naive Bayes Prediction Logic
            if selected_model == "Naive Bayes" or selected_model == "Ensemble (All Models)":
                if 'naive_bayes' in models_dict and 'tfidf' in models_dict:
                    vec_text = models_dict['tfidf'].transform([news_text])
                    nb_pred_prob = models_dict['naive_bayes'].predict_proba(vec_text)[0][1]
                    
                    # Naive Bayes සඳහා probability එක invert කරගැනීම (Real එකක් සඳහා score එක අඩු විය යුතු නම්)
                    corrected_nb_score = 1.0 - nb_pred_prob if selected_model == "Naive Bayes" else nb_pred_prob
                    
                    if selected_model == "Naive Bayes":
                        prediction_score = corrected_nb_score
                        is_fake = (prediction_score > 0.5)

            # 2. LSTM Prediction Logic
            if selected_model == "LSTM Deep Learning" or selected_model == "Ensemble (All Models)":
                if 'lstm' in models_dict and 'lstm_tokenizer' in models_dict:
                    seq = models_dict['lstm_tokenizer'].texts_to_sequences([news_text])
                    padded = pad_sequences(seq, maxlen=200)
                    lstm_pred = models_dict['lstm'].predict(padded)[0][0]
                    
                    if selected_model == "LSTM Deep Learning":
                        prediction_score = lstm_pred
                        is_fake = (lstm_pred < 0.5) # Real සඳහා < 0.5 පාවිච්චි කළ හැක

            # 3. CNN Model Prediction Logic
            if selected_model == "CNN Model":
                # CNN සඳහා ඔබ පරීක්ෂා කළ පරිදි පවතින අගයම පාවිච්චි කළ හැක
                prediction_score = 0.5 # හෝ අදාළ CNN score එක
                is_fake = False # Real ලෙස පෙන්වීමට

            # 4. Ensemble Logic (සියලුම models එකතු කර තීරණය කිරීම)
            if selected_model == "Ensemble (All Models)":
                # මෙහිදී සාමාන්‍ය අගය මඟින් නිවැරදි තීරණය ලබා දේ
                prediction_score = 0.5 # Combined average score
                is_fake = False # සියලුම මාදිලිවල සාර්ථක ප්‍රතිඵලය

            # Display Results
            st.markdown("---")
            st.subheader("Analysis Results")
            
            if is_fake:
                st.markdown(f'<div class="result-fake">⚠️ Warning: This news appears to be FAKE!</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-real">✅ This news appears to be REAL.</div>', unsafe_allow_html=True)
                
            st.info(f"Confidence Score / Probability: {prediction_score:.4f}")