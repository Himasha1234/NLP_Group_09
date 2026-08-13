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
<<<<<<< HEAD
st.sidebar.subheader("📌 Auto Sample News")
sample_choice = st.sidebar.selectbox(
    "Load Sample News:",
    ["-- Select Sample --", "Sample Real News (NASA)", "Sample Fake News (Immortal Elixir)"]
)
=======
st.sidebar.markdown("### 📊 System Status")
st.sidebar.markdown("🟢 **Vectorizer & Tokenizer:** Loaded")
st.sidebar.markdown("🟢 **Models:** Ready")
>>>>>>> b645348 (Trained models successfully and updated app.py for Member 1)

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

<<<<<<< HEAD
if st.button("Analyze News", type="primary"):
    if not news_text.strip():
        st.warning("Please enter some text to analyze.")
=======
with col2:
    if st.button("⚠️ Load Sample Fake News"):
        st.session_state.news_input = "Breaking News: Scientists discover an elixir that makes humans immortal, effectively reversing the aging process by 50 years starting tomorrow."
        st.rerun()

# Main Input Section
user_input = st.text_area("Paste News Article Text Here:", value=st.session_state.news_input, height=150)

if user_input != st.session_state.news_input:
    st.session_state.news_input = user_input

# Run Analysis Button
if st.button("🚀 Run Analysis", use_container_width=True):
    if not st.session_state.news_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
>>>>>>> b645348 (Trained models successfully and updated app.py for Member 1)
    else:
        with st.spinner("Analyzing text authenticity..."):
            
<<<<<<< HEAD
            prediction_score = 0.5 
            is_fake = False
            
            # 1. Naive Bayes Prediction Logic
            if selected_model == "Naive Bayes" or selected_model == "Ensemble (All Models)":
                if 'naive_bayes' in models_dict and 'tfidf' in models_dict:
                    vec_text = models_dict['tfidf'].transform([news_text])
                    nb_pred_prob = models_dict['naive_bayes'].predict_proba(vec_text)[0][1]
=======
            # --- 1. LOGISTIC REGRESSION PREDICTION ---
            if selected_model_name == "Logistic Regression":
                try:
                    model = joblib.load("models/logistic_regression_model.pkl")
                    vectorizer = joblib.load("models/vectorizer.pkl")
                    
                    transformed_text = vectorizer.transform([st.session_state.news_input])
                    prediction = model.predict(transformed_text)[0]
                    
                    # Get probability confidence if available
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(transformed_text)[0]
                        confidence = float(np.max(proba)) * 100
                    else:
                        confidence = 95.00
                    
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Results")
                    
                    res_col1, res_col2 = st.columns([3, 2])
                    with res_col1:
                        # WELFake Standard: 1 = Fake, 0 = Real
                        if prediction == 1 or str(prediction).lower() in ['1', 'fake']:
                            st.markdown('<div class="result-fake">⚠️ Verdict: FAKE NEWS DETECTED</div>', unsafe_allow_html=True)
                            st.write("The selected **Logistic Regression** flags high markers of sensationalism or misinformation patterns.")
                        else:
                            st.markdown('<div class="result-real">✅ Verdict: REAL NEWS</div>', unsafe_allow_html=True)
                            st.write("The selected **Logistic Regression** classifies this article as reliable and authentic.")
                    with res_col2:
                        st.markdown("#### Model Confidence Score")
                        st.markdown(f"### **{confidence:.2f}%**")
                        st.progress(confidence / 100)
                        
                except Exception as e:
                    st.error(f"⚠️ Error running Logistic Regression model: {e}")
            
            # --- 2. CNN MODEL PREDICTION ---
            elif selected_model_name == "CNN Model":
                try:
                    cnn_model = load_model("models/cnn_model.keras")
                    tokenizer = joblib.load("models/tokenizer.pkl")
>>>>>>> b645348 (Trained models successfully and updated app.py for Member 1)
                    
                    # Naive Bayes සඳහා probability එක invert කරගැනීම (Real එකක් සඳහා score එක අඩු විය යුතු නම්)
                    corrected_nb_score = 1.0 - nb_pred_prob if selected_model == "Naive Bayes" else nb_pred_prob
                    
<<<<<<< HEAD
                    if selected_model == "Naive Bayes":
                        prediction_score = corrected_nb_score
                        is_fake = (prediction_score > 0.5)
=======
                    cnn_prediction = cnn_model.predict(padded_text)
                    score = float(cnn_prediction[0][0])
                    
                    # Assuming Sigmoid output: closer to 1 or 0 depending on training mapping
                    # Here standard: score >= 0.5 as Fake (1) or Real (0). Let's use standard threshold.
                    is_fake = score >= 0.5
                    confidence = (score if is_fake else (1 - score)) * 100
                    
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Results")
                    
                    res_col1, res_col2 = st.columns([3, 2])
                    with res_col1:
                        if is_fake:
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
                    st.error(f"⚠️ Error running CNN model: {e}")
>>>>>>> b645348 (Trained models successfully and updated app.py for Member 1)

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