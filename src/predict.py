import joblib
import re

# 1. Load Vectorizer and Model 
log_model = joblib.load('models/logistic_regression.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def predict_news(text):
    clean_text = preprocess_text(text)
    
    # Vectorization
    text_vec = vectorizer.transform([clean_text])
    prediction = log_model.predict(text_vec)
    
    # DEBUG: ප්‍රතිඵලය පෙන්වීම
    print(f"DEBUG: Model prediction array is {prediction}")
    
    # [1] යනු 'Fake News' ලෙස පද්ධතිය හඳුනාගන්නා බැවින් මෙය භාවිත කරන්න:
    if prediction[0] == 1:
        return "Fake News"
    else:
        return "Real News"