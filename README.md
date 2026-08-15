# 📰 Fake News Detection System

An advanced Natural Language Processing (NLP) project designed to detect and classify fake news articles. This system leverages a combination of traditional Machine Learning, Deep Learning (CNN/LSTM), and Transformer-based models (BERT) to ensure high accuracy in misinformation detection.

---

## 🚀 Features
* **Data Preprocessing:** Robust cleaning pipeline to handle noise, special characters, and text normalization.
* **Exploratory Data Analysis (EDA):** Visual insights into linguistic patterns, including word clouds, class distributions, and n-gram analysis.
* **Diverse Model Architecture:** 
    * **Classical ML:** Logistic Regression, Support Vector Machine (SVM).
    * **Deep Learning:** CNN and LSTM-based architectures.
    * **State-of-the-Art:** BERT (Bidirectional Encoder Representations from Transformers).
* **Performance Evaluation:** Comprehensive model comparison based on Accuracy, Precision, Recall, and F1-Score.
* **Interactive Interface:** A user-friendly web application (`app.py`) for real-time fake news prediction.

---

## 📂 Project Directory Structure

```text
Fake_News_Detection/
│
├── data/                 # Visualizations, word clouds & data analysis charts
├── models/               # Trained .pkl models, tokenizers, and model results
├── notebooks/            # Jupyter notebooks for EDA and advanced model experiments
├── app.py                # Streamlit/Web application interface
├── bert_tokenization.py  # BERT-specific tokenization scripts
├── data_cleaning.py      # Data cleaning and preprocessing pipeline
├── ensemble_predict.py   # Ensemble prediction logic
├── requirements.txt      # Python dependencies
└── ...

🛠️ Installation & Setup
1. Clone the repository:
git clone [https://github.com/Himasha1234/NLP_Group_09.git](https://github.com/Himasha1234/NLP_Group_09.git)
cd Fake_News_Detection

2. Install dependencies:
pip install -r requirements.txt

3. Run the Application:
streamlit run app.py

📊 Model Performance Summary
The models were evaluated on the provided dataset, achieving superior performance with fine-tuned SVM and BERT architectures. Detailed comparison graphs and confusion matrices are available in the data/ and models/ directories.

👥 Contributors
Himasha Edirisinghe
Ushan Umayanga
Chamodya

BSc in Data Science - SLTC Research University (NLP Group 09)

📄 License
This project is for academic purposes under the SLTC Research University NLP curriculum.