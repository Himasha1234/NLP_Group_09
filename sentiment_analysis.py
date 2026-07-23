import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon for sentiment extraction
nltk.download('vader_lexicon')

print("1. Loading cleaned dataset...")
df = pd.read_csv('cleaned_WELFake_Dataset.csv')

print("2. Initializing VADER Sentiment Analyzer...")
sia = SentimentIntensityAnalyzer()

print("3. Extracting sentiment scores...")

def get_sentiment_scores(text):
    if not isinstance(text, str):
        return 0.0, 0.0, 0.0, 0.0
    scores = sia.polarity_scores(text)
    return scores['compound'], scores['pos'], scores['neu'], scores['neg']

sentiment_data = df['clean_text'].apply(get_sentiment_scores)

df['sentiment_compound'] = [s[0] for s in sentiment_data]
df['sentiment_pos'] = [s[1] for s in sentiment_data]
df['sentiment_neu'] = [s[2] for s in sentiment_data]
df['sentiment_neg'] = [s[3] for s in sentiment_data]

print("4. Sentiment Scores Sample:")
print(df[['clean_text', 'sentiment_compound', 'label']].head())

print("5. Saving dataset with sentiment features...")
df.to_csv('cleaned_WELFake_with_sentiment.csv', index=False)

print("\n Done! Output saved as 'cleaned_WELFake_with_sentiment.csv'")