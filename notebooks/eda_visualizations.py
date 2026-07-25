import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# 1. Load Cleaned Data
df = pd.read_csv('data/WELFake_cleaned.csv')

# -----------------------------------------------------------------------------
# TASK 1: Class Distribution
# -----------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label')
plt.title('Distribution of Fake (1) vs Real (0) News')
plt.xlabel('Label (0 = Real, 1 = Fake)')
plt.ylabel('Count')
plt.savefig('data/class_distribution.png')
plt.show()

# -----------------------------------------------------------------------------
# TASK 2: Word Clouds (Fake vs. Real)
# -----------------------------------------------------------------------------
fake_text = " ".join(df[df['label'] == 1]['text_clean_ml'].dropna())
real_text = " ".join(df[df['label'] == 0]['text_clean_ml'].dropna())

# WordCloud for Fake News
wc_fake = WordCloud(width=800, height=400, background_color='black').generate(fake_text)
plt.figure(figsize=(10, 5))
plt.imshow(wc_fake, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Fake News')
plt.savefig('data/wordcloud_fake.png')
plt.show()

# WordCloud for Real News
wc_real = WordCloud(width=800, height=400, background_color='white').generate(real_text)
plt.figure(figsize=(10, 5))
plt.imshow(wc_real, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Real News')
plt.savefig('data/wordcloud_real.png')
plt.show()

# -----------------------------------------------------------------------------
# TASK 3: N-gram Analysis (Top Bigrams)
# -----------------------------------------------------------------------------
def plot_top_ngrams(text_series, title, n=2, top_k=10):
    vec = CountVectorizer(ngram_range=(n, n)).fit(text_series.dropna())
    bag_of_words = vec.transform(text_series.dropna())
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_k]
    
    ngram_df = pd.DataFrame(words_freq, columns=['Ngram', 'Count'])
    
    plt.figure(figsize=(8, 4))
    sns.barplot(x='Count', y='Ngram', data=ngram_df, palette='viridis')
    plt.title(title)
    plt.savefig(f'data/{title.lower().replace(" ", "_")}.png')
    plt.show()

# Plot Top 10 Bigrams (2-word phrases) for Fake vs Real
plot_top_ngrams(df[df['label'] == 1]['text_clean_ml'], 'Top 10 Bigrams in Fake News', n=2)
plot_top_ngrams(df[df['label'] == 0]['text_clean_ml'], 'Top 10 Bigrams in Real News', n=2)