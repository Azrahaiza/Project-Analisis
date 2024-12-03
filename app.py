import streamlit as st
import pandas as pd
from app.data_processing import clean_text, remove_stopwords, apply_stemming
from app.model import train_model, evaluate_model
from app.visualization import plot_sentiment_distribution, plot_wordcloud

# Load data
st.title("Analisis Sentimen Aplikasi Info BMKG")
df = pd.read_csv('data/data_reviews_with_sentiment.csv')

# Preprocessing
st.header("Preprocessing Data")
df_clean = clean_text(df, 'content', 'text_clean')
df_clean['text_no_stopwords'] = df_clean['text_clean'].apply(remove_stopwords)
df_clean['text_stemmed'] = df_clean['text_no_stopwords'].apply(apply_stemming)
st.write(df_clean.head())

# Visualizations
st.header("Visualisasi Data")
sentiment_plot = plot_sentiment_distribution(df_clean)
st.pyplot(sentiment_plot)

all_text = " ".join(df_clean["text_stemmed"].dropna())
wordcloud_plot = plot_wordcloud(all_text)
st.pyplot(wordcloud_plot)

# Modeling
st.header("Training Model")
X = df_clean['text_stemmed']
y = df_clean['sentiment']
model, vectorizer = train_model(X, y)
st.success("Model trained successfully!")

# Evaluation
st.header("Evaluasi Model")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
report = evaluate_model(model, vectorizer, X_test, y_test)
st.text(report)
