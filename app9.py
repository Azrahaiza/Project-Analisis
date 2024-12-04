import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from nltk.stem import SnowballStemmer
from imblearn.over_sampling import SMOTE

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Streamlit setup
st.set_page_config(page_title="Analisis Sentimen Penggunaan Aplikasi Info BMKG", page_icon="🌍", layout="wide")

# Custom Title and Description
st.markdown("""
<style>
.title {
    font-size: 48px;
    font-weight: bold;
    color: #87CEFA;
    text-align: center;
}
.description {
    font-size: 24px;
    color: #FFA500;
    text-align: center;
    margin-bottom: 20px;
}
.subheader {
    font-size: 18px;
    font-weight: bold;
    color: #4682B4;
}
.button {
    font-size: 14px;
    background-color: #2E8B57;
    color: green;
    padding: 10px 14px;
    border-radius: 5px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Custom Title and Description
st.markdown("""
<div class="title">Analisis Sentimen Penggunaan Aplikasi Info BMKG</div>
<div class="description">Visualisasi Pemrosesan Data</div>
""", unsafe_allow_html=True)

# Sidebar untuk mengunggah data dan navigasi
st.sidebar.header("Navigasi")
st.sidebar.write("Gunakan sidebar untuk mengunggah dataset dan memilih bagian yang ingin dilihat.")

# Unggah file CSV
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    f_busu = pd.read_csv(uploaded_file)

    # Penentuan sentimen berdasarkan score
    def assign_sentiment(row):
        if row['score'] >= 4:
            return 'Positif'
        elif row['score'] == 3:
            return 'Netral'
        else:
            return 'Negatif'

    f_busu['sentiment'] = f_busu.apply(assign_sentiment, axis=1)

    # Tampilan data
    st.subheader("Data Ulasan:")
    st.write(f_busu.head(10))

    # Tampilan data berdasarkan sentimen
    st.subheader("Data dengan Sentimen Positif:")
    st.write(f_busu[f_busu['sentiment'] == 'Positif'].head(10))
    st.subheader("Data dengan Sentimen Netral:")
    st.write(f_busu[f_busu['sentiment'] == 'Netral'].head(10))
    st.subheader("Data dengan Sentimen Negatif:")
    st.write(f_busu[f_busu['sentiment'] == 'Negatif'].head(10))

    # Pembersihan data
    def clean_data(df):
        df['text_clean'] = df['content'].str.lower()
        df['text_clean'] = df['text_clean'].apply(lambda x: re.sub(r"@[A-Za-z0-9]+|(\w+:\/\/\S+)|^rt|http\S*|[^\w\s]", "", x))
        df['text_clean'] = df['text_clean'].apply(lambda x: re.sub(r"\d+", "", x))
        return df

    f_busu_clean = clean_data(f_busu)

    # Penghapusan stopwords
    def remove_stopwords(text):
        stop = stopwords.words('indonesian')
        return ' '.join([word for word in text.split() if word not in stop])

    f_busu_clean['text_StopWord'] = f_busu_clean['text_clean'].apply(remove_stopwords)

    # Stemming
    def stemming(text):
        stemmer = SnowballStemmer('indonesian')
        return ' '.join([stemmer.stem(word) for word in text.split()])

    f_busu_clean['text_stemmed'] = f_busu_clean['text_StopWord'].apply(stemming)

    # Tampilan data setelah pembersihan
    st.subheader("Data Setelah Pembersihan (Clean Text):")
    st.write(f_busu_clean[['content', 'text_clean']].head(10))
    st.subheader("Data Setelah Penghapusan Stopword:")
