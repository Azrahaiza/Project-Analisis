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
from sklearn.metrics import confusion_matrix, classification_report
from nltk.stem import PorterStemmer
from imblearn.over_sampling import SMOTE

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Streamlit setup
st.title("Analisis Sentimen Ulasan Penggunaan Aplikasi Info BMKG")
st.write("**Visualisasi data mentah dan hasil proses pembersihan data secara bertahap.**")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    f_busu = pd.read_csv(uploaded_file)
    st.subheader("Data Mentah:")
    st.write(f_busu.head(10))  # Menampilkan 10 data pertama dari dataset mentah

    # --- Penentuan Sentimen Berdasarkan Score ---
    def assign_sentiment(row):
        if row['score'] >= 4:  # Score 4 atau 5 = Positif
            return 'Positif'
        elif row['score'] == 3:  # Score 3 = Netral
            return 'Netral'
        else:  # Score 1 atau 2 = Negatif
            return 'Negatif'

    # Misalkan kolom 'score' berisi score (1-5)
    f_busu['sentiment'] = f_busu.apply(assign_sentiment, axis=1)

    # Tampilkan data berdasarkan sentiment
    st.subheader("Data dengan Sentimen Positif:")
    st.write(f_busu[f_busu['sentiment'] == 'Positif'].head(10))  # Menampilkan 10 data pertama dengan sentimen Positif

    st.subheader("Data dengan Sentimen Netral:")
    st.write(f_busu[f_busu['sentiment'] == 'Netral'].head(10))  # Menampilkan 10 data pertama dengan sentimen Netral

    st.subheader("Data dengan Sentimen Negatif:")
    st.write(f_busu[f_busu['sentiment'] == 'Negatif'].head(10))  # Menampilkan 10 data pertama dengan sentimen Negatif

    # --- Pembersihan Data ---
    def clean_text(df, text_field, new_text_field_name):
        df[new_text_field_name] = df[text_field].str.lower()  # Mengubah teks menjadi huruf kecil
        df[new_text_field_name] = df[new_text_field_name].apply(
            lambda elem: re.sub(r"@[A-Za-z0-9]+|(\w+:\/\/\S+)|^rt|http\S*|[^\w\s]", "", elem)
        )
        df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
        return df

    def remove_stopwords(text):
        stop = stopwords.words('indonesian')
        return ' '.join([word for word in text.split() if word not in stop])

    def apply_stemming(text):
        stemmer = PorterStemmer()
        return ' '.join([stemmer.stem(word) for word in text.split()])

    # Pembersihan teks
    f_busu_clean = clean_text(f_busu, 'content', 'text_clean')

    # Menampilkan data setelah pembersihan
    st.subheader("Data Setelah Pembersihan (Clean Text):")
    st.write(f_busu_clean[['content', 'text_clean']].head(10))

    # Menghapus stopword
    f_busu_clean['text_StopWord'] = f_busu_clean['text_clean'].apply(remove_stopwords)

    # Menampilkan data setelah penghapusan stopword
    st.subheader("Data Setelah Penghapusan Stopword:")
    st.write(f_busu_clean[['text_clean', 'text_StopWord']].head(10))

    # Melakukan stemming
    f_busu_clean['text_stemmed'] = f_busu_clean['text_StopWord'].apply(apply_stemming)

    # Menampilkan data setelah stemming
    st.subheader("Data Setelah Stemming:")
    st.write(f_busu_clean[['text_StopWord', 'text_stemmed']].head(10))

    # --- Visualisasi ---
    def plot_sentiment_distribution(data, sentiment_column):
        sns.countplot(x=sentiment_column, data=data)
        plt.title('Distribusi Sentimen')
        plt.xlabel('Sentimen')
        plt.ylabel('Jumlah')
        st.pyplot(plt)

    def plot_wordcloud(text):
        wordcloud = WordCloud(width=700, height=400, background_color="white", colormap="jet").generate(text)
        plt.figure(figsize=(5, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

    # Distribusi Sentimen
    st.subheader("Distribusi Sentimen:")
    plot_sentiment_distribution(f_busu_clean, 'sentiment')

    # Word cloud
    all_text = " ".join(f_busu_clean["text_stemmed"].dropna())
    st.subheader("Word Cloud:")
    plot_wordcloud(all_text)

    # --- Machine Learning ---
    X = f_busu_clean['text_stemmed']
    y = f_busu_clean['sentiment']

    # Menyiapkan data untuk pelatihan
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # SMOTE untuk mengatasi ketidakseimbangan data
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)

    # Melatih model SVM
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train_smote, y_train_smote)

    # Evaluasi model
    y_pred = svm_model.predict(X_test_tfidf)
    st.subheader("Laporan Klasifikasi:")
    st.write(classification_report(y_test, y_pred))

    # Matriks kebingunguan
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Netral', 'Positif'], yticklabels=['Negatif', 'Netral', 'Positif'])
    plt.title('Matriks Kebingunguan')
    plt.xlabel('Prediksi')
    plt.ylabel('Sebenarnya')
    st.pyplot(plt)

    # --- Menyimpan dan Mengunduh Data yang Sudah Diproses ---
    f_busu_clean.to_csv('data_reviews_with_sentiment_cleaned.csv', index=False)
    st.download_button(
        label="Unduh Data yang Sudah Diproses",
        data=f_busu_clean.to_csv(index=False),
        file_name="data_reviews_with_sentiment_cleaned.csv",
        mime="text/csv"
    )
