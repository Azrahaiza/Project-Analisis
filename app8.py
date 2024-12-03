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
st.set_page_config(page_title="Analisis Sentimen Penggunaan Aplikasi Info BMKG", page_icon="🌍", layout="wide")
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
            color: #FFF5E1;
            text-align: center;
            margin-bottom: 20px;
        }
        .subheader {
            font-size: 24px;
            font-weight: bold;
            color: #4682B4;
        }
        .button {
            font-size: 16px;
            background-color: #2E8B57;
            color: green;
            padding: 10px 16px;
            border-radius: 5px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Custom Title and Description
st.markdown("""
    <div class="title">
        Analisis Sentimen Penggunaan Aplikasi Info BMKG
    </div>
    <div class="description">
        Visualisasi Pemrosesan Data
    </div>
""", unsafe_allow_html=True)

# Sidebar for uploading data and navigation
st.sidebar.header("Navigasi")
st.sidebar.write("Gunakan sidebar untuk mengunggah dataset dan memilih bagian yang ingin dilihat.")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    f_busu = pd.read_csv(uploaded_file)

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

    # --- Tampilan Data ---
    st.subheader("Data Ulasan:")
    st.write(f_busu.head(10))  # Tampilkan 10 data pertama

    # Menampilkan data berdasarkan sentimen
    st.subheader("Data dengan Sentimen Positif:")
    st.write(f_busu[f_busu['sentiment'] == 'Positif'].head(10))

    st.subheader("Data dengan Sentimen Netral:")
    st.write(f_busu[f_busu['sentiment'] == 'Netral'].head(10))

    st.subheader("Data dengan Sentimen Negatif:")
    st.write(f_busu[f_busu['sentiment'] == 'Negatif'].head(10))

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
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sns.countplot(x=sentiment_column, data=data, palette="Set2")
        plt.title('Distribusi Sentimen', fontsize=16)
        plt.xlabel('Sentimen', fontsize=14)
        plt.ylabel('Jumlah', fontsize=14)
        st.pyplot(plt)

    def plot_wordcloud(text):
        wordcloud = WordCloud(width=700, height=400, background_color="white", colormap="jet").generate(text)
        plt.figure(figsize=(8, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

    # Visualisasi Distribusi Sentimen
    st.subheader("Distribusi Sentimen:")
    plot_sentiment_distribution(f_busu_clean, 'sentiment')

    # Word Cloud
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
    
    # Menampilkan Laporan Klasifikasi dengan Gaya Menarik
    st.subheader("Laporan Klasifikasi:")
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Menampilkan laporan dalam format DataFrame untuk gaya lebih menarik
    report_df = pd.DataFrame(report).T
    st.markdown("**Laporan Klasifikasi (Macro Avg & Weighted Avg)**")
    st.write(report_df.style.background_gradient(cmap='coolwarm').highlight_max(axis=0))

    # Menampilkan Confusion Matrix dengan styling
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Netral', 'Positif'], yticklabels=['Negatif', 'Netral', 'Positif'])
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Prediksi', fontsize=14)
    plt.ylabel('Sebenarnya', fontsize=14)
    st.pyplot(plt)

    # --- Menyimpan dan Mengunduh Data yang Sudah Diproses ---
    f_busu_clean.to_csv('data_reviews_with_sentiment_cleaned.csv', index=False)
    st.download_button(
        label="Unduh Data yang Sudah Diproses",
        data=f_busu_clean.to_csv(index=False),
        file_name="data_reviews_with_sentiment_cleaned.csv",
        mime="text/csv"
    )
