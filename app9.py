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
from imblearn.over_sampling import SMOTE
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

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
            color: #FFA500;
            text-align: center;
            margin-bottom: 20px;
        }
        .subheader {
            font-size: 18px;
            font-weight: bold;
            color: #4682B4;
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

    f_busu['sentiment'] = f_busu.apply(assign_sentiment, axis=1)

    # --- Pembersihan Data ---
    @st.cache_data
    def clean_text(df, text_field, new_text_field_name):
        df[new_text_field_name] = df[text_field].str.lower()  # Mengubah teks menjadi huruf kecil
        df[new_text_field_name] = df[new_text_field_name].apply(
            lambda elem: re.sub(r"@[A-Za-z0-9]+|(\w+:\/\/\S+)|^rt|http\S*|[^\w\s]", "", elem)
        )
        df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
        return df

    @st.cache_data
    def remove_stopwords(text):
        stop = stopwords.words('indonesian')
        return ' '.join([word for word in text.split() if word not in stop])

    # Membuat stemmer dari Sastrawi
    @st.cache_resource
    def create_stemmer():
        factory = StemmerFactory()
        return factory.create_stemmer()

    stemmer = create_stemmer()

    @st.cache_data
    def apply_stemming(text):
        return stemmer.stem(text)

    # Pembersihan teks
    f_busu_clean = clean_text(f_busu, 'content', 'text_clean')

    # Menghapus stopword
    f_busu_clean['text_StopWord'] = f_busu_clean['text_clean'].apply(remove_stopwords)

    # Melakukan stemming
    f_busu_clean['text_stemmed'] = f_busu_clean['text_StopWord'].apply(apply_stemming)

    # Menampilkan data setelah stemming
    st.subheader("Data Setelah Stemming:")
    st.write(f_busu_clean[['content', 'text_stemmed']].head(10))

    # --- Visualisasi ---
    def plot_sentiment_distribution(data, sentiment_column):
        sns.set(style="whitegrid")
        plt.figure(figsize=(7, 4))
        sns.countplot(x=sentiment_column, data=data, palette="Set2")
        plt.title('Distribusi Sentimen', fontsize=14)
        plt.xlabel('Sentimen', fontsize=12)
        plt.ylabel('Jumlah', fontsize=12)
        st.pyplot(plt)

    def plot_wordcloud(text):
        wordcloud = WordCloud(width=700, height=400, background_color="white", colormap="jet").generate(text)
        plt.figure(figsize=(7, 4))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

    st.subheader("Distribusi Sentimen:")
    plot_sentiment_distribution(f_busu_clean, 'sentiment')

    all_text = " ".join(f_busu_clean["text_stemmed"].dropna())
    st.subheader("Word Cloud:")
    plot_wordcloud(all_text)

    # --- Machine Learning ---
    X = f_busu_clean['text_stemmed']
    y = f_busu_clean['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)

    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train_smote, y_train_smote)

    y_pred = svm_model.predict(X_test_tfidf)

    st.subheader("Laporan Klasifikasi:")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).T
    st.write(report_df.style.background_gradient(cmap='coolwarm').highlight_max(axis=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Netral', 'Positif'], yticklabels=['Negatif', 'Netral', 'Positif'])
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Prediksi', fontsize=12)
    plt.ylabel('Sebenarnya', fontsize=12)
    st.pyplot(plt)

    # Tampilkan Matriks TF-IDF
    st.subheader("Matriks TF-IDF (Fitur Teratas):")
    tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
    st.write(tfidf_df.head(10))

    # Tambahkan tabel distribusi sentimen
    st.subheader("Tabel Distribusi Sentimen:")
    sentiment_counts = f_busu_clean['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentimen', 'Jumlah']
    st.write(sentiment_counts)

    # Tambahkan tampilan kata hasil stemming
    st.subheader("Kata-Kata Setelah Stemming:")
    stemmed_words = " ".join(f_busu_clean["text_stemmed"].dropna()).split()
    unique_stemmed_words = pd.DataFrame(stemmed_words, columns=["Kata"]).value_counts().reset_index()
    unique_stemmed_words.columns = ["Kata", "Frekuensi"]
    st.write(unique_stemmed_words.head(20))

    f_busu_clean.to_csv('data_reviews_with_sentiment_cleaned.csv', index=False)
    st.download_button(
        label="Unduh Data yang Sudah Diproses",
        data=f_busu_clean.to_csv(index=False),
        file_name="data_reviews_with_sentiment_cleaned.csv",
        mime="text/csv"
    )
