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

# --- Load Dataset ---
st.title("Analisis Sentimen Ulasan Penggunaan Aplikasi Info BMKG")
st.write("**Visualisasi data mentah dan hasil proses pembersihan data secara bertahap.**")

# Load data
f_busu = pd.read_csv('data_reviews_with_sentiment (5).csv')  # Replace with your dataset

# Tampilkan data mentah
st.subheader("Data Mentah:")
st.write(f_busu.head(10))  # Menampilkan 10 data pertama dari dataset mentah

# --- Data Cleaning ---
def clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()  # Convert to lowercase
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

# Clean text
f_busu_clean = clean_text(f_busu, 'content', 'text_clean')

# Tampilkan data setelah clean text
st.subheader("Data Setelah Pembersihan (Clean Text):")
st.write(f_busu_clean[['content', 'text_clean']].head(10))

# Hapus stopword
f_busu_clean['text_StopWord'] = f_busu_clean['text_clean'].apply(remove_stopwords)

# Tampilkan data setelah penghapusan stopword
st.subheader("Data Setelah Penghapusan Stopword:")
st.write(f_busu_clean[['text_clean', 'text_StopWord']].head(10))

# Lakukan stemming
f_busu_clean['text_stemmed'] = f_busu_clean['text_StopWord'].apply(apply_stemming)

# Tampilkan data setelah stemming
st.subheader("Data Setelah Stemming:")
st.write(f_busu_clean[['text_StopWord', 'text_stemmed']].head(10))

# --- Visualization ---
def plot_sentiment_distribution(data, sentiment_column):
    sns.countplot(x=sentiment_column, data=data)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    st.pyplot(plt)

def plot_wordcloud(text):
    wordcloud = WordCloud(width=700, height=400, background_color="white", colormap="jet").generate(text)
    plt.figure(figsize=(5, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# Visualizations
if 'sentiment' in f_busu_clean.columns:
    st.subheader("Distribusi Sentimen:")
    plot_sentiment_distribution(f_busu_clean, 'sentiment')

all_text = " ".join(f_busu_clean["text_stemmed"].dropna())
st.subheader("Word Cloud:")
plot_wordcloud(all_text)

# --- Machine Learning ---
X = f_busu_clean['text_stemmed']
y = f_busu_clean['sentiment']

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)

# Train model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_smote, y_train_smote)

# Evaluate model
y_pred = svm_model.predict(X_test_tfidf)
st.subheader("Classification Report:")

# Menyusun data untuk macro avg dan weighted avg
classification_rep = classification_report(y_test, y_pred, output_dict=True)

# Menampilkan Macro avg dan Weighted avg dalam format tabel
macro_avg = classification_rep['macro avg']
weighted_avg = classification_rep['weighted avg']

metrics_data = {
    "Metric": ["Precision", "Recall", "F1-Score", "Support"],
    "Macro Avg": [
        round(macro_avg["precision"], 2),
        round(macro_avg["recall"], 2),
        round(macro_avg["f1-score"], 2),
        int(macro_avg["support"])
    ],
    "Weighted Avg": [
        round(weighted_avg["precision"], 2),
        round(weighted_avg["recall"], 2),
        round(weighted_avg["f1-score"], 2),
        int(weighted_avg["support"])
    ]
}

# Mengonversi ke dataframe untuk ditampilkan dalam tabel
metrics_df = pd.DataFrame(metrics_data)

# Menampilkan tabel dengan tampilan yang menarik
st.table(metrics_df)

# Menampilkan lebih detail menggunakan markdown untuk penekanan
st.markdown(
    f"""
    **Macro Avg**:
    - Precision: {round(macro_avg['precision'], 2)}
    - Recall: {round(macro_avg['recall'], 2)}
    - F1-Score: {round(macro_avg['f1-score'], 2)}
    - Support: {int(macro_avg['support'])}

    **Weighted Avg**:
    - Precision: {round(weighted_avg['precision'], 2)}
    - Recall: {round(weighted_avg['recall'], 2)}
    - F1-Score: {round(weighted_avg['f1-score'], 2)}
    - Support: {int(weighted_avg['support'])}
    """
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Netral', 'Positif'], yticklabels=['Negatif', 'Netral', 'Positif'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
st.pyplot(plt)

# --- Save and Download Processed Data ---
f_busu_clean.to_csv('data_reviews_with_sentiment_cleaned.csv', index=False)
st.download_button(
    label="Download Cleaned Data",
    data=f_busu_clean.to_csv(index=False),
    file_name="data_reviews_with_sentiment_cleaned.csv",
    mime="text/csv"
)

