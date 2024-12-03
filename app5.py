import streamlit as st
import pandas as pd
import re
import nltk
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Cleaning functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r"@[A-Za-z0-9]+|https?:\/\/\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('indonesian'))
    return " ".join([word for word in text.split() if word not in stop_words])

def stem_text(text):
    stemmer = PorterStemmer()
    return " ".join([stemmer.stem(word) for word in text.split()])

# Load data
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Select Section", ["Home", "Data Analysis", "Model Training"])

if menu == "Home":
    st.title("Sentiment Analysis Application")
    st.markdown(
        """
        <div style="background-color:#e8f5e9;padding:10px;border-radius:10px;">
            <h3 style="color:#2e7d32;">Welcome to the Sentiment Analysis App!</h3>
            <p>Analyze reviews to determine sentiment (positive, negative, or neutral).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif menu == "Data Analysis":
    st.title("Data Analysis")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.write(df.head())

        # Clean text data
        df['cleaned_text'] = df['content'].apply(clean_text)
        df['cleaned_text'] = df['cleaned_text'].apply(remove_stopwords)
        df['cleaned_text'] = df['cleaned_text'].apply(stem_text)

        # Word Cloud
        st.subheader("Word Cloud")
        all_text = " ".join(df['cleaned_text'])
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)

        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # Sentiment Distribution
        if 'sentiment' in df.columns:
            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x='sentiment', data=df, ax=ax, palette="Set2")
            st.pyplot(fig)

elif menu == "Model Training":
    st.title("Model Training")

    if 'df' in locals():
        if 'sentiment' in df.columns:
            # Prepare data
            X = df['cleaned_text']
            y = df['sentiment']

            vectorizer = TfidfVectorizer(max_features=5000)
            X_vec = vectorizer.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)

            # Train model
            model = SVC(kernel='linear')
            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)

            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            st.pyplot(fig)

    else:
        st.warning("Please upload a dataset in the Data Analysis section first.")
