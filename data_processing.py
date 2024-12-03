import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Clean text function
def clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(
        lambda elem: re.sub(r"@[A-Za-z0-9]+|(\w+:\/\/\S+)|^rt|http\S*|[^\w\s]", "", elem)
    )
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    return df

# Remove stopwords
def remove_stopwords(text):
    stop = stopwords.words('indonesian')
    return ' '.join([word for word in text.split() if word not in stop])

# Apply stemming
def apply_stemming(text):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in text.split()])
