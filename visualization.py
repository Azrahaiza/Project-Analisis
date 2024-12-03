import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Sentiment distribution
def plot_sentiment_distribution(df):
    sns.countplot(x='sentiment', data=df)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    return plt

# Word cloud
def plot_wordcloud(text):
    wordcloud = WordCloud(width=700, height=400, background_color="white").generate(text)
    plt.figure(figsize=(5, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return plt
