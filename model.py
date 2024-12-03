from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

# Train model function
def train_model(X, y):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

    model = SVC(kernel='linear')
    model.fit(X_resampled, y_resampled)
    return model, vectorizer

# Evaluate model
def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    return classification_report(y_test, y_pred)
