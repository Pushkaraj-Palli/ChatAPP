import joblib

vectorizer = joblib.load("tfidfvectoizer.pkl")
print(type(vectorizer))  # Should print <class 'sklearn.feature_extraction.text.TfidfVectorizer'>
