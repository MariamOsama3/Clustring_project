import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import Normalizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """Clean and preprocess text"""
    # Step 1: Remove newlines and extra spaces
    text = text.replace("\n", " ").strip()

    # Step 2: Convert to lowercase
    text = text.lower()

    # Step 3: Remove punctuation
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)

    # Step 4: Tokenization
    words = word_tokenize(text)

    # Step 5: Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Step 6: Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Step 7: Join words back
    return ' '.join(words)

def vectorize_text(text_series, vectorizer_type='tfidf', max_features=15000):
    """Vectorize text using specified vectorizer"""
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features)
    else:  # count
        vectorizer = CountVectorizer(max_features=max_features)
        
    numeric_text = vectorizer.fit_transform(text_series)
    normalizer = Normalizer()
    return normalizer.fit_transform(numeric_text)