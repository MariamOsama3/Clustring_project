from requirments.py import *

def preprocess_text(text):
    text = text.replace("\n", " ").strip().lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)
