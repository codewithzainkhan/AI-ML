import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel


df= pd.read_csv(r'Data\IMDB Dataset.csv')

def clean_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def tokenize_text(text):
    return word_tokenize(text.lower())


stop_words = set(stopwords.words('english'))
def remove_stopwords(tokens):
    return [token for token in tokens if token not in stop_words]

stemmer = PorterStemmer()
def apply_stemming(tokens):
    return [stemmer.stem(token) for token in tokens]

def preprocess_text(text):
    text = clean_html_tags(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = apply_stemming(tokens)
    return ' '.join(tokens)

df['clean_review'] = df['review'].apply(preprocess_text)

X = df['clean_review']
y = df['sentiment']

vectorizer = CountVectorizer()
X_vectors = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

classifier = MultinomialNB()

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

print('\nClassification Report:')
print(classification_report(y_test, predictions))