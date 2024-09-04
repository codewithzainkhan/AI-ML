import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv(r'Data\IMDB Dataset.csv')

def clean_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

df['clean_review'] = df['review'].apply(clean_html_tags)

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

df['clean_review'] = df['clean_review'].apply(remove_stopwords)

X = df['clean_review']
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)

maxlen = 100
X_pad = pad_sequences(X_seq, maxlen=maxlen)

X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=maxlen))
model.add(SimpleRNN(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, callbacks=[EarlyStopping(patience=3)])

predictions = model.predict(X_test)
predictions_binary = [1 if p > 0.5 else 0 for p in predictions]
accuracy = accuracy_score(y_test, predictions_binary)
print(f'Accuracy: {accuracy:.2f}')

print('\nClassification Report:')
print(classification_report(y_test, predictions_binary))
