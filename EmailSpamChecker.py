import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from imblearn.under_sampling import RandomUnderSampler
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping



def removeHTML(df):
    df['clean_text'] = df['Message'].apply(lambda x: BeautifulSoup(x, "html.parser").get_text())
    return df


def tokenization(df):
    df['tokens'] = df['clean_text'].apply(lambda x: word_tokenize(x.lower()))
    return df


def Normalization(df):
    lemmatizer = WordNetLemmatizer()
    df['normalized_tokens'] = df['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    return df

def removeStopWords(df):
    stop_words = set(stopwords.words('english'))
    df['filtered_tokens'] = df['normalized_tokens'].apply(lambda x: [word for word in x if word not in stop_words])
    return df

def Vectorization(df):
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)  
    X_tfidf = tfidf_vectorizer.fit_transform(df['filtered_tokens'].apply(lambda x: ' '.join(x)))
    return X_tfidf

df = pd.read_csv(r'Data\spam.csv')

df = removeHTML(df)

df = tokenization(df)

df = Normalization(df)

df = removeStopWords(df)

X_tfidf = Vectorization(df)

df['spam'] = df['Category'].apply(lambda x: 1 if x =='spam' else 0)
print(df.shape)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df.spam,stratify=df['spam'], test_size=0.2)



print("Training Set - Class Distribution:")
print(y_train.value_counts())

print("\nTesting Set - Class Distribution:")
print(y_test.value_counts())


maxlen = X_tfidf.shape[1]
print("Shape of X_tfidf:", X_tfidf.shape)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_tfidf.shape[1],)))  
model.add(Dropout(0.2)) 
model.add(Dense(64, activation='relu'))  
model.add(Dropout(0.2))  
model.add(Dense(1, activation='sigmoid'))  

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[EarlyStopping(patience=3)])

predictions = model.predict(X_test)
predictions_binary = [1 if p > 0.5 else 0 for p in predictions]
accuracy = accuracy_score(y_test, predictions_binary)
print(f'Accuracy: {accuracy:.2f}')

print('\nClassification Report:')
print(classification_report(y_test, predictions_binary))


print(X_tfidf)
