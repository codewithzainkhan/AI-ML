import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def drop_nan_rows(df):
    df.dropna(axis=0, inplace=True)
    return df

def removeHTML(df):
    df['question1'] = df['question1'].apply(lambda x: BeautifulSoup(x, "html.parser").get_text())
    df['question2'] = df['question2'].apply(lambda x: BeautifulSoup(x, "html.parser").get_text())
    return df

def preprocess_text(df):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    df['question1'] = df['question1'].apply(lambda x: [lemmatizer.lemmatize(word) for word in word_tokenize(x.lower()) if word not in stop_words])
    df['question2'] = df['question2'].apply(lambda x: [lemmatizer.lemmatize(word) for word in word_tokenize(x.lower()) if word not in stop_words])
    
    return df

def calculate_common_words(row):
    set1 = set(row['question1'])
    set2 = set(row['question2'])
    return len(set1.intersection(set2))

def vectorize_text(df):
    tfidf_vectorizer = TfidfVectorizer(max_features=500)
    X_tfidf1 = tfidf_vectorizer.fit_transform(df['question1'].apply(lambda x: ' '.join(x)))
    X_tfidf2 = tfidf_vectorizer.transform(df['question2'].apply(lambda x: ' '.join(x)))
    X_tfidf = pd.concat([pd.DataFrame(X_tfidf1.toarray()), pd.DataFrame(X_tfidf2.toarray())], axis=1)
    return X_tfidf

def preprocess_data(df):
    df = drop_nan_rows(df)
    df = removeHTML(df)
    df = preprocess_text(df)
    df['common_words'] = df.apply(calculate_common_words, axis=1)
    return df

df = pd.read_csv(r'Data\questions.csv')

df.drop(columns=['id', 'qid1', 'qid2'], inplace=True)

df = preprocess_data(df)

X_tfidf = vectorize_text(df)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['is_duplicate'], stratify=df['is_duplicate'], test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=300)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(X_train_pca.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train_pca, y_train_smote, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

predictions = model.predict(X_test_pca)
predictions_binary = [1 if p > 0.5 else 0 for p in predictions]
accuracy = accuracy_score(y_test, predictions_binary)
print(f'Accuracy: {accuracy:.2f}')

print('\nClassification Report:')
print(classification_report(y_test, predictions_binary))

print("Shape of X_tfidf:", X_tfidf.shape)
print(X_tfidf)
