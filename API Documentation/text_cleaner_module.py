import pandas as pd
import regex as re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Convert Filter File to Data Frame
df_kamus_alay = pd.read_csv('API Documentation/filter/new_kamusalay.csv', encoding='latin-1', sep=',', names=['Alay', 'Normal'])
df_kamus_alay_dict = dict(zip(df_kamus_alay.iloc[:, 0], df_kamus_alay.iloc[:, 1]))

df_stopwordbahasa = pd.read_csv('API Documentation/filter/stopwordbahasa.csv')

# Load stopwords into a set for faster lookup
stopwords = set(df_stopwordbahasa['stopword'])

# Create stemmer object
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Text Preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove emoticons and special characters
    text = re.sub(r'\\x[\da-fA-F]{2}', '', text)

    # Keep only characters from a to z (both uppercase and lowercase) and digits
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)

    # Remove excessive white spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove digits or numbers
    text = re.sub(r"\b\d+\b", " ", text)

    # Remove digits from strings
    text = re.sub("\S*\d\S*", "", text).strip()

    return text

# Removing Stopwords
def remove_stopwords(text):
    return ' '.join(word for word in text.split() if word not in stopwords)

# Normalization
def normalize(text):
    words = text.split()
    normalized_words = [df_kamus_alay_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

# Stemming Words
def stem_text(text):
    return stemmer.stem(text)

# Text Cleaner Function
def cleansing_text(text):
    text = preprocess_text(text)
    text = normalize(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    return text