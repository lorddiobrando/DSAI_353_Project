import re
import numpy as np
from datasets import load_dataset 
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)  # Remove HTML breaks
    text = re.sub(r"[^a-z\s]", "", text)    # Remove punctuation/numbers
    return text

def load_and_process_data():
    dataset = load_dataset("imdb")
    texts = dataset['train']['text'] + dataset['test']['text']
    labels = dataset['train']['label'] + dataset['test']['label']
    texts = [preprocess(t) for t in texts]
    return texts, np.array(labels)

def vectorize_texts(texts):
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X = vectorizer.fit_transform(texts)
    return X, vectorizer
