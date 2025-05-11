import joblib
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    clf = LogisticRegression(max_iter=300)
    clf.fit(X_train, y_train)
    return clf

def save_model(model, vectorizer, model_path="lr_model.joblib", vectorizer_path="tfidf_vectorizer.joblib"):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

def load_model(model_path="lr_model.joblib", vectorizer_path="tfidf_vectorizer.joblib"):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer
