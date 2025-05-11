from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.data_processing import load_and_process_data, vectorize_texts
from src.model import train_model, save_model

def main():
    texts, labels = load_and_process_data()
    X, vectorizer = vectorize_texts(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    save_model(model, vectorizer)

if __name__ == "__main__":
    main()
