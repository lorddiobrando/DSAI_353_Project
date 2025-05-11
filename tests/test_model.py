import numpy as np
from src.model import load_model

def test_model_prediction():
    model, vectorizer = load_model()
    sample_texts = ["This movie was fantastic!", "Terrible plot and bad acting."]
    X_sample = vectorizer.transform(sample_texts)
    predictions = model.predict(X_sample)
    assert len(predictions) == 2
    assert all(p in [0, 1] for p in predictions)

if __name__ == "__main__":
    test_model_prediction()
    print("test model prediction passed.")
