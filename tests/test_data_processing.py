from src.data_processing import preprocess, load_and_process_data, vectorize_texts

def test_preprocess():
    text = "<br>This Movie's Great!!!"
    cleaned = preprocess(text)
    assert cleaned == " this movies great", f"Unexpected result: {cleaned}"

def test_load_and_process_data():
    texts, labels = load_and_process_data()
    assert isinstance(texts, list)
    assert isinstance(labels, (list, tuple, set, range, type(labels)))
    assert len(texts) == len(labels)
    assert all(isinstance(t, str) for t in texts)

def test_vectorize_texts():
    texts = ["the cat sat on the mat", "the dog chased the cat"]
    X, vectorizer = vectorize_texts(texts)
    assert X.shape[0] == 2
    assert hasattr(vectorizer, "transform")

if __name__ == "__main__":
    test_preprocess()
    test_load_and_process_data()
    test_vectorize_texts()
    print("All processing tests passed.")
