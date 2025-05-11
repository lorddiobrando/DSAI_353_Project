from datasets import load_dataset
import json

def main():
    print("Downloading IMDb dataset...")
    dataset = load_dataset("imdb")

    texts = dataset['train']['text'] + dataset['test']['text']
    labels = dataset['train']['label'] + dataset['test']['label']

    print(f"Total samples: {len(texts)}")

    with open("imdb_texts.txt", "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line.replace("\n", " ") + "\n")

    with open("imdb_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f)

if __name__ == "__main__":
    main()
