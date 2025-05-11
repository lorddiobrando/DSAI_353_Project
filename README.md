# DSAI_353_Project

## Project Overview

This repository hosts a comprehensive exploration of sentiment analysis techniques applied to the IMDB Large Movie Review Dataset. The project evaluates a variety of models, including traditional machine learning approaches (e.g., Logistic Regression, RandomSyndrome, Random Forest, Ensemble Voting Classifier) and deep learning architectures (e.g., Bidirectional LSTM, Temporal Convolutional Network, and a Hybrid CNN+TCN+Bi-LSTM model). The goal is to benchmark these models for binary sentiment classification, identifying positive or negative sentiments in movie reviews.

## Setup

Follow these steps to set up the project environment:

1. **Clone the Repository**: Clone the project from GitHub to your local machine:

   ```bash
   git clone https://github.com/lorddiobrando/DSAI_353_Project.git
   cd DSAI_353_Project
   ```

2. **Install Dependencies**:

   - Ensure Python 3.6 or higher is installed.
   - For deep learning models, a GPU with CUDA support is recommended but not required.
   - Install dependencies using:

     ```bash
     pip install -r requirements.txt
     ```

     If no `requirements.txt` is present, install the following libraries manually:

     ```bash
     pip install scikit-learn torch numpy pandas nltk joblib matplotlib flask
     ```

     For GPU support with PyTorch, install the appropriate version:

     ```bash
     pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
     ```

     Check the [PyTorch website](https://pytorch.org/get-started/locally/) for the correct command based on your CUDA version.

3. **Download NLTK Data**: The project uses NLTK for text preprocessing. Download required NLTK data:

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

4. **Download Datasets and Embeddings**:

   - **IMDB Dataset**: Download the IMDB Large Movie Review Dataset from [Stanford AI](http://ai.stanford.edu/~amaas/data/sentiment/). Extract it to the `./Data/` directory or follow notebook instructions for automatic downloading.
   - **GloVe Embeddings**: Download the GloVe 6B 300d embeddings from [Stanford NLP](https://nlp.stanford.edu/projects/glove/). Place the `glove.6B.300d.txt` file in the `./Embeddings/` directory.

## Project Structure

The repository is organized as follows:

| Directory | Description |
| --- | --- |
| `./Models/` | Contains trained models: `.joblib` for scikit-learn, `.pth` for PyTorch. |
| `./Embeddings/` | Stores pre-trained embeddings, such as GloVe (`glove.6B.300d.txt`). |
| `./Notebooks/` | Includes Jupyter notebooks for training and evaluating each model. |
| `./Data/` | (Optional) Stores the IMDB dataset if included in the repository. |
| `./templates/` | Contains HTML templates for the Flask API (e.g., `index.html`). |

## Usage

The project is primarily driven by Jupyter notebooks, which provide detailed code for training, evaluating, and using the sentiment analysis models. Below are the steps to use the project:

1. **Run Jupyter Notebooks**:

   - Launch Jupyter Notebook from the project directory:

     ```bash
     jupyter notebook
     ```
   - Open notebooks in the `./Notebooks/` directory, such as `logistic-regression-model.ipynb`, `bi-lstm-only-architecture.ipynb`, or `nlp-complete-model.ipynb`.
   - Run the cells in each notebook to train models, evaluate performance, or visualize results.

2. **Train and Evaluate Models**:

   - Each notebook corresponds to a specific model (e.g., Logistic Regression, Bi-LSTM, Hybrid model).
   - Follow the instructions within each notebook to:
     - Load and preprocess the IMDB dataset.
     - Train the model.
     - Evaluate performance using metrics like accuracy and F1-score.
   - Notebooks may include hyperparameter tuning and architectural enhancements.

3. **Make Predictions with Trained Models**:

   - **Traditional Models** (e.g., Logistic Regression, Random Forest):
     - Models are saved as `.joblib` files in `./Models/` (e.g., `logistic_regression.joblib`).
     - Example usage:

       ```python
       import joblib
       from sklearn.feature_extraction.text import TfidfVectorizer
       model = joblib.load('./Models/logistic_regression.joblib')
       vectorizer = joblib.load('./Models/tfidf_vectorizer.pkl')
       new_review = ["This movie was fantastic!"]
       features = vectorizer.transform(new_review)
       prediction = model.predict(features)
       print("Positive" if prediction[0] == 1 else "Negative")
       ```
   - **Deep Learning Models** (e.g., Bi-LSTM, TCN, Hybrid):
     - Models are saved as `.pth` files in `./Models/` (e.g., `complete_model.pth`).
     - Example usage (requires model architecture definition):

       ```python
       import torch
       from model import SentimentModel  # Import model class from notebook or script
       model = SentimentModel(vocab_size=20000, embedding_dim=300, hidden_dim=128, padding_idx=0)
       model.load_state_dict(torch.load('./Models/complete_model.pth', map_location=torch.device('cpu')))
       model.eval()
       # Preprocess new_review into a tensor (refer to notebook for preprocessing)
       with torch.no_grad():
           output = model(new_review_tensor)
           prediction = "Positive" if output.item() > 0.5 else "Negative"
       print(prediction)
       ```
   - Refer to the specific notebook for each model for detailed preprocessing and inference steps.

## Flask API for Model Deployment

The project includes a Flask API that allows users to select a trained model, input a movie review, and receive a sentiment prediction through a web interface. The API supports both traditional machine learning models (`.joblib`) and deep learning models (`.pth`) stored in the `./Models` directory.

### Setting Up the Flask API

1. **Install Dependencies**:
   - Ensure you have installed  `scikit-learn`, `torch`, `numpy`, `pandas`, `nltk`, `joblib`, `matplotlib`, and `flask` installed as described in the "Setup" section.

2. **Verify Model Files**:
   - Ensure the `./Models` directory contains the trained model files (e.g., `logistic_regression.joblib`, `complete_model.pth`) and supporting files (`tfidf_vectorizer.pkl`, `vocab.pkl`, `config.json`).
   - Ensure the `./Embeddings` directory contains `glove.6B.300d.txt`.

3. **Directory Structure**:
   - Ensure the `app.py` file and the `./templates/index.html` file are in the project directory.

### Using the Flask API

1. **Run the Flask App**:
   - Navigate to the project directory and run:

     ```bash
     python app.py
     ```

   - Open a web browser and visit `http://127.0.0.1:5000` to access the API.

2. **Interact with the Interface**:
   - **Select a Model**: Choose a model from the dropdown menu (e.g., `logistic_regression.joblib`, `complete_model.pth`).
   - **Enter a Review**: Type a movie review in the text area (e.g., "This movie was absolutely fantastic!").
   - **Submit**: Click the "Predict" button to see the predicted sentiment ("Positive" or "Negative").

### Example Prediction
- **Input**: "I loved this movie!"
- **Model**: `logistic_regression.joblib`
- **Output**: "Positive"

### Notes
- The API dynamically loads models to optimize memory usage.
- Traditional models use TF-IDF vectorization, while deep learning models require tokenization and padding.
- If errors occur, check that all required files are present and correctly formatted.

## Contribution

We welcome contributions to enhance the project. To contribute, follow these steps:

1. **Fork the Repository**:
   - Fork the repository on GitHub.

2. **Create a Branch**:
   - Create a new branch for your feature or bug fix:

     ```bash
     git checkout -b feature/your-feature-name
     ```

3. **Make Changes**:
   - Implement your changes, ensuring code is well-documented.
   - Commit your changes with a clear message:

     ```bash
     git add .
     git commit -m "Add your descriptive commit message"
     ```

4. **Push Changes**:
   - Push your branch to your forked repository:

     ```bash
     git push origin feature/your-feature-name
     ```

5. **Create a Pull Request**:
   - Navigate to your fork on GitHub and create a pull request to the main repository’s `main` branch.
   - Provide a detailed description of your changes and their purpose.

6. **Follow Code Style**:
   - Adhere to [PEP 8 guidelines](https://www.python.org/dev/peps/pep-0008/) for Python code.
   - Use meaningful variable names and include comments to explain complex logic.

7. **Testing**:
   - Ensure your changes do not break existing functionality.
   - If tests are implemented (check the `./Tests/` directory or notebooks), run them to verify your changes.
   - If no tests exist, manually verify that your changes work with the IMDB dataset.

## Acknowledgments

- **IMDB Dataset**: Provided by [Stanford AI](http://ai.stanford.edu/~amaas/data/sentiment/).
- **GloVe Embeddings**: Provided by [Stanford NLP](https://nlp.stanford.edu/projects/glove/).