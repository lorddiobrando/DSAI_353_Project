import os
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from flask import Flask, request, render_template

app = Flask(__name__)

# Define the model architecture
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.norm(self.relu(self.conv(x)))

class TCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_levels=3):
        super().__init__()
        layers = []
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else out_channels
            layers.append(TemporalBlock(in_ch, out_channels, kernel_size=3, dilation=dilation))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.tcn = TCN(128, 64)
        self.ln = nn.LayerNorm(64)
        self.bilstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.3)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        nn.init.constant_(self.fc2.bias, -1.0)

    def forward(self, x):
        x = self.embedding(x)              # [B, L, E]
        x = x.permute(0, 2, 1)             # [B, E, L]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # [B, C, L//2]
        x = self.tcn(x)
        x = x.permute(0, 2, 1)             # [B, L, C]
        x = self.ln(x)
        lstm_out, _ = self.bilstm(x)       # [B, L, 2H]
        lstm_out = self.dropout_lstm(lstm_out)
        lstm_out = lstm_out.permute(0, 2, 1)
        pooled = self.global_pool(lstm_out).squeeze(-1)
        x = self.dropout(F.relu(self.fc1(pooled)))
        return torch.sigmoid(self.fc2(x)).squeeze()

def load_artifacts():
    with open("./Models/config.json", "r") as f:
        config = json.load(f)
    with open("./Models/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    return config, vocab

def load_model(config, vocab):
    model = SentimentModel(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        padding_idx=vocab["<pad>"]
    )
    model.load_state_dict(torch.load(os.path.join("./Models", "complete_model.pth"), map_location=torch.device("cpu"), weights_only=True))
    model.eval()
    return model

def preprocess_text(text, vocab, min_len=6):
    tokens = [vocab.get(word, vocab["<pad>"]) for word in text.lower().split()]
    if len(tokens) < min_len:
        tokens += [vocab["<pad>"]] * (min_len - len(tokens))
    return torch.tensor([tokens])


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    
    if len(text.strip().split()) < 2:
        return render_template('index.html', prediction="Please enter a longer sentence (at least 2 words).", text=text)

    try:
        config, vocab = load_artifacts()
        model = load_model(config, vocab)

        input_tensor = preprocess_text(text, vocab)

        with torch.no_grad():
            output = model(input_tensor)
            prediction = "Positive" if output.item() > 0.5 else "Negative"

        print("Text received:", text)
        print("Model output:", output.item())
        print("Prediction:", prediction)
        return render_template('index.html', prediction=prediction, text=text)
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "An error occurred during prediction.", 500


if __name__ == '__main__':
    app.run(debug=True)