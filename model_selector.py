import torch
import torch.nn as nn
import numpy as np
import gensim.downloader as api
from sklearn.decomposition import TruncatedSVD
import re
import tkinter as tk
from tkinter import messagebox
from torchtext.data.utils import get_tokenizer
import pandas as pd
import argparse
from sklearn.decomposition import TruncatedSVD
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from tqdm import tqdm

delim = '%@%'

class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, 5)  # produce a prediction for each of 5 labels
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out[-1])
        tag_scores = F.softmax(tag_space, dim=0)
        return tag_scores.squeeze()

    def embed(self, word2idx, embedding_dim):
        print('loading glove')
        emb = load_embedding_model("glove-wiki-gigaword-200")
        m = np.ones((len(word2idx.keys()), 200))
        print('embedding glove')
        for word in word2idx.keys():
            try:
                m[word2idx[word]] = emb.get_vector(word)  # get word embedding (200 numbers)
            except KeyError:
                m[word2idx[word]] = emb.get_vector("random")  # garbage initialization
                continue

        if embedding_dim < 200:
            m_reduced = reduce_dimension(torch.from_numpy(m), embedding_dim)  # reduce to EMBEDDING_DIM per word
        else:
            m_reduced = m
        embed = torch.from_numpy(m_reduced).float()  # set embedding for word
        self.word_embeddings.weight.data = embed

def load_embedding_model(model):
    return api.load(model)

def reduce_dimension(M, k=2):
    n_iter = 10
    random_state = 595
    M_reduced = TruncatedSVD(n_components=k, n_iter=n_iter, random_state=random_state).fit_transform(M)
    M_reduced = M_reduced / np.linalg.norm(M_reduced, axis=1, keepdims=True)
    return M_reduced

def tokenize(line):
    result = line.lower()
    special_tokens = ["?", "x", "*", "^", "-", "+", "=", "<", ">"]
    pattern = '|'.join(map(re.escape, special_tokens))
    result = re.sub(f'({pattern})', r' \1 ', result)  # add spaces around special tokens
    result = re.sub(r'\d+', '<num>', result)  # replace all numbers with a special number token
    result = result.split()
    return result

def generateDict(data):
    word2idx = {}
    _wordcount = {}
    RARE_THRESH = 3
    word2idx["unka"] = 0
    for line in data:
        for word in tokenize(line):
            if word not in word2idx:
                word2idx[word] = len(word2idx)
                _wordcount[word] = 0
            _wordcount[word] += 1
    rare_words = [k for k, v in _wordcount.items() if v < RARE_THRESH]
    for word in rare_words:
        word2idx[word] = 0
    return word2idx

def encode_labels(labels):
    labels_encoding = {
        'CONV': 0,
        'SCIENCE': 1,
        'MATH': 2,
        'LAW': 3,
        'CODE': 4,
    }
    return np.array([labels_encoding[label] for label in labels], dtype=int)

def create_tensors(input, dict):
    def create_tensor_helper(line, dict):
        return torch.tensor([dict[word] if word in dict else dict["unka"] for word in tokenize(line)])
    return [create_tensor_helper(line, dict) for line in input]

def load_data(set):
    train_file = 'data/final_data/train.csv'
    val_file = 'data/final_data/val.csv'
    test_file = 'data/final_data/test.csv'

    # read csv into pandas dataframe then covert to numpy array
    file = train_file if set == 'train' else val_file if set == 'val' else test_file
    df = pd.read_csv(file, delimiter=delim, engine='python')
    data = df['question'].to_numpy()
    labels = df['label'].to_numpy()
    return data, labels

class ModelSelectorApp:
    def __init__(self, master, word2idx):
        self.master = master
        self.master.title("Model Selector App")
        self.word2idx = word2idx  # Make word2idx an attribute

        # Load the trained model
        model_path = 'output/model.torch'  # Replace with the actual path
        self.model = torch.load(model_path)
        self.model.eval()

        # Create UI elements
        self.label = tk.Label(master, text="Enter your question:")
        self.label.pack(pady=10)

        self.entry = tk.Entry(master, width=50)
        self.entry.pack(pady=10)

        self.button = tk.Button(master, text="Get Best Model", command=self.get_best_model)
        self.button.pack(pady=20)

    def get_best_model(self):
        # Get the user's question from the entry widget
        question = self.entry.get()

        # Convert question to input tensor
        tokenizer = get_tokenizer("basic_english")
        tokens = tokenizer(question)
        input_tensor = torch.tensor([self.word2idx[token] if token in self.word2idx else self.word2idx["unka"] for token in tokens])

        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)

        # Assuming output is a tensor with softmax applied, get the predicted label index
        _, predicted_index = torch.max(output, 0)

        # Display the predicted label
        labels_encoding = {
            0: 'CONV',
            1: 'SCIENCE',
            2: 'MATH',
            3: 'LAW',
            4: 'CODE',
        }
        predicted_label = labels_encoding[predicted_index.item()]

        # Show the result in a message box
        result_message = f"The predicted model for this question is: {predicted_label}"
        messagebox.showinfo("Model Selection Result", result_message)

def main():
    root = tk.Tk()

    # Load the data and generate word2idx
    train_data_raw, _ = load_data('train')  # Assuming train data is available
    word2idx = generateDict(train_data_raw)

    app = ModelSelectorApp(root, word2idx)
    root.mainloop()

if __name__ == "__main__":
    main()
