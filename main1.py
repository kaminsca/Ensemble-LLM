import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
import gensim.downloader as api
import sklearn.metrics as metrics
from tqdm import tqdm
import re
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
delim = '%@%'

labels_encoding = {
    'CONV': 0,
    'SCIENCE': 1,
    'MATH': 2,
    'LAW': 3,
    'CODE': 4,
}

LOSS = nn.CrossEntropyLoss()


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


class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, 5)  # produce a prediction for each of 5 labels
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out[-1])
        tag_scores = torch.nn.functional.softmax(tag_space, dim=1)
        return tag_scores.squeeze()  # produces a tensor size (5,) each value is the probability of that label

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
    # list of special tokens so math equations and code get parsed correctly instead of just being one token
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
    rareWords = [k for k, v in _wordcount.items() if v < RARE_THRESH]
    for word in rareWords:
        word2idx[word] = 0
    return word2idx


def encode_labels(labels):
    return np.array([labels_encoding[label] for label in labels], dtype=int)


def create_tensors(input, word_dict):
    def create_tensor_helper(line, dict):
        return torch.tensor([dict[word] if word in dict else dict["unka"] for word in tokenize(line)])

    return [create_tensor_helper(line, word_dict) for line in input]


def plot_acc(train_acc, val_acc, test_acc):
    plt.figure()
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='Train')
    plt.plot(range(1, len(train_acc) + 1), val_acc, label='Validation')
    plt.plot(range(1, len(train_acc) + 1), [test_acc] * len(train_acc), label='Test', linestyle='--')
    plt.xticks(range(1, len(train_acc) + 1))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Epoch')
    plt.legend()
    plt.savefig('output/accuracy_by_epoch.png')


def train(params):
    print('initializing...')
    # load data, generate word2idx dictionary, create tensor for every question, encode labels, initialize model
    train_data, train_labels = load_data('train')
    word2idx = generateDict(train_data)
    train_data = create_tensors(train_data, word2idx)
    train_labels = encode_labels(train_labels)

    model = RNN(params.embed_dim, params.hidden_dim, len(word2idx))
    optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, weight_decay=0.001, momentum=0.9)
    model.embed(word2idx, params.embed_dim)

    validation_data, validation_labels = load_data('val')
    validation_data = create_tensors(validation_data, word2idx)
    validation_labels = encode_labels(validation_labels)

    outfile = open('output/report.txt', 'w')

    print('training...')
    best_validation_score = 0
    pbar_total = tqdm(total=len(train_data) * params.epochs, leave=False)
    pbar_total.set_description('total training progress')

    val_label_data = []
    train_acc = []
    val_acc = []

    for epoch in range(params.epochs):
        pbar = tqdm(total=len(train_data), leave=False)
        pbar.set_description(f'epoch {epoch + 1}')
        correct_guesses = 0
        for question, label in zip(train_data, train_labels):
            optimizer.zero_grad()
            out = model(question)
            loss = LOSS(out.unsqueeze(0), torch.LongTensor([label]))
            loss.backward()
            optimizer.step()
            pbar.update(1)
            pbar_total.update(1)
            correct_guesses += int(torch.argmax(out) == label)
        pbar.close()
        train_acc.append(correct_guesses / len(train_data))
        outfile.write(f'Epoch {epoch + 1} training accuracy: {correct_guesses / len(train_data)}\n')
        validation_score, validation_by_label = validate(model, word2idx, validation_data, validation_labels)
        val_label_data.append(validation_by_label)
        val_acc.append(validation_score)
        outfile.write(f'Epoch {epoch + 1} validation accuracy: {validation_score}\n')
        if validation_score > best_validation_score:
            best_validation_score = validation_score
            torch.save(model, 'output/model.torch')
            outfile.write('saved model\n')
        outfile.write('\n')
    pbar_total.close()
    outfile.close()

    # plot validation accuracy by label
    for l in range(len(labels_encoding.keys())):
        plt.plot(range(1, params.epochs + 1), [val_label_data[i][l] for i in range(len(val_label_data))])
    plt.legend(labels_encoding.keys())
    plt.xticks(range(1, params.epochs + 1))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy by Label')
    plt.savefig('output/validation_by_label.png')

    model = torch.load('output/model.torch')
    print('done training')
    return model, word2idx, train_acc, val_acc


def validate(model, word2idx, data, labels):
    if word2idx is None or data is None or labels is None:
        data_raw, labels_raw = load_data('val')
        if word2idx is None:
            word2idx = generateDict(data_raw)
        if data is None:
            data = create_tensors(data_raw, word2idx)
        if labels is None:
            labels = encode_labels(labels_raw)
    correct_guesses = 0
    val_bar = tqdm(total=len(data), leave=False)
    val_bar.set_description('validating')
    guess_by_label = [0] * len(labels_encoding.keys())
    for question, label in zip(data, labels):
        with torch.no_grad():
            out = model(question)
        pred = torch.argmax(out)
        correct_guesses += int(pred == label)
        guess_by_label[label] += int(pred == label)
        val_bar.update(1)
    val_bar.close()

    for l in range(len(guess_by_label)):
        guess_by_label[l] /= np.count_nonzero(labels == l)

    return correct_guesses / len(data), guess_by_label


def test(model, word2idx):
    if word2idx is None:
        train_data_raw, _ = load_data('train')
        word2idx = generateDict(train_data_raw)
    data_raw, labels_raw = load_data('test')
    data = create_tensors(data_raw, word2idx)
    labels = encode_labels(labels_raw)
    correct_guesses = 0
    val_bar = tqdm(total=len(data), leave=False)
    val_bar.set_description('testing')
    preds = np.array([])
    for question, label in zip(data, labels):
        with torch.no_grad():
            out = model(question)
        correct_guesses += int(torch.argmax(out) == label)
        preds = np.append(preds, torch.argmax(out))
        val_bar.update(1)
    val_bar.close()
    # confusion matrix and save to file
    conf_mat = metrics.confusion_matrix(labels, preds)
    conf_mat = conf_mat.astype('float') / conf_mat.astype('float').sum(axis=1)
    plt.figure()
    cm_plot = sns.heatmap(conf_mat, annot=True, fmt='g', xticklabels=labels_encoding.keys(),
                          yticklabels=labels_encoding.keys(), cmap='Blues', cbar_kws={'label': 'Proportion of Predictions'})
    cm_plot.set(xlabel='Predicted Label', ylabel='True Label', )
    cm_plot.set_title('Confusion Matrix for Test Classification')
    fig = cm_plot.get_figure()
    fig.savefig('output/confusion_matrix.png')
    return correct_guesses / len(data)


def main(params):
    model, word2idx, train_acc, val_acc = train(params)
    model = torch.load('output/model.torch')
    test_acc = test(model, word2idx)
    print("Testing accuracy: ", test_acc)
    plot_acc(train_acc, val_acc, test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int, default=500)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    main(args)
