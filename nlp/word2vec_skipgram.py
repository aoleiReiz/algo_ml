import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def random_batch(skip_grams_, batch_size_, vocab_size_):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(skip_grams_)), batch_size_, replace=False)

    for i in random_index:
        random_inputs.append(np.eye(vocab_size_)[skip_grams_[i][0]])
        random_labels.append(skip_grams_[i][1])

    return random_inputs, random_labels


class Word2Vec(nn.Module):
    def __init__(self, embedding_size_, vocab_size_):
        super(Word2Vec, self).__init__()
        self.W = nn.Linear(vocab_size_, embedding_size_, bias=False)
        self.WT = nn.Linear(embedding_size_, vocab_size_, bias=False)

    def forward(self, X):
        hidden_layer = self.W(X)
        output_layer = self.WT(hidden_layer)
        return output_layer


def main():
    batch_size = 5
    embedding_size = 2

    sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]

    word_sequence = " ".join(sentences).split()
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    voc_size = len(word_list)

    # Make skip gram of one size window
    skip_grams = []
    for i in range(1, len(word_sequence) - 1):
        target = word_dict[word_sequence[i]]
        context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
        for w in context:
            skip_grams.append([target, w])

    model = Word2Vec(embedding_size, voc_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10000):
        input_batch, target_batch = random_batch(skip_grams,batch_size, voc_size)
        input_batch = torch.Tensor(input_batch)
        target_batch = torch.LongTensor(target_batch)

        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    for i, label in enumerate(word_list):
        W, WT = model.parameters()
        x, y = W[0][i].item(), W[1][i].item()
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()


if __name__ == '__main__':
    main()