import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas


def generate_real():
    real_data = torch.FloatTensor(
        [random.uniform(0.8, 1.0),
         random.uniform(0.0, 0.2),
         random.uniform(0.8, 1.0),
         random.uniform(0.0, 0.2)])
    return real_data


def generate_random(size):
    random_data = torch.rand(size)
    return random_data


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(4, 3),
            nn.Sigmoid(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

        self.loss_function = nn.MSELoss()

        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        # simply run model
        return self.model(inputs)

    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)
        # increase counter and accumulate error every 10 epochs
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter % 10000 == 0):
            print("counter = ", self.counter)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1, 3),
            nn.Sigmoid(),
            nn.Linear(3, 4),
            nn.Sigmoid()
        )

        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, D: Discriminator, inputs, targets):
        g_output = self.forward(inputs)
        d_output = D.forward(g_output)

        loss = D.loss_function(d_output, targets)

        self.counter += 1
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


def train_discriminator():
    print("Random noise pattern:", generate_random(4))
    D = Discriminator()

    for i in range(10000):
        # real data
        D.train(generate_real(), torch.FloatTensor([1.0]))
        # fake data
        D.train(generate_random(4), torch.FloatTensor([0.0]))

    D.plot_progress()
    plt.savefig('output/legend.png')

    print("Real data source:", D.forward(generate_real()).item())

    print("Random noise:", D.forward(generate_random(4)).item())


D = Discriminator()
G = Generator()

# train Discriminator and Generator

for i in range(10000):
    # train discriminator on true
    D.train(generate_real(), torch.FloatTensor([1.0]))

    # train discriminator on false
    # use detach() so gradients in G are not calculated
    D.train(G.forward(torch.FloatTensor([0.5])).detach(), torch.FloatTensor([0.0]))

    # train generator
    G.train(D, torch.FloatTensor([0.5]), torch.FloatTensor([1.0]))



D.plot_progress()
plt.xlabel('Discriminator loss chart')
plt.savefig('output/Discriminator.png')