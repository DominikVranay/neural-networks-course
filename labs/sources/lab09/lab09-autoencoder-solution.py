
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

def plot_results(x_test, encoded_imgs, decoded_imgs, n=10):
    plt.figure(figsize=(40, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i, 0])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display encoded
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(encoded_imgs[i].reshape(8, 4))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n * 2)
        plt.imshow(decoded_imgs[i, 0])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def plot_results2(encoded_imgs, decoded_imgs, n=10):
    plt.figure(figsize=(40, 4))
    for i in range(n):
        # display encoded
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(encoded_imgs[i].reshape(8, 4))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(decoded_imgs[i, 0])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.all = torch.nn.Sequential(torch.nn.Linear(32, 256),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(256, 28 * 28),
                                       torch.nn.Sigmoid())  # pridať vrstvy

    def forward(self, x):
        x = self.all(x).view(-1, 1, 28, 28)
        return x


class AutoEncoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


"""## 3. Predspracovanie údajov

Keďže dnes nebudeme používať konvolúciu, dataset musíme upraviť tak, aby obrázky boli reprezentované ako jednorozmerné vektory. Originálne MNIST dataset obsahuje obrázky *28x28*, ktoré mi prekonvertujeme na tvar *(1x)784*. Pred tým ale pixely normalizujeme, t.j. pretypujeme ich na torchovský float - pixelové hodnoty od 0 po 255 namapujeme do intervalu 0 až 1.
Taktiež si vytvoríme loadery, ktoré nám budú dávkovať dáta v batchoch.
"""

if __name__ == '__main__':
    trainset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
    testset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                         transform=torchvision.transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    encoder = torch.nn.Sequential(torch.nn.Flatten(),
                                  torch.nn.Linear(28 * 28, 256),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(256, 32))  # pridať vrstvy
    decoder = Decoder()
    model = AutoEncoder(encoder, decoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    for epoch in range(1):
        ls = []
        for inputs, _ in trainloader:
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, inputs)
            loss.backward()
            optimizer.step()
            ls.append(loss.detach().item())
        print("Loss:", sum(ls) / len(ls))
    inputs, _ = next(iter(testloader))
    encoded_imgs = encoder(inputs)
    decoded_imgs = decoder(encoded_imgs)
    # decoded_imgs = autoencoder.predict(x_test)

    plot_results(inputs, encoded_imgs.detach().numpy(), decoded_imgs.detach().numpy())

    inpu = torch.rand((300, 32))
    decoded_imgs = decoder(inpu)
    plot_results2(inpu, decoded_imgs.detach().numpy())
