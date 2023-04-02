import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np


def plot_results(x_test, encoded_imgs, decoded_imgs, n=10):
    plt.figure(figsize=(40, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i])
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
        plt.imshow(decoded_imgs[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


trainset = torchvision.datasets.MNIST('/files/', train=True, download=True, transform=torchvision.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
testset = torchvision.datasets.MNIST('/files/', train=True, download=True, transform=torchvision.transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

encoder = torch.nn.Sequential(torch.nn.Flatten,
                              torch.nn.Linear(28*28, 256),
                              torch.nn.ReLU(),
                              torch.nn.Linear(256, 32)) # pridať vrstvy
class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.all = torch.nn.Sequential(torch.nn.Linear(32, 256),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(256, 28*28),
                                       torch.nn.Sigmoid()) # pridať vrstvy
        
    
    def forward(self, x):
        x = self.all(x).view(-1, 28, 28)
        return x
decoder = Decoder()

class AutoEncoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = AutoEncoder(encoder, decoder)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()
for epoch in range(100):
    ls = []
    for inputs, _ in trainloader:
        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, inputs)
        loss.backward()
        optimizer.step()
        ls.append(loss.detach().item())
    print("Loss:", sum(ls)/len(ls))

inputs, _ = next(testloader)
encoded_imgs = encoder(inputs)
decoded_imgs = decoder(encoded_imgs)

plot_results(inputs, encoded_imgs, decoded_imgs)

def plot_results(encoded_imgs, decoded_imgs, n=10):
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
        plt.imshow(decoded_imgs[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
inpu = np.random.random((300, 32))
decoded_imgs = decoder(inpu)
plot_results(inpu, decoded_imgs)
