import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5), std=(0.5))])
trainset = torchvision.datasets.MNIST('/files/', train=True, download=True, transform=transform)
trainset2 = torchvision.datasets.MNIST('/files/', train=True, download=True, transform=torchvision.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
trainloader2 = torch.utils.data.DataLoader(trainset2, batch_size=128, shuffle=True)

class Encoder(nn.Module):
    '''
    simple encoder with a single hidden dense layer (ReLU activation)
    and linear projections to the diag-Gauss parameters
    '''
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim) 
        self.mu = nn.Linear(hidden_dim, embedding_dim) 
        self.sigma = nn.Linear(hidden_dim, embedding_dim) 

    
    def forward(self, x):
        out = self.fc(x)
        out = F.relu(out)
        mu = self.mu(out)
        log_var = self.sigma(out)
        return mu, log_var


class Decoder(nn.Module):
    '''
    simple decoder: single dense hidden layer (ReLU activation) followed by 
    output layer with a sigmoid to squish values
    '''
    def __init__(self, embedding_dim, dec_hidden_units, image_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, dec_hidden_units) 
        self.fc2 = nn.Linear(dec_hidden_units, image_dim) 
    
    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out
        
# Sampling function (using the reparameterisation trick)
def sample(mu, log_sigma2):
    eps = torch.randn(mu.shape[0], mu.shape[1])
    return mu + torch.exp(log_sigma2 / 2) * eps


#parameters
embedding_dim = 12
enc_hidden_units = 512
dec_hidden_units = 512
image_dim = 28**2
nEpoch = 10

# construct the encoder, decoder and optimiser
enc = Encoder(image_dim, enc_hidden_units, embedding_dim)
dec = Decoder(embedding_dim, dec_hidden_units, image_dim)
optimizer = torch.optim.Adam(list(enc.parameters())+ list(dec.parameters()), lr=1e-3)

# training loop
for epoch in range(nEpoch):
    losses = []
    for inputs, _ in trainloader:
        optimizer.zero_grad()
        inputs = inputs.view(-1, 28**2)

        mu, log_sigma2 = enc(inputs)
        z = sample(mu, log_sigma2)
        outputs = dec(z)

        # E[log P(X|z)] - as images are binary it makes most sense to use binary cross entropy
        # we need to be a little careful - by default torch averages over every observation 
        # (e.g. each  pixel in each image of each batch), whereas we want the average over entire
        # images instead
        recon = F.binary_cross_entropy(outputs, inputs, reduction='sum') / inputs.shape[0]
        
        loss = recon
        loss.backward()
        optimizer.step()

        # keep track of the loss and update the stats
        losses.append(loss.item())

    print("Epoch: ", epoch)
    for i in range(8):
        plt.subplot(241+i)
        a = torch.reshape(testset.data[i].type(torch.FloatTensor), (1, 784))
        mu, log_sigma2 = enc(a)
        z = sample(mu, log_sigma2)
        outputs = dec(z)
        outputs = torch.reshape(outputs, (28, 28)).type(torch.ByteTensor)
        plt.imshow(outputs, cmap=plt.get_cmap('gray'))

    # show the plot
    plt.show()
 
class Discriminator(nn.Module):
    '''
    simple Discriminator: single dense hidden layer (ReLU activation) followed by 
    output layer with a sigmoid to squish values
    '''
    def __init__(self, image_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(image_dim, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) 
        self.fc3 = nn.Linear(hidden_dim, 1) 

    
    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = torch.sigmoid(out)
        return out


class Generator(nn.Module):
    '''
    simple Generator: single dense hidden layer (ReLU activation) followed by 
    output layer with a sigmoid to squish values
    '''
    def __init__(self, embedding_dim, hidden_dim, image_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, image_dim) 
    
    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = torch.tanh(out)
        return out

#parameters
embedding_dim = 100
gen_hidden_units = 512
dis_hidden_units = 512
image_dim = 28**2
nEpoch = 50
device = "cuda:0"
# construct the encoder, decoder and optimiser
gen = Generator(embedding_dim, gen_hidden_units, image_dim).to(device)
dis = Discriminator(image_dim, dis_hidden_units).to(device)
lr = 0.0002 
gen_optimizer = torch.optim.Adam(gen.parameters(), lr=lr)
dis_optimizer = torch.optim.Adam(dis.parameters(), lr=lr)
criterion = torch.nn.BCELoss().to(device)

# training loop
for epoch in range(nEpoch):
    losses_dis = []
    losses_gen = []
    correct = 0
    for inputs, _ in trainloader2:
        batch = len(inputs)
        dis_optimizer.zero_grad()
        inputs = inputs.view(-1, 28**2).to(device)
        gen_inputs = torch.randn(batch, embedding_dim).to(device)
        generated = gen(gen_inputs)
        dis_inputs = torch.cat((inputs, generated), 0)
        outputs = dis(dis_inputs)
        labels = torch.cat((torch.ones(batch, 1), torch.zeros(batch, 1)), 0).to(device)
        correct += (torch.where(abs(outputs-labels)<0.5, 1, 0)).sum().item()
        loss_dis = criterion(outputs, labels)
        loss_dis.backward()
        dis_optimizer.step()

        gen_optimizer.zero_grad()
        gen_inputs = torch.randn(2*batch, embedding_dim).to(device)
        generated = gen(gen_inputs)
        outputs = dis(generated)
        loss_gen = criterion(outputs, torch.ones(2*batch, 1).to(device))
        loss_gen.backward()
        gen_optimizer.step()
        # keep track of the loss and update the stats
        losses_dis.append(loss_dis.item())
        losses_gen.append(loss_gen.item())

    print("Epoch: ", epoch, "Accuracy: ", correct/(len(trainset)*2))
    gen_inputs = torch.randn(4, embedding_dim).to(device)
    generated = gen(gen_inputs)
    for i in range(4):
        plt.subplot(141+i)
        outputs = torch.reshape(generated[i], (28, 28)).detach().cpu().numpy()
        plt.imshow(outputs, cmap=plt.get_cmap('gray'))

    # show the plot
    plt.show()