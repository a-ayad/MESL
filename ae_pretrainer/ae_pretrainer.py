import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD
import torchvision
from torchvision import transforms
import torch

batchsize = 64
test_batches = 1500
epoch = 250
lr = 0.0001

prev_loss = 999
diverge_tresh = 1.1
lr_adapt = 0.5

# Create Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Load Datasets
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batches, shuffle=False, num_workers=2)
total_batch = len(train_loader)



class Encode(nn.Module):
    """
    encoder model
    """
    def __init__(self):
        super(Encode, self).__init__()
        self.conva = nn.Conv2d(64, 16, 3, padding=1)
        self.convb = nn.Conv2d(16, 8, 3, padding=1)
        self.convc = nn.Conv2d(8, 4, 3, padding=1)##

    def forward(self, x):
        x = self.conva(x)
        x = self.convb(x)
        x = self.convc(x)
        return x

class Decode(nn.Module):
    """
    decoder model
    """
    def __init__(self):
        super(Decode, self).__init__()
        self.t_convx = nn.ConvTranspose2d(4, 8, 1, stride=1)
        self.t_conva = nn.ConvTranspose2d(8, 16, 1, stride=1)
        self.t_convb = nn.ConvTranspose2d(16, 64, 1, stride=1)

    def forward(self, x):
        x = self.t_convx(x)
        x = self.t_conva(x)
        x= self.t_convb(x)
        return x


# just the part of the clientmidel before the autoencoder
class Client(nn.Module):
    """
    client model
    """
    def __init__(self):
        super(Client, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout2d(0.2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
        self.relu3 = nn.ReLU()
        self.norm3 = nn.BatchNorm2d(64)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu3(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.norm3(x)
        return x


def start_training():
    """
    function, which does the training process for the autoencoder and seves the weights at the end of every epoch,
    if no divergenvce occurs

    --> adaptive learning rate, depending on divergences --> in case of divergence, results of the epoch are dismissed,
    and the training of this epoch restarts with a smaller learning rate

    """
    for c, batch_test in enumerate(test_loader):
        x_tester, label_tester = batch_test
        break
    prevloss = prev_loss
    learnrate = lr###
    optimizerencode = Adam(encode.parameters(), lr=learnrate)###
    optimizerdecode = Adam(decode.parameters(), lr=learnrate)###

    for e in range(epoch):
        x_test, label_test = x_tester, label_tester
        x_test, label_test = x_test.to(device), label_test.to(device)
        for b, batch in enumerate(train_loader):
                x_train, label_train = batch
                x_train, label_train = x_train.to(device), label_train.to(device)
                #print(x_train.shape)
                x_train = client(x_train)
                output_train = encode(x_train)
                output_train = decode(output_train)

                optimizerencode.zero_grad()
                optimizerdecode.zero_grad()
                loss_train = error(x_train, output_train)  # calculates cross-entropy loss
                loss_train.backward()

                optimizerencode.step()
                optimizerdecode.step()

        with torch.no_grad():
            x_test = client(x_test)
            output_test = encode(x_test)
            output_test = decode(output_test)
            loss_test = error(x_test, output_test).data
        print("epoch: {}, train-loss: {:.6f}, test-loss: {:.6f} ".format(e + 1,  loss_train, loss_test))


        if loss_test <= prevloss*diverge_tresh:
            torch.save(encode.state_dict(), "./convencoder.pth")
            encode2 = Encode()
            encode2.load_state_dict(torch.load("./convencoder.pth"))
            encode2.eval()

            torch.save(decode.state_dict(), "./convdecoder.pth" )
            prevloss = loss_test

        else:
            print ("diverge")
            encode.load_state_dict(torch.load("./convencoder.pth"))
            encode.eval()
            encode.to(device)
            decode.load_state_dict(torch.load("./convdecoder.pth"))
            decode.eval()
            decode.to(device)
            #break
            learnrate = lr_adapt*learnrate
            print(learnrate)
            optimizerencode = Adam(encode.parameters(), lr=learnrate)###
            optimizerdecode = Adam(decode.parameters(), lr=learnrate)###


def main():
    """
    initialize device, client model, encoder and decoder and starts the training process
    """

    global device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # cuda:0
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    global error
    error = nn.MSELoss()

    global client
    client = Client()
    client.to(device)

    global encode
    encode = Encode()
    encode.to(device)

    global decode
    decode = Decode()
    decode.to(device)

    start_training()


if __name__ == '__main__':
    main()