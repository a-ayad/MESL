# Advanced client side configurations

## Dataset
The dataset can be customized. The Data trandformation, as well as the dataset itself can be adjusted in the client.py file 

```python
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3403, 0.3121, 0.3214),
                         (0.2724, 0.2608, 0.2669))
])


# Create Datasets
from client.GTSRB import gtsrb_dataset as dataset
trainset = dataset.GTSRB(root_dir='.\GTSRB', train=True,  transform=transform)
testset = dataset.GTSRB(root_dir=".\GTSRB", train=False,  transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)
```

## Host configureation
By default, Host IP and Port are read from the parameter_client.json file. The paramter can also be set directly in the client.py file. 

```python
host = data["host"]
port = data["port"]
```

## Training parameter configuration
By default, Training and tcp paramters (epochs, learningratem batchsize, batchconcat, max recv) are read from the parameter_client.json file. The paramter can also be set directly in the client.py file, by changing the following variables.
```python
epoch = data["trainingepochs"]
lr = data["learningrate"]
batchsize = data["batchsize"]
batch_concat = data["batch_concat"]
max_recv = data["max_recv"]
```
Further, device, optimizer and lossfunction can be set int the main method of the client.py file.

```python
global device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

global optimizer
optimizer = SGD(client.parameters(), lr=lr, momentum=0.9)

global error
error = nn.CrossEntropyLoss()
```
## Model
The model of the client can also be modified in the cleint.py file in the Client Class. 
Attention: the input shape has to match the dataset and the outputshape the encoder/ server side model!
```python
class Client(nn.Module):
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
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.norm2(x)
        x = self.drop1(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.norm3(x)
        return x
```

## Costumize Autoencoder
Customizing the autoencoder is explaines in the advanced_settings file.

