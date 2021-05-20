# Advanced server side configurations

## Host configureation
By default, Host IP and Port are read from the parameter_client.json file. The paramter can also be set directly in the server.py file. Server IP should alwas be "0.0.0.0" (local host)

```python
host = data["host"]
port = data["port"]
```

## Training parameter configuration
By default, Training and tcp paramters (learningratem batchsize, update threshold, max recv) are read from the parameter_server.json file. The paramter can also be set directly in the server.py file, by changing the following variables.
```python
max_recv = data["max_recv"]
lr = data["learningrate"]
update_treshold = data["update_threshold"]
max_numclients = data["max_nr_clients"]
```
Further, device, optimizer and lossfunction can be set int the main method of the server.py file.

```python
global device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

global optimizer
optimizer = SGD(client.parameters(), lr=lr, momentum=0.9)

global error
error = nn.CrossEntropyLoss()
```
## Model
The model of the client can also be modified in the server.py file in the Server Class. 
Attention: the input shape has to match the dataset and the outputshape the decoder/ client side model!
```python
class Server(nn.Module):
    def __init__(self):
        super(Server, self).__init__()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.relu4 = nn.ReLU()
        self.norm4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.drop2 = nn.Dropout2d(0.3)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
        self.relu5 = nn.ReLU()
        self.norm5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
        self.relu6 = nn.ReLU()
        self.norm6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.drop3 = nn.Dropout2d(0.4)
        self.linear1 = nn.Linear(in_features=128, out_features=43, bias=True)

    def forward(self, x):
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.norm4(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.norm5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.norm6(x)
        x = self.pool3(x)
        x = self.drop3(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.log_softmax(self.linear1(x), dim=1)
        return x
```
## Costumize Autoencoder
Customizing the autoencoder is explaines in the advanced_settings file.

