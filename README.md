# Improving the Communication and Computation Efficiency of Split Learning for IoT Applications
## More Efficient Split Learning Method (MESL)
This is the repository containing the code for this paper "Improving the Communication and Computation Efficiency of Split Learning for IoT Applications" accepted in "2021 IEEE Global Communications Conference"


## Requirements

### Server
* Python
#### Packages
* socket
* struct
* pickle
* numpy
* json
* torch
* threading
* time

### Client
* Python
#### Packages
* socket
* struct
* pickle
* time
* json
* numpy
* matplotlib
* torch
* torchvision
* sys


## Simple usage

Here's a brief overview of how you can use this project to run split learning on a server and a client.

### Download the dataset
The GTSRB dataset can be downloaded from 
[link](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html). download and unpack the zip file. Copy the folders "testset" and "trainingset" into the ./server/GTSRB/GTSRB directory in the project.

### Set client side configurations

To set the client side configuraitons, the file parameter_client.json can be edited.  

The initial settings are:

```json
{
    "training_epochs": 50,
    "batchsize": 64,
    "learningrate": 0.0002,

    "batch_concat": 1,

    "host": "192.168.0.0",
    "port": 10087,
    "max_recv": 4096
}

```

To adjust the training parameter, the number of training_epochs and the bacthsize can be set as integer values and the learningrate as a float value.
Additionally you can set how many batches should be concatinated for the forwardpass through the autoencoder (encoder and decoder).

To configure the connection, "host" needs to be set to thge ip adderess of the server and the port, to the port that will be configured at the server.

The max_recv property is the bit rate of the tcp port.

### Set serverside configurations

To set the server side configuraitons, the file parameter_server.json can be edited.  

The initial settings are:
```json
{
    "learningrate": 0.0002,
    "update_threshold": 0.0,
    "max_nr_clients": 5,

    "host": "0.0.0.0",
    "port": 10087,
    "max_recv": 4096
}
```
One can set the server side learningrate manually (in all the configurations of the corresponding paper, the same learning rate for server and client was chosen).
Also, the update_threshold can be configured (for batches above this threshold, gradients are not sent back to the client --> client side update dismissed)

max_nr_clients is the maximal number of clients, that can be connected to the server at the same time.

To configer the server, the host is initially set to "0.0.0.0" (localhost).
Also the Port has to be set.

The max_recv property is the bit rate of the tcp port.

### Running the model

To run the model, the server.py script has to be run the server. Afterwards, the client.py script needs to be run at the client




## Advanced settings

To do advanced configurations, plase follow the instructions in the document advanced_settings.pdf

## Author
Ahmad Ayad, Melvin Renner, Zhenghang Zhong
