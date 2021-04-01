import struct
import socket
import pickle
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import SGD
from torchvision import transforms


f = open('parameter_client.json', )
data = json.load(f)

epoch = data["trainingepochs"]
lr = data["learningrate"]
batchsize = data["batchsize"]

#value on server has to be the same
batch_concat = data["batch_concat"]

host = data["host"]
port = data["port"]
max_recv = data["max_recv"]


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
total_batch = len(train_loader)
total_batch_train = len(train_loader)
total_batch_test = len(test_loader)




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
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.norm2(x)
        x = self.drop1(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.norm3(x)
        #x = encode(x)
        return x




def send_msg(sock, getid, content):
    """
    pickles the content (creates bitstream), adds header and send message via tcp port

    :param sock: socket
    :param content: content to send via tcp port
    """
    msg = [getid, content]  # add getid
    msg = pickle.dumps(msg)
    msg = struct.pack('>I', len(msg)) + msg  # add 4-byte length in network byte order
    #print("communication overhead send: ", sys.getsizeof(msg), " bytes")
    sock.sendall(msg)


def recieve_msg(sock):
    """
    recieves the meassage with helper function, umpickles the message and separates the getid from the actual massage content
    :param sock: socket
    """

    msg = recv_msg(sock)  # receive client message from socket
    msg = pickle.loads(msg)
    return msg

def recv_msg(sock):

    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    return recvall(sock, msglen)


def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def start_training(s):
    """
    actuall function, which does the training from the train loader and testing from the tesloader epoch/bacth wise
    :param s: socket
    """
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    train_losses.append(0)
    train_accs.append(0)
    test_losses.append(0)
    test_accs.append(0)

    start_time_training = time.time()
    for e in range(epoch):
        train_loss = 0.0
        correct_train, total_train = 0, 0
        loss_test = 0.0
        correct_test, total_test = 0,0
        active_training_time_epoch_client = 0
        active_training_time_epoch_server = 0

        concat_counter_send = 0
        concat_counter_recv = 0

        batches_aborted, total_train_nr, total_test_nr = 0, 0, 0

        concat_tensors = []
        concat_labels = []
        epoch_start_time = time.time()
        for b, batch in enumerate(train_loader):
            active_training_time_batch_client = 0
            start_time_batch_forward = time.time()
            x_train, label_train = batch
            x_train, label_train = x_train.to(device), label_train.to(device)
            optimizer.zero_grad()

            # batch concat
            if concat_counter_send < batch_concat:
                concat_tensors.append(client(x_train))
                concat_labels.append(label_train)
                concat_counter_send +=1
                continue
            else:
                client_output_train = concat_tensors[0]
                for k in range(batch_concat-1):
                    client_output_train = torch.cat((client_output_train, concat_tensors[k+1]), 0)

            client_output = client_output_train.clone().detach().requires_grad_(False)

            with torch.no_grad():
                client_output = encode(client_output)

            msg = {
                'client_output_train': client_output,
                'label_train': concat_labels,
                'batch_concat' : batch_concat,
                'batchsize' : batchsize
            }

            active_training_time_batch_client += time.time()-start_time_batch_forward
            send_msg(s, 0, msg)

            concat_labels = []

            while concat_counter_recv < concat_counter_send:
                msg = recieve_msg(s)
                start_time_batch_backward = time.time()

                client_grad = msg["grad_client"]
                if client_grad == "abort":
                    train_loss_add, add_correct_train, add_total_train = msg["train_loss"], msg["add_correct_train"],  msg["add_total_train"]
                    correct_train += add_correct_train
                    total_train_nr += 1#
                    total_train += add_total_train
                    train_loss += train_loss_add
                    batches_aborted += 1
                    pass
                else:
                    concat_tensors[concat_counter_recv].to(device)
                    concat_tensors[concat_counter_recv].backward(client_grad)
                    optimizer.step()
                    train_loss_add, add_correct_train, add_total_train = msg["train_loss"], msg["add_correct_train"], msg["add_total_train"]

                    correct_train += add_correct_train
                    total_train_nr += 1
                    total_train += add_total_train
                    train_loss += train_loss_add

                concat_counter_recv += 1

                active_training_time_batch_client += time.time() - start_time_batch_backward
                active_training_time_batch_server = msg["active_trtime_batch_server"]
                active_training_time_epoch_client += active_training_time_batch_client
                active_training_time_epoch_server += active_training_time_batch_server

            concat_counter_send = 0
            concat_counter_recv = 0
            concat_tensors = []

        epoch_endtime = time.time() - epoch_start_time
        with torch.no_grad():
                for b_t, batch_t in enumerate(test_loader):

                    x_test, label_test = batch_t
                    x_test, label_test = x_test.to(device), label_test.to(device)
                    optimizer.zero_grad()
                    output_test = client(x_test)
                    client_output_test = output_test.clone().detach().requires_grad_(True)
                    client_output_test = encode(client_output_test)

                    msg = {'client_output_test': client_output_test,
                           'label_test': label_test,
                           }
                    send_msg(s, 1, msg)
                    msg = recieve_msg(s)

                    correct_test_add = msg["correct_test"]
                    test_loss = msg["test_loss"]
                    loss_test += test_loss
                    correct_test += correct_test_add
                    total_test_add = len(label_test)
                    total_test += total_test_add
                    total_test_nr +=1

        initial_weights = client.state_dict()
        send_msg(s, 2, initial_weights)

        train_losses.append(train_loss / total_train_nr)
        train_accs.append(correct_train / total_train)
        test_losses.append(loss_test / total_test_nr)
        test_accs.append(correct_test / total_test)



        status_epoch = "epoch: {}, train-loss: {:.4f}, train-acc: {:.2f}%, test-loss: {:.4f}, test-acc: {:.2f}%, trainingtime for epoch: {:.6f}s, batches abortrate:{:.2f}  ".format(
            e + 1, train_loss / total_train_nr, (correct_train / total_train) * 100, loss_test / total_test_nr,
            (correct_test / total_test) * 100, epoch_endtime,  batches_aborted/total_train_nr)
        print(status_epoch)

    total_training_time = time.time() - start_time_training
    time_info = "trainingtime for {} epochs: {:.2f}s".format(epoch, total_training_time)
    print(time_info)
    plot(test_accs, train_accs, train_losses, test_losses)




def plot(test_accs, train_accs, train_losses, test_losses):
    """
    plots the accuracy and loss for thraining and test over the epochs
    :param test_accs: list with test accuracies
    :param train_accs: list with trainings accuracies
    :param train_losses: list with trainings losses
    :param test_losses: list with test accuracies
    :return:
    """
    plt.subplot(211)
    plt.axis([1, max(len(train_losses), len(test_losses))-1, 0, 3])
    #plt.xticks(range(1, max(len(train_losses), len(test_losses))))
    plt.plot(train_losses, 'b')
    plt.plot(test_losses, 'r')
    plt.ylabel('loss')
    plt.xlabel("epochs")

    plt.subplot(212)
    plt.axis([1, max(len(train_accs), len(test_accs))-1, 0.2, 1])
    #plt.xticks(range(1, max(len(train_losses), len(test_losses))))
    plt.plot(train_accs, 'b')
    plt.plot(test_accs, 'r')
    plt.ylabel('accuracy')
    plt.xlabel("epochs")

    plt.show()



def initialize_model(s):
    """
    if new connected client is neot the first connected client, the initial weights are fetched from the server
    :param conn:
    """
    msg = recieve_msg(s)
    if msg == 0:
        pass
    else:
        client.load_state_dict(msg, strict=False)
        print("model successfully initialized")
    start_training(s)

def main():
    """
    initialize device, client model, optimizer, loss and decoder and starts the training process
    """

    global device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if(torch.cuda.is_available()):
        print("training on gpu")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    global client
    client = Client()
    client.to(device)

    global optimizer
    optimizer = SGD(client.parameters(), lr=lr, momentum=0.9)

    global error
    error = nn.CrossEntropyLoss()

    global encode
    encode = Encode()
    encode.load_state_dict(torch.load("./convencoder.pth"))
    encode.eval()
    encode.to(device)

    s = socket.socket()
    s.connect((host, port))
    initialize_model(s)



if __name__ == '__main__':
    main()