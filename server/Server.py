import socket
import struct
import pickle
import numpy as np
import json
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from threading import Thread
import time

#load data from json file
f = open('parameter_server.json', )
data = json.load(f)

#set parameters fron json file
host = data["host"]
port = data["port"]
max_recv = data["max_recv"]
lr = data["learningrate"]
update_treshold = data["update_threshold"]
max_numclients = data["max_nr_clients"]


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
        x = self.t_convb(x)
        return x


class Server(nn.Module):
    """
    server model
    """
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


class initial_Client(nn.Module):
    """
        initial client model with the actual weights
    """
    def __init__(self):
        super(initial_Client, self).__init__()
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


def send_msg(sock, content):
    """
    pickles the content (creates bitstream), adds header and send message via tcp port

    :param sock: socket
    :param content: content to send via tcp port
    """
    msg = pickle.dumps(content)
    msg = struct.pack('>I', len(msg)) + msg  # add 4-byte length in netwwork byte order
    sock.sendall(msg)


def recieve_msg(sock):
    """
    recieves the meassage with helper function, unpickles the message and separates 
    the getid from the actual massage content
    calls the request handler
    :param
        sock: socket
    :return: none
    """
    msg = recv_msg(sock)  # receive client message from socket
    msg = pickle.loads(msg)
    getid = msg[0]
    content = msg[1]

    handle_request(sock, getid, content)


def recv_msg(sock):
    """
    gets the message length (which corresponds to the first for bytes of the recieved bytestream) with the recvall function

    :param
        sock: socket
    :return: returns the data retrieved from the recvall function
    """
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    return recvall(sock, msglen)


def recvall(sock, n):
    """
    returns the data from a recieved bytestream, helper function to receive n bytes or return None if EOF is hit
    :param sock: socket
    :param n: length in bytes (number of bytes)
    :return: message
    """
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def handle_request(sock, getid, content):
    """
    executes the requested function, depending on the get id, and passes the recieved message

    :param sock: socket
    :param getid: id of the function, that should be executed if the message is recieved
    :param content: message content
    """
    switcher = {
        0: calc_gradients,
        1: get_testacc,
        2: updateclientmodels,
    }
    switcher.get(getid, "invalid request recieved")(sock, content)


def get_testacc(conn, msg):
    """
    this method does the forward propagation with the recieved data, from to first layer of the decoder to the last layer
    of the model. It sends information about loss/accuracy back to the client.

    :param conn: connection
    :param msg: message
    """
    with torch.no_grad():
        client_output_test, label_test = msg['client_output_test'], msg['label_test']
        client_output_test, label_test = client_output_test.to(device), label_test.to(device)
        client_output_test = decode(client_output_test)  #
        client_output_test = client_output_test.clone().detach().requires_grad_(True)
        output_test = server(client_output_test)
        loss_test = error(output_test, label_test)
        test_loss = loss_test.data
        correct_test = torch.sum(output_test.argmax(dim=1) == label_test).item()

    msg = {"test_loss": test_loss,
           "correct_test": correct_test,
           }
    send_msg(conn, msg)


def updateclientmodels(sock, updatedweights):
    """
    send the actual clientside weights to all connected clients, 
    except from the clint that is currently training

    :param sock: the socket
    :param updatedweights: the client side weghts with actual status
    """
    client.load_state_dict(updatedweights)
    for clientss in connectedclients:
        try:
            if clientss != sock:
                send_msg(clientss, updatedweights)
        except:
            pass


def calc_gradients(conn, msg):
    """
    this method does the forward propagation with the recieved data, 
    from to first layer of the decoder to the last layer
    of the model. it calculates the loss, and does the backward propagation up to the 
    cutlayer of the model.
    Depending on if the loss threshold is reached it sends the gradient of the back 
    propagation at the cut layer and
    information about loss/accuracy/trainingtime back to the client.

    :param conn: the connected socket of the currently training client
    :param msg: the recieved data
    """
    start_time_training = time.time()
    optimizer.zero_grad()
    with torch.no_grad():
        client_output_train, label_train, batchsize, batch_concat = msg['client_output_train'], msg['label_train'], msg[
            'batchsize'], msg['batch_concat']  # client output tensor
        client_output_train, label_train = client_output_train.to(device), label_train
        client_output_train = decode(client_output_train)
    splittensor = torch.split(client_output_train, batchsize, dim=0)

    dc = 0
    while dc < batch_concat:
        tenss = splittensor[dc]
        tenss = tenss.requires_grad_(True)
        tenss = tenss.to(device)

        output_train = server(tenss)  # forward propagation
        with torch.no_grad():
            lbl_train = label_train[dc].to(device)
        loss_train = error(output_train, lbl_train)  # calculates cross-entropy loss
        train_loss = loss_train.data
        loss_train.backward()  # backward propagation
        client_grad = tenss.grad.clone().detach()
        optimizer.step()
        add_correct_train = torch.sum(output_train.argmax(dim=1) == lbl_train).item()
        add_total_train = len(lbl_train)
        total_training_time = time.time() - start_time_training
	
	
        if train_loss.item() > update_treshold:
            pass
        else:
            client_grad = "abort"

        msg = {"grad_client": client_grad,
               "train_loss": train_loss,
               "add_correct_train": add_correct_train,
               "add_total_train": add_total_train,
               "active_trtime_batch_server": total_training_time,
               }
        print("Create the msg")
        send_msg(conn, msg)
        print("Send the msg back")
        dc += 1


def initialize_client(conn):
    """
    called when new client connect. if new connected client is not the first connected 
    client, the send the initial weights to
    the new connected client
    :param conn:
    """
    print("connected clients: ",len(connectedclients))
    if len(connectedclients) == 1:
        msg = 0
    else:
        initial_weights = client.state_dict()
        msg = initial_weights
    send_msg(conn, msg)


def clientHandler(conn, addr):
    initialize_client(conn)
    while True:
        try:
            recieve_msg(conn)
        except:
            print("No message, wait!")
            pass


connectedclients = []
trds = []


def main():
    """
    initialize device, server model, initial client model, optimizer, loss, decoder and accepts new clients
    """
    print(torch.version.cuda)
    global device
    device = 'cpu' 
    #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if (torch.cuda.is_available()):
        print("training on gpu")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    global server
    server = Server()
    server.to(device)

    global client
    client = initial_Client()
    client.to(device)
    print("initial_Client complete.")

    global optimizer
    optimizer = SGD(server.parameters(), lr=lr, momentum=0.9)

    global error
    error = nn.CrossEntropyLoss()
    print("Calculate CrossEntropyLoss complete.")
	
    global decode
    decode = Decode()
    decode.load_state_dict(torch.load("./convdecoder.pth"))
    decode.eval()
    decode.to(device)
    print("Load decoder parameters complete.")

    s = socket.socket()
    s.bind((host, port))
    s.listen(max_numclients)
    print("Listen to client reply.")
	
    for i in range(max_numclients):
        conn, addr = s.accept()
        connectedclients.append(conn)
        print('Conntected with', addr)
        t = Thread(target=clientHandler, args=(conn, addr))
        print('Thread established')
        trds.append(t)
        t.start()
        print('Thread start')

    for t in trds:
        t.join()


if __name__ == '__main__':
    main()
