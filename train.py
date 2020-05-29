#%%
import torch
from PIL import Image
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob
import re
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--quantum', type=bool, default=False)
argparser.add_argument('--display', type=bool, default=False)
argparser.add_argument('--epochs', type=int, default=10)
args = argparser.parse_args()

# %%
class CNN(nn.Module):
    def __init__(self,num_channels=1):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(num_channels, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(196, 10)
        )

    def forward(self,x):
        num_examples = x.shape[0]
        x = self.cnn_layers(x).view(num_examples,-1)
        x = self.linear_layers(x)
        #x = F.sigmoid(x)
        return x
#%%
''' CONFIG Variables '''
TEST_SIZE = 0.1
BATCH_SIZE = 100

# %%
# Can just load the whole CSV into memory since it's only 70MB
DATA = pd.read_csv('./fashion-mnist/fashion-mnist_train.csv')
labels_to_image = {
    0:"t-shirt",
    1:"trouser",
    2:"pullover",
    3:"dress",
    4:"coat",
    5:"sandal",
    6:"shirt",
    7:"sneaker",
    8:"bag",
    9:"ankle boot"
}

# %%
if not args.quantum:
    num_quantum = 3000
    num_test = 1000
    # if doing on just classical values
    X, y = DATA.values[:,1:], DATA['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    y_train_torch, y_test_torch = torch.LongTensor(y_train)[:num_quantum], torch.LongTensor(y_test)[:num_test]
    N_train = len(X_train)
    N_test = len(X_test)
    X_train = X_train / 255
    X_train_torch = torch.Tensor(X_train.reshape(N_train,1,28,28).round())
    X_test = X_test / 255
    X_test_torch = torch.Tensor(X_test.reshape(N_test,1,28,28).round())
    X_train_torch = X_train_torch[:num_quantum]
    X_test_torch = X_test_torch[:num_test]
    cnn = CNN(1)
    N_train = len(X_train_torch)
    N_test = len(X_test_torch)
else:
    print("Grabbing data")
    N_train = 3000
    N_test = 1000
    y_train_torch = torch.LongTensor(DATA['label'].values[20000:23000])
    qtraindata = np.zeros(shape=(N_train,5,28,28))
    imgs_fpath = sorted(glob.glob('./quantum_data/trainpri/*.npy'), key = lambda x : int(re.search('\d+',x)[0]))
    #load all of the quantum data
    for idx, img_fpath in enumerate(imgs_fpath):
        qtraindata[idx] = np.load(img_fpath)[:,:-1,:-1]
    imgs_fpath = sorted(glob.glob('./quantum_data/testpri/*.npy'), key = lambda x : int(re.search('\d+',x)[0]))
    #load all of the quantum data
    qtestdata = np.zeros(shape=(len(imgs_fpath),5,28,28))
    for idx, img_fpath in enumerate(imgs_fpath):
        qtestdata[idx] = np.load(img_fpath)[:,:-1,:-1]
    y_test_torch = torch.LongTensor(DATA['label'].values[23000:23000+len(qtestdata)])
    X_train_torch = torch.Tensor(qtraindata)
    X_test_torch = torch.Tensor(qtestdata)
    qdata, qtraindata = None, None # to garbage collect at some point
    cnn = CNN(5)
    


# %%
if args.display:
    plt.gray()
    print("Displaying 5 images")
    test_idx = np.random.randint(0, N_train)
    for i in range(5):
        if not args.quantum:
            test_idx = np.random.randint(0, N_train)
            test_img = X_train[test_idx].round().reshape((28,28))
            print("Label is: ",labels_to_image[y_train[test_idx]])
            plt.imshow(test_img)
            plt.title(f"Label is {labels_to_image[y_train[test_idx]]}")
            plt.show()
        else:
            test_img = X_train_torch[test_idx][i]
            plt.imshow(test_img)
            plt.title(f"Label is {labels_to_image[int(y_train_torch[test_idx])]}")
            plt.show()
    plt.close()


optim = torch.optim.Adam(cnn.parameters(),lr=0.01, weight_decay=0.01)
# %%
losses = []
test_losses = []
accs = []
test_accs = []
if args.epochs == 0:
    print("Not training")
criterion = nn.CrossEntropyLoss()
for epoch in range(args.epochs):
    for i in range(N_train // BATCH_SIZE):
        imgs = X_train_torch[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        labels = y_train_torch[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        preds = cnn(imgs)
        loss = criterion(preds,labels)
        lossrepr = float(loss)
        if i == 0: 
            print(f"INITIAL EPOCH: {epoch} LOSS: {lossrepr:.2f}")
        losses.append(lossrepr)
        optim.zero_grad()
        loss.backward()
        optim.step()
        acc = float(torch.sum(preds.argmax(dim=1) == labels)) / BATCH_SIZE
        accs.append(acc)
        test_preds = cnn(X_test_torch)
        test_loss = float(criterion(cnn(X_test_torch),y_test_torch))
        test_losses.append(test_loss)
        test_acc = float(torch.sum(test_preds.argmax(dim=1) == y_test_torch)) / N_test
        test_accs.append(test_acc)
        print(f"Train loss: {lossrepr:.4f} || Train acc: {acc:.2f}% Test loss: {test_losses[-1]:.4f} || Test acc: {test_accs[-1]:.2f}%")
    print("EPOCH:",epoch)
mode = "classical" if not args.quantum else "quantum"
print("Finally getting to save some stuff!")
np.save(f"./results/{mode}/losses.npy",np.asarray(losses))
np.save(f"./results/{mode}/test_losses.npy",np.asarray(test_losses))
np.save(f"./results/{mode}/accs.npy",np.asarray(accs))
np.save(f"./results/{mode}/testaccs.npy",np.asarray(test_accs))
