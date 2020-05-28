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
plt.gray()

argparser = argparse.ArgumentParser()
argparser.add_argument('--quantum', type=bool, default=False)
argparser.add_argument('--display', type=bool, default=False)
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
BATCH_SIZE = 128
EPOCHS = 10

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
    # if doing on just classical values
    X, y = DATA.values[:,1:], DATA['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    y_train_torch, y_test_torch = torch.LongTensor(y_train), torch.LongTensor(y_test)
    N_train = len(X_train)
    N_test = len(X_test)
    X_train = X_train / 255
    X_train_torch = torch.Tensor(X_train.reshape(N_train,1,28,28).round())
    X_test = X_test / 255
    X_test_torch = torch.Tensor(X_test.reshape(N_test,1,28,28).round())
    cnn = CNN(1)
else:
    print("Grabbing data")
    N_train = 10000
    N_test = 10000
    y_train_torch = torch.LongTensor(DATA['label'].values[:10000])
    test_idxs = np.random.choice(range(10000,60000),10000)
    y_test_torch = torch.LongTensor(DATA['label'].values[test_idxs])
    X_test_torch = DATA.values[test_idxs,1:] / 255
    X_test_torch = torch.Tensor((X_test_torch.reshape(N_test,1,28,28).round()))
    qdata = np.zeros(shape=(10000,5,28,28))
    imgs_fpath = sorted(glob.glob('./quantum_data/*.npy'), key = lambda x : int(re.search('\d+',x)[0]))
    #load all of the quantum data
    for idx, img_fpath in enumerate(imgs_fpath):
        qdata[idx] = np.load(img_fpath)[:,:-1,:-1]
    X_train_torch = torch.Tensor(qdata)
    qdata=None # to garbage collect at some point
    cnn = CNN(5)
    


# %%
if args.display:
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


optim = torch.optim.Adam(cnn.parameters(),lr=0.01)
# %%
losses = []
epoch_loss = []
test_losses = []
criterion = nn.CrossEntropyLoss()
acc_sum = 0
total = 0
for epoch in range(EPOCHS):
    for i in range(10000 // BATCH_SIZE):
        total += BATCH_SIZE   
        imgs = X_train_torch[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        labels = y_train_torch[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        preds = cnn(imgs)
        acc_sum += torch.sum(preds.argmax(dim=1) == labels)
        loss = criterion(preds,labels)
        lossrepr = float(loss)
        if i == 0: 
            print(f"INITIAL EPOCH: {epoch} LOSS: {lossrepr:.2f}"); epoch_loss.append(float(loss))
        losses.append(lossrepr)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % 10 == 0: 
            print(f"{lossrepr:.2f} || {float(acc_sum)/total:.2f}%"); acc_sum=0; total = 0
            if not args.quantum:
                test_loss = float(criterion(cnn(X_test_torch),y_test_torch))
            else:
                tmp = X_test_torch
                for i in range(4):
                    tmp = torch.cat((tmp,X_test_torch),dim=1)
                test_loss = float(criterion(cnn(tmp),y_test_torch))
            print(f"TEST LOSS: {test_loss:.2f}")
            test_losses.append(test_loss)
    print("EPOCH:",epoch)
mode = "classical" if not args.quantum else "quantum"
np.save(f"./results/{mode}/losses.npy",np.asarray(losses))
np.save(f"./results/{mode}/test_losses.npy",np.asarray(test_losses))
