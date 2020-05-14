#%%
import torch
from PIL import Image
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
plt.gray()
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
X, y = DATA.values[:,1:], DATA['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
y_train_torch, y_test_torch = torch.LongTensor(y_train), torch.LongTensor(y_test)
N_train = len(X_train)
N_test = len(X_test)
# %%
if False:
    print("Displaying 10 images")
    for _ in range(10):
        test_idx = np.random.randint(0, N_train)
        test_img = X_train[test_idx].reshape((28,28))
        print("Label is:",labels_to_image[y_train[test_idx]])
        plt.imshow(test_img)
        plt.show()
# %%
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
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
        return x

# %%
X_train = X_train / 255
X_train_torch = torch.Tensor(X_train.reshape(N_train,1,28,28))
X_test = X_test / 255
X_test_torch = torch.Tensor(X_test.reshape(N_test,1,28,28))

# %%
cnn = CNN()
optim = torch.optim.Adam(cnn.parameters(),lr=0.01)
# %%
losses = []
epoch_loss = []
criterion = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    for i in range(N_train // BATCH_SIZE):   
        imgs = X_train_torch[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        labels = y_train_torch[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        preds = cnn(imgs)
        loss = criterion(preds,labels)
        lossrepr = float(loss)
        if i == 0: print(f"INITIAL EPOCH: {epoch} LOSS: {lossrepr:.2f}"); epoch_loss.append(float(loss))
        losses.append(lossrepr)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % 50 == 0: print(f"{lossrepr:.2f}")
    test_acc = float(criterion(cnn(X_test_torch),y_test_torch))
    print(f"TEST ACCURACY: {test_acc:.2f}")
np.save("./results/classical/losses.npy",np.asarray(losses))
# %%
print(epoch_loss)

# %%
