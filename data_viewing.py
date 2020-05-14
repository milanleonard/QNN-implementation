#%%
import torch
from PIL import Image
import numpy as np
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#%%
''' CONFIG Variables '''
test_size = 0.1


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
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = test_size)
N_train = len(X_train)
# %%
print("Displaying 10 images")
for _ in range(10):
    test_idx = np.random.randint(0, 60000)
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
            nn.Linear(4 * 7 * 7, 10)
        )

    def forward(self,x):
        x = self.cnn_layers(x)
        x = self.linear_layers(x)

# %%
X_train = X_train / 255
X_train_torch = torch.Tensor(X_train.reshape(60000,1,28,28))


# %%
