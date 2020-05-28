#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
def exponential_smoothing(input_arr, alpha=0.1):
    xs = np.zeros(len(input_arr))
    xs[0] = input_arr[0]
    for idx, x in enumerate(input_arr[1:]):
        xs[idx+1] = alpha*x + (1-alpha)*xs[idx]
    return xs
# %%
c_losses = np.load('./results/classical/losses.npy')
c_tlosses = np.load('./results/classical/test_losses.npy')

q_losses = np.load('./results/quantum/losses.npy')
q_tlosses = np.load('./results/quantum/test_losses.npy')

#%%
x = np.linspace(0,10,780)
plt.plot(exponential_smoothing(c_losses), label='classical')
plt.plot(exponential_smoothing(q_losses), label='quantum')
plt.xlabel('Training iteration')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.show()
# %%
xt = np.linspace(0,10,80)
plt.plot(xt,exponential_smoothing(c_tlosses), label='classical')
plt.plot(xt,exponential_smoothing(q_tlosses), label='quantum')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.legend()
plt.show()

# %%
