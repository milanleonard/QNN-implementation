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
c_acc = np.load('./results/classical/accs.npy')
c_tacc = np.load('./results/classical/testaccs.npy')


q_losses = np.load('./results/quantum/losses.npy')
q_tlosses = np.load('./results/quantum/test_losses.npy')
q_acc = np.load('./results/quantum/accs.npy')
q_tacc = np.load('./results/quantum/testaccs.npy')

#%%

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 9))

ax1.plot(exponential_smoothing(c_losses), label='classical')
ax1.plot(exponential_smoothing(q_losses), label='quantum')
ax1.set_xlabel('Training iteration')
ax1.set_ylabel('Loss')
ax1.legend()
# %%
ax2.plot(exponential_smoothing(c_tlosses), label='classical')
ax2.plot(exponential_smoothing(q_tlosses), label='quantum')
ax2.set_xlabel('Training iteration')
ax2.set_ylabel('Test Loss')
ax2.legend()

# %%
ax3.plot(exponential_smoothing(c_tlosses), label='classical')
ax3.plot(exponential_smoothing(q_tlosses), label='quantum')
ax3.set_xlabel('Training iteration')
ax3.set_ylabel('Test Loss')
ax3.legend()

#%%
ax4.plot(exponential_smoothing(c_tacc), label='classical')
ax4.plot(exponential_smoothing(q_tacc), label='quantum')
ax4.set_xlabel('Training iteration')
ax4.set_ylabel('Test accuracy')
ax4.legend()

plt.tight_layout()
plt.show()
# %%
