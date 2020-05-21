#%%
'''This exists because I don't want to use multiprocessing to handle
running it over the data on many machines, so I'll just manually set it off
Mac : 4 Cores at 2.5GHz
PARTCH : 6 Cores / 6 threads at 3.3Ghz, boosts to 4.5GHz, ptentially shared
PC : 6 Cores / 12 threads at 3.3GHz
'''
from quantum_pipe import generate_random_circuit, conv, prepare_img
import pandas as pd
import numpy as np
import argparse
import gc

parser = argparse.ArgumentParser()
parser.add_argument('-start_idx',type=int)
parser.add_argument('-num_datapoints',type=int)
args = parser.parse_args()

#seed means that the circuits generated are the same accross machines
np.random.seed(42)
train_data = pd.read_csv('./fashion-mnist/fashion-mnist_train.csv')
this_data = train_data[args.start_idx:args.start_idx+args.num_datapoints].drop(['label'],axis=1)
this_data = (this_data / 255).round().astype(np.uint8).values
train_data = None # this is just for garbage collector to deal with
# num qubits is square of filter size
quantum_circuits = [generate_random_circuit(depth=10,num_qubits=4,prob_appl_single=0.3,prob_appl_multi=0.7) for _ in range(2)]

# Probably want to save a numpy array for every 100 images for memory efficiency? Can come to this problem if it arises
#%%

for idx, image in enumerate(this_data):
    image = image.reshape((28,28))
    image = prepare_img(2,image)
    outputs = [conv(qc, 2, image) for qc in quantum_circuits]
    print(f"IMAGE COMPLETED: {idx+1} of {args.num_datapoints}")
    np.save(f'quantum_data/img{args.start_idx + idx}.npy',outputs)
    outputs, image = None, None
    gc.collect()
