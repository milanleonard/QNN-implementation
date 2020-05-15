#%%
'''
Quantum part, needs to generate N random quantum circuits, run over each image
and produce 
'''
from qiskit import QuantumCircuit, Aer, execute, QuantumRegister
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
SINGLE_GATE_SET = np.array(['X','Y','Z','T','S','H'])
MULTI_GATE_SET = np.array(['CNOT'])
train_data = pd.read_csv('./fashion-mnist/fashion-mnist_train.csv')
#%%
def generate_random_circuit(depth, num_qubits, prob_appl_single, prob_appl_multi):
    assert prob_appl_multi > prob_appl_single, "multi gate qubit application should be less likely than single (arbitrary, could change)"
    qc = QuantumCircuit(qr)
    for step in range(depth):
        for qubit in range(num_qubits):
            random_prob = np.random.uniform(0,1)
            if random_prob > 0.1:
                qc.h(qubit) # let's do lots of hadamards
            if random_prob > prob_appl_multi:
                # handles two qubit operations, should extend to N qubits for Toffoli at some point
                valid_choices = list(range(qubit)) + list(range(qubit + 1,num_qubits))
                gate = np.random.choice(MULTI_GATE_SET)
                other_qubit = np.random.choice(valid_choices)
                handle_gate(qc, gate, qubit, other_qubit) 
            elif random_prob > prob_appl_single:
                gate = np.random.choice(SINGLE_GATE_SET)
                handle_gate(qc, gate, qubit)
            else:
                continue
                #qc.i(qubit)
    qc.measure_all()
    return qc
#%%
# Maybe make this a subfunction of generate random circuit method? so can nonlocal qubit
def handle_gate(qc, gate, qubit, other_qubit=0):
    """
    Takes a str of a gate and a quantum circuit and adds the gate
    to a random qubit at the depth of the circuit
    Just a bookkeeping function
    """
    if gate == "CNOT":
        qc.cx(qubit,other_qubit)
    if gate == "X":
        qc.x(qubit)
    if gate == "Y":
        qc.y(qubit)
    if gate == "Z":
        qc.z(qubit)
    if gate == "T":
        qc.t(qubit)
    if gate == "S":
        qc.s(qubit)
    if gate == "H":
        qc.h(qubit)

# %%
backend = Aer.get_backend('qasm_simulator')
qc = generate_random_circuit(depth=10,num_qubits=4,prob_appl_single=0.5,prob_appl_multi=0.8)
job = execute(qc, backend, shots=1000)
result = job.result()
result.get_counts(qc)

# %%
''' Now the goal is to init the state threshholded on our dummy input, let's pick out
an input to play with'''

#currently not being clever with my threshold at all just rounding 
dummy_example = train_data.values[0,1:].reshape(28,28) / 255
dummy_example_rounded = np.round(dummy_example)
plt.gray()
print("Pre thresholding")
plt.imshow(dummy_example)
plt.show()
print("Post thresholding")
plt.imshow(dummy_example_rounded)
plt.show()
# %%
''' I just realized trying do this with fashion-mnist might be a little problematic but whatever
moving on with my life 
We want to initialize the state s.t. arbitray combinations of a NxN grid of 0s and 1s map to a different quantum state
Here we try and define the convolutional filter that we need
'''
padded_dummy = np.pad(dummy_example_rounded,1)
def conv_(qc, filter_size, image, mode='threshold'):
    ''' Write the loops to slide our 'filter' over our image '''
    # here filter doesn't actually matter, we just use the flattened binary list as our init
    # might as well hard-code 3x3 filters, can happily handle 2^9 = 512 states

    img_height, img_width = image.shape
    for down_idx in range(img_height - (filter_size-1)):
        for across_idx in range(img_width  - (filter_size-1)):
            section = image[down_idx:down_idx + filter_size, across_idx: across_idx + filter_size]
            # TODO NEED TO ADD QUANTUM BIT HERE
            init_arr = encoding_function(section,mode)
            qc.initialize(init_arr, qc.qubits)
            job = execute(qc, backend, shots=1000)
            results = job.result()
            counts = results.get_counts(qc)
            # THIS BIT BELOW IN INCORRECT, NEED TO INDEX INTO 2**n LONG VECTOR
            sorted_vec = np.array([counts[key] for key in sorted(counts)])
            sorted_vec = sorted_vec/ np.sqrt(np.sum(sorted_vec**2))
            print(sorted_vec)
            break
        break


# %%
def encoding_function(section,mode):
    '''
    Takes a section of the image and returns a list
    that sets the initialization state 
    '''
    num_qubits = section.size
    out_arr = np.zeros(2**num_qubits)
    if mode == 'threshold':
        # I suspect this is a stupid way to do this but whatever
        flattened_section = section.flatten().astype(np.uint8) # 1 or 0 array, can store cheaply
        idx = int("".join([str(x) for x in flattened_section]),2)
        out_arr[idx] = 1
    return out_arr




# %%
