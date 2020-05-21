#%%
'''
Quantum part, needs to generate N random quantum circuits, run over each image
and produce 
'''
from qiskit import QuantumCircuit, Aer, execute, QuantumRegister
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

#%%
SINGLE_GATE_SET = np.array(['X','Y','Z','T','S','H'])
MULTI_GATE_SET = np.array(['CNOT'])
BACKEND = Aer.get_backend('qasm_simulator')
np.random.seed(42)

def generate_random_circuit(depth, num_qubits, prob_appl_single, prob_appl_multi):
    assert prob_appl_multi > prob_appl_single, "multi gate qubit application should be less likely than single (arbitrary, could change)"
    qc = QuantumCircuit(num_qubits)
    for step in range(depth):
        for qubit in range(num_qubits):
            random_prob = np.random.uniform(0,1)
            if random_prob > 0.2:
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
''' I just realized trying do this with fashion-mnist might be a little problematic but whatever
moving on with my life 
We want to initialize the state s.t. arbitray combinations of a NxN grid of 0s and 1s map to a different quantum state
Here we try and define the convolutional filter that we need
'''

def prepare_img(filter_size,img):
    ''' Unfortunately 28*28 takes WAYY too long so having to resize image'''
    image = Image.fromarray(img).resize((14,14))
    image = np.array(image)
    image = pad_img(filter_size, image)
    return image

# Pass in mean value on the qubits
def pad_img(filter_size,img):
    if filter_size % 2 == 0:
        return np.pad(img,((1,0),(1,0)))
    else:
        return np.pad(img,1)

def conv(qc, filter_size, image, mode='threshold'):
    ''' Write the loops to slide our 'filter' over our image '''
    # here filter doesn't actually matter, we just use the flattened binary list as our init
    # might as well hard-code 3x3 filters, can happily handle 2^9 = 512 states
    prepped_img = prepare_img(filter_size, image)
    print(image.shape)
    img_height, img_width = prepped_img.shape
    conv_output = np.zeros(image.shape)
    for down_idx in range(img_height - (filter_size-1)):
        for across_idx in range(img_width  - (filter_size-1)):
            section = prepped_img[down_idx:down_idx + filter_size, across_idx: across_idx + filter_size]
            init_arr = encoding_function(section,mode)
            qc.initialize(init_arr, qc.qubits)
            job = execute(qc, BACKEND, shots=500)
            results = job.result()
            counts = results.get_counts(qc)
            output = np.zeros(len(init_arr))
            for key, value in counts.items():
                keyidx = int(key,2)
                output[keyidx] = value
            output = output/ np.sqrt(np.sum(output**2))
            entropy = shannon_entropy(output)
            conv_output[down_idx,across_idx] = entropy
    print("filter completed")
    return conv_output


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
''' Output encoding '''

def shannon_entropy(input_arr):
    return -np.sum(input_arr * np.log(input_arr+1e-11)) # offset for non-zero elements
# %%
if __name__ == '__main__':
    train_data = pd.read_csv('./fashion-mnist/fashion-mnist_train.csv')
    # qc = generate_random_circuit(depth=10,num_qubits=9,prob_appl_single=0.5,prob_appl_multi=0.8)
    # job = execute(qc, BACKEND, shots=1000)
    result = job.result()
    result.get_counts(qc)
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
