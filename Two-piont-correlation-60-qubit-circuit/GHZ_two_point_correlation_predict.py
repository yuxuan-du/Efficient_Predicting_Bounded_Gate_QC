'''This file collects the code of predicting the two-point correlation of 60-qubit GHZ states on any pair of qubit'''

import numpy as np
import h5py
from pennylane import classical_shadow, shadow_expval, ClassicalShadow
import pennylane as qml
import itertools
import os

seed = np.random.seed(123)
dim_feature = 3
observable = None
test_data = np.random.uniform(-np.pi, np.pi, dim_feature) # a particular input for prediction
print('The test point is {}'.format(test_data))


### Load training data, Trigo representations, and (estimated) label with the specified backend

# Generate the frequency set with the specified truncations
frequence_index_single = [0, 1, -1]
frequence_index_all_qubits = []
[frequence_index_all_qubits.append(frequence_index_single) for i in range(3)] # non-truncation by setting 3
frequence_list = list(itertools.product(*frequence_index_all_qubits))


# precomputing the length of each frequency in the frequency set
f1 = lambda x: np.cos(x)
f2 = lambda x: np.sin(x)

lentgh_freq_list = []
for index_list in frequence_list:
    index_list = np.asarray(index_list)
    lentgh_freq_list.append(np.count_nonzero(index_list))

def single_data_point_generation_TriMKernel(params):
    map_freq_coefficients = np.zeros(shape=(3, dim_feature)) # Compute Phi_w;The order of each column corresponds to [0, 1, -1]
    data = np.reshape(params, newshape=(-1))
    map_freq_coefficients[0, :] = 1
    map_freq_coefficients[1, :] = f1(data)
    map_freq_coefficients[2, :] = f2(data)


    # Use frequence_list and map_freq_coefficients to compute all Phi_w(theta)
    data_feature = []
    for index_list in frequence_list:
        index_list = np.asarray(index_list)
        coefficients = [map_freq_coefficients[index_list[i] ,i] for i in range(dim_feature)]
        freq_coff_all = np.prod(coefficients)
        data_feature.append(freq_coff_all)
    return data_feature

# load data and calculate shadow estimations for each training point
file_dir = 'dataset/data_60_qubits/shots1000/'
num_data = 500
n_qubits = 60
X_train = np.zeros(shape=(num_data, 3**dim_feature)) # num_data x dim of classical control
trace_est = np.zeros(shape=(num_data, 1)) # record the shadow estimation
two_point_est_array = np.zeros(shape=(n_qubits, n_qubits, num_data))
obs = qml.PauliZ(0) @ qml.PauliY(59)


for i in range(num_data):
    index_ = i + 1
    f = open(file_dir + 'angles_' + str(index_) + '.txt', 'r')
    data_tempt = f.read().split()
    data_tempt = np.reshape(data_tempt, newshape = (dim_feature))
    data_tempt = data_tempt.astype(float)
    X_train[i] = single_data_point_generation_TriMKernel(data_tempt)
    hf = h5py.File(file_dir + 'qst_circuit_'+ str(index_) + '.h5', 'r')
    GHZ_shadow_bases = hf.get('bases')[:]
    GHZ_shadow_bases = np.transpose(GHZ_shadow_bases)
    dic_map_basis = {b'X':0, b'Y':1, b'Z':2} # Map bases to the format acceptable by Pennylane
    GHZ_recipes = np.vectorize(dic_map_basis.get)(GHZ_shadow_bases)
    GHZ_shadow_res = hf.get('outcomes')[:]
    GHZ_shadow_res = np.transpose(GHZ_shadow_res)
    shadow = ClassicalShadow(GHZ_shadow_res, GHZ_recipes)
    for qubit_i in range(n_qubits):
        for qubit_j in range(n_qubits):
            obs_two_point = (qml.PauliZ(qubit_i) @ qml.PauliZ(qubit_j) + qml.PauliY(qubit_i) @ qml.PauliY(qubit_j) + qml.PauliX(qubit_i) @ qml.PauliX(qubit_j))/3
            two_point_est_array[qubit_i, qubit_j, i] = shadow.expval(obs_two_point)

print('data loading is finished')

# Compute classical representations
def Tri_geo_kernel_pre_compute(train_data_, y_train_, truncation_length):
    r'''
     :param train_data: data feature of train set
     :param y_train: label of training dataset
     :return: a representation vector obtained from traiing data
     '''
    n_train = len(y_train_)
    representation_train_data = np.zeros(shape=(3** dim_feature))
    for i in range(n_train):
        # compute the kernel W(x,xl)=sum_w 2^|w| Phi_w(x) Phi_w(x')
        for j in range(3 ** (dim_feature)):
            if lentgh_freq_list[j] <= truncation_length:
                representation_train_data[j] += (2 ** lentgh_freq_list[j]) * train_data_[i][j] * y_train_[i]
    representation_train_data = representation_train_data / n_train
    return representation_train_data

trunc_length = 3
prediction_array = np.zeros(shape=(n_qubits, n_qubits))
for qubit_i in range(n_qubits):
    for qubit_j in range(n_qubits):
        representation_train_correlator_ij = Tri_geo_kernel_pre_compute(X_train,  two_point_est_array[qubit_i, qubit_j, :], truncation_length= trunc_length)
        rep_test_data = single_data_point_generation_TriMKernel(test_data)
        prediction_array[qubit_i, qubit_j] = np.inner(representation_train_correlator_ij, rep_test_data)

np.save('prediction_two_point_correlation', prediction_array)
print(prediction_array)
### start to prediction


