import numpy as np
import pickle

# file_path = '/allen/programs/braintv/workgroups/cortexmodels/brianh/exps/tensor-decomp/cifar10_wide-resnet28-1_lr0.1_mmt0.9_Wd0.0001_NoAug'
file_path = '/allen/programs/braintv/workgroups/cortexmodels/brianh/exps/tensor-decomp/cifar10_corrupt1_wide-resnet28-1_lr0.1_mmt0.9_Wd0.0001_NoAug'

array_total = []
for epoch in range(1, 301):
    array = pickle.load(
        open(file_path+'/act_epoch_' + str(epoch) + '.pkl', 'rb'))
    array_neuron = []
    for key, val in array.items():
        array_neuron.append(val)
    array_neuron = np.concatenate(array_neuron, axis=1)
    array_total.append(array_neuron)

# stack and shape into neuron x stim x repeat
array_total = np.stack(array_total, axis=2)
array_total = np.transpose(array_total, (1, 0, 2))
