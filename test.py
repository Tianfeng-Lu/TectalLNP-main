#%%
import numpy as np
n_neuron = 14597
tmp = np.random.normal(0, 1, (n_neuron, n_neuron)).astype(np.float32)
tmp2 = np.random.normal(0, 1, (1, 5)).astype(np.float32)
spike_cells = np.random.choice(n_neuron, 10000, replace=False)
#%% slowest
%%timeit
arr1 = (tmp[:, spike_cells] @ np.ones((len(spike_cells), 1), dtype=np.float32) @ tmp2)

#%% best
%%timeit
mask = np.zeros((n_neuron, 1), dtype=np.float32)
mask[spike_cells] = 1
arr2 = (tmp @ mask @ tmp2)

#%% index the row
%%timeit
arr3 = (tmp2.T @ np.ones((1, len(spike_cells)), dtype=np.float32) @ tmp.T[spike_cells, :]).T

#%%
%%timeit
arr4 = np.sum(tmp[:, spike_cells], axis=1, keepdims=True, dtype=np.float32) @ tmp2
#%%
arr1 = (tmp[:, spike_cells] @ np.ones((len(spike_cells), 1), dtype=np.float32) @ tmp2)
mask = np.zeros((n_neuron, 1), dtype=np.float32)
mask[spike_cells] = 1
arr2 = (tmp @ mask @ tmp2)
arr3 = (tmp2.T @ np.ones((1, len(spike_cells)), dtype=np.float32) @ tmp.T[spike_cells, :]).T

np.allclose(arr1, arr2, atol=1e-3)

#%%
%%timeit
w_ex = np.zeros((n_neuron, n_neuron), dtype=np.float32)

for i in range(n_neuron):
    w_ex[i, :] = np.exp(-tmp[i]**2/2)

#%%
%%timeit
## why this is so slow? 
w_ex = np.exp(-tmp**2/2)
