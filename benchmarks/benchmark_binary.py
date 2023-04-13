import numpy as np
import copy
import os
import sys
import time

from qiskit import Aer
from qiskit.algorithms.optimizers import L_BFGS_B

sys.path.insert(1,  os.path.dirname(sys.path[0]))
from QAOA.QAOA_binary import QAOA_Binary

def load_lattices(path, lattice_type):
    with open(path+'/lattices/mat_'+lattice_type+'.npy', 'rb') as f:
        m = np.load(f)

    print("Nb lattices: ", len(m))

    return m

def save_results(results_t, params_t, filt_t, obj_t, hist_t, path, prefix, i):
    os.makedirs(path+'/benchmarks/results', exist_ok=True)
    with open(path + '/benchmarks/results/results_bin_' + prefix +'_'+str(i)+'.npy', 'wb') as f:
        np.save(f, (results_t, params_t, filt_t, obj_t))

    with open(path + '/benchmarks/results/results_bin_' + prefix + '_' + str(i) + '_hist.npy', 'wb') as f:
        np.save(f, hist_t)

try:
    mat_index = int(sys.argv[1])
    print("User set index:", mat_index)
except:
    mat_index=-1

# Parameters
k_r = 1
grid_size = 100
p_limit = 12
save = True
lattice_type='generic_3d_10r_1k_un'
prefix = lattice_type+'_'+str(k_r)+'k'

path = os.path.dirname(sys.path[0])
mat_list=load_lattices(path, lattice_type)
optimizer = L_BFGS_B()
backend = Aer.get_backend('statevector_simulator')
qaoa = QAOA_Binary(backend)

results_t = []
filt_t = []
obj_t = []
params_t = []
hist_t = []
mat_index=0

if mat_index != -1:
    index_list = range(mat_index, mat_index+1)
else:
    index_list = range(len(mat_list))

s_time_g = time.process_time()

for i in index_list:
    B = mat_list[i]
    print("Lattice index:", i)
    print(B)

    pba = []
    results_i = []
    results_filt = []
    results_obj = []
    results_p = []
    results_hist = []

    for p in range(1, p_limit + 1):
        s_time_i = time.process_time()
        qaoa.set_parameters(B, k_r, p)
        qaoa.compute_opt_sol()
        qaoa.iter_cost = []

        if p == 1:
            print("Optimal cost:", qaoa.opt_sol)
            pba, bv = qaoa.grid_search1(grid_size)
            br_t = optimizer.minimize(qaoa.evaluate_cost1_c, pba)
            pba=br_t.x
        else:
            br_t = optimizer.minimize(qaoa.evaluate_cost, np.concatenate((pba, pba[-2:])))
            pba = br_t.x

        bve = qaoa.evaluate_cost(pba, True, True, True)
        hist_d = qaoa.evaluate_cost(pba, hist=True)
        print(hist_d)
        e_time_i = time.process_time()
        print("p:", p, "- cost:", bve, "- arg:", pba, "- iter:", qaoa.iter_count, "- time:", e_time_i-s_time_i, "s")

        results_i.append(copy.deepcopy(bve[0]))
        results_filt.append(copy.deepcopy(bve[1]))
        results_obj.append(copy.deepcopy(bve[2]))
        results_p.append(copy.deepcopy(pba))
        results_hist.append(copy.deepcopy(hist_d))

    results_t.append(copy.deepcopy(results_i))
    filt_t.append(copy.deepcopy(results_filt))
    obj_t.append(copy.deepcopy(results_obj))
    params_t.append(copy.deepcopy(results_p))
    hist_t.append(copy.deepcopy(results_hist))

    if save:
        save_results(results_t, params_t, filt_t, obj_t, hist_t, path, prefix, i)

e_time_g = time.process_time()
print("Total time:", e_time_g-s_time_g, "s")
