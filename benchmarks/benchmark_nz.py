import numpy as np
import copy
import os
import sys
import time

from qiskit import Aer
from qiskit.algorithms.optimizers import L_BFGS_B

sys.path.insert(1, os.path.dirname(sys.path[0]))
from QAOA.QAOA_nz import QAOA_NZ

def load_lattices(path, lattice_type):
    with open(path+'/lattices/mat_'+lattice_type+'.npy', 'rb') as f:
        m = np.load(f)

    print("Nb lattices:", len(m))

    return m

def save_results(results_t, params_t, filt_t, obj_t,hist_t, path, prefix, i):
    os.makedirs(path+'/benchmarks/results', exist_ok=True)
    with open(path + '/benchmarks/results/results_nz_' + prefix +'_'+str(i)+'.npy', 'wb') as f:
        a=[results_t, params_t, filt_t, obj_t]
        print(a[0])
        np.save(f, [results_t, params_t, filt_t, obj_t])

    with open(path + '/benchmarks/results/results_nz_' + prefix + '_' + str(i) + '_hist.npy', 'wb') as f:
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
single_it = False
single_rand_it = True
save = True
lattice_type='generic_3d_10r_1k_un'
prefix = lattice_type+'_'+str(k_r)+'k'

path = os.path.dirname(sys.path[0])
mat_list=load_lattices(path, lattice_type)
optimizer = L_BFGS_B()
backend = Aer.get_backend('statevector_simulator')

qaoa = QAOA_NZ(backend)
results_t = []
filt_t = []
obj_t = []
params_t = []
hist_t = []

if mat_index != -1:
    index_list = range(mat_index, mat_index+1)
else:
    index_list = range(len(mat_list))

s_time_g = time.process_time()

for i in index_list:
    B = mat_list[i]
    d = np.shape(B)[0]
    print("Lattice index:", i)
    print(B)

    if single_it:
        dd = [0]
        s_d = 1
    elif single_rand_it:
        qaoa.set_parameters(B, k_r, 1, 0)
        opt_nz_i = qaoa.compute_glob_opt_sol()
        print("Optimal non-zero index:", opt_nz_i)
        dd = [opt_nz_i]
        s_d = 1
    else:
        dd = range(d)
        s_d = d

    pba_f = [[]] * (s_d)
    bv_f = np.ones(s_d) * -1
    bve_f = np.ones(s_d) * -1
    bve_filt_f = np.ones(s_d) * -1
    bve_obj_f = np.ones(s_d) * -1
    hist_f = [[]] * (s_d)

    results_i = []
    results_p = []
    results_filt = []
    results_obj = []
    results_hist = []


    for p in range(1, p_limit + 1):
        s_time_i = time.process_time()

        for h in range(len(dd)):
            qaoa.set_parameters(B, k_r, p, dd[h])
            qaoa.range_l = 21 * 2 ** (k_r - 1)

            if p == 1:
                print("Optimal cost:", qaoa.opt_sol)
                pba_f[h], bv_f[h] = qaoa.grid_search1(grid_size)
                br_t = optimizer.minimize(qaoa.evaluate_cost1_c, pba_f[h])

            else:
                br_t = optimizer.minimize(qaoa.evaluate_cost, np.concatenate((pba_f[h], pba_f[h][-2:])))

            bve = qaoa.evaluate_cost(br_t.x, True, True, True)
            hist_f[h] = qaoa.evaluate_cost(br_t.x, hist=True)
            print(hist_f[h])

            pba_f[h] = br_t.x
            bve_f[h] = bve[0]
            bve_filt_f[h] = bve[1]
            bve_obj_f[h] = bve[2]

        e_time_i = time.process_time()

        print("p:", p, "- cost:", np.min(bve_f), ",", bve_filt_f[np.argmin(bve_f)], bve_obj_f[np.argmin(bve_f)], "- arg:",
             pba_f[np.argmin(bve_f)], "- iter:", qaoa.iter_count, "- time:", e_time_i-s_time_i, "s")

        results_i.append(copy.deepcopy(bve_f))
        results_p.append(copy.deepcopy(pba_f))
        results_filt.append(copy.deepcopy(bve_filt_f))
        results_obj.append(copy.deepcopy(bve_obj_f))
        results_hist.append(copy.deepcopy(hist_f))

    results_t.append(results_i)
    params_t.append(results_p)
    filt_t.append(results_filt)
    obj_t.append(results_obj)
    hist_t.append(copy.deepcopy(results_hist))

    if save:
        save_results(results_t, params_t, filt_t, obj_t, hist_t, path, prefix, i)

e_time_g = time.process_time()
print("Total time:", e_time_g-s_time_g, "s")
