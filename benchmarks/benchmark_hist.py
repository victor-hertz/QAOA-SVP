import numpy as np
import copy
import os
import sys
import time

from qiskit import Aer

sys.path.insert(1,  os.path.dirname(sys.path[0]))
from QAOA.QAOA_binary import QAOA_Binary
from QAOA.QAOA_nz import QAOA_NZ
from QAOA.QAOA_dpenalty import QAOA_dPenalty
from QAOA.QAOA_xy import QAOA_XY

def load_lattices(path, lattice_type):
    with open(path+'/lattices/mat_'+lattice_type+'.npy', 'rb') as f:
        m = np.load(f)

    print("Nb lattices:", len(m))

    return m

def load_results(path, i, lattice_type, encoding_type):
    with open(path+'/benchmarks/results/results_hist_'+encoding_type+'_' + lattice_type +'_'+str(i)+'.npy', 'rb') as f:
        results = np.load(f, allow_pickle=True)[1][0]

    return results

def save_results(results, i):
    os.makedirs(path+'/benchmarks/results', exist_ok=True)
    with open(path + '/benchmarks/results/results_hist_'+str(i)+'.npy', 'wb') as f:
        np.save(f, results)

try:
    mat_index = int(sys.argv[1])
    print("User set index:", mat_index)
except:
    mat_index=-1
    
# Parameters
k_r = 1
grid_size = 100
p_limit = 12
lattice_type_prefix='_10r_1k_un'
save = True

path = os.path.dirname(sys.path[0])
backend = Aer.get_backend('statevector_simulator')

for mat_index in range(0,250+1):
    print("Index:", mat_index)

    results = []
    s_time_g = time.process_time()

    for i in range(5,6):
        mat_list = []
        lattice_type = str(i)+'d'+lattice_type_prefix
        mat_list.append(load_lattices(path, 'generic_'+lattice_type)[mat_index])
        mat_list.append(load_lattices(path, 'cyclic_'+lattice_type)[mat_index])

        q_bin = [QAOA_Binary(backend), load_results(path, mat_index, 'generic_'+lattice_type,'bin'), load_results(path, mat_index, 'cyclic_'+lattice_type,'bin'), ]
        q_dpen = [QAOA_dPenalty(backend), load_results(path, mat_index, 'generic_'+lattice_type,'dpen'), load_results(path, mat_index, 'cyclic_'+lattice_type,'dpen')]
        q_nz = [QAOA_NZ(backend), load_results(path, mat_index, 'generic_'+lattice_type,'nz'), load_results(path, mat_index, 'cyclic_'+lattice_type,'nz')]
        q_xy = [QAOA_XY(backend), load_results(path, mat_index, 'generic_'+lattice_type,'xy'), load_results(path, mat_index, 'cyclic_'+lattice_type,'xy')]

        results_type = []

        for j in range(2):

            B = mat_list[j]

            if(j == 0):
                print("Generic")
            else:
                print("Cyclic")

            print(B)

            results_p = []

            for p in range(1, p_limit + 1):
                s_time_i = time.process_time()

                q_bin[0].set_parameters(B, k_r, p)
                q_dpen[0].set_parameters(B, k_r, p)
                q_dpen[0].toggle_penalty(False)
                q_nz[0].set_parameters(B, k_r, 1, 0)
                opt_nz_i = q_nz[0].compute_glob_opt_sol()
                q_nz[0].set_parameters(B, k_r, p,opt_nz_i)
                q_xy[0].set_parameters(B, k_r, p)
                r_bin = q_bin[0].evaluate_cost(q_bin[j+1][p-1], hist=True)
                r_dpen = q_dpen[0].evaluate_cost(q_dpen[j+1][p-1], hist=True)
                r_nz = q_nz[0].evaluate_cost(q_nz[j+1][p-1][0], hist=True)
                r_xy = q_xy[0].evaluate_cost(q_xy[j+1][p-1], hist=True)

                results_p.append([copy.deepcopy(r_bin), copy.deepcopy(r_dpen), copy.deepcopy(r_nz), copy.deepcopy(r_xy)])
                print("d:", i, "- Type:", j, "- p:", p,"- Time:", time.process_time()-s_time_i, "s")

            results_type.append([copy.deepcopy(q_bin[0].cost_m), copy.deepcopy(q_dpen[0].cost_m), copy.deepcopy(q_nz[0].cost_m), copy.deepcopy(q_xy[0].cost_m),copy.deepcopy(results_p)])
        results.append(copy.deepcopy(results_type))
    e_time_g = time.process_time()
    print("Total time:", e_time_g-s_time_g, "s")

    save_results(results, mat_index)
