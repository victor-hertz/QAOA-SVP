import numpy as np
import copy
import os
import sys
import time

from qiskit import Aer

sys.path.insert(1,  os.path.dirname(sys.path[0]))
from asymptotics.QAOA_comp_generic import QAOA_Comp_Generic

def load_lattices(path, lattice_type):
    with open(path+'/lattices/mat_'+lattice_type+'.npy', 'rb') as f:
        m = np.load(f)

    print("Nb lattices: ", len(m))

    return m

def save_results(results, mat_index, d):
    with open(path + '/benchmarks/results/results_histas_' + type +'_'+str(nb_mat)+'mat_'+str(d)+'d_'+str(np.abs(mat_index))+'i.npy', 'wb') as f:
        np.save(f, results)

def load_results(results, mat_index, d):
    os.makedirs(path+'/benchmarks/results', exist_ok=True)
    with open(path + '/benchmarks/results/results_histas_' + type +'_'+str(nb_mat)+'mat_'+str(d)+'d_'+str(np.abs(mat_index))+'i.npy', 'wb') as f:
        return np.load(f)

try:
    user_index = int(sys.argv[1])
    print("User set index:", user_index)
except:
    user_index=-1

# Parameters
d_list=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
k_list=[1,2,3]
r=10
nb_mat=100
grid_size=100
save = True

path = os.path.dirname(sys.path[0])
lattice_type='_10r_1k_un'
prefix = lattice_type
backend = Aer.get_backend('statevector_simulator')

mat_list = []

if user_index !=-1:
    trials=1
    mat_index = user_index % nb_mat
    d_index = int(np.floor(user_index/nb_mat))
    d_chosen = d_list[d_index]

    print("Lattice index / d index / d:", mat_index, d_index, d_chosen)

path = os.path.dirname(sys.path[0])
lattice_type = type+'_'+str(d_chosen)+'d_'+str(r)+'r_1k_un'
B =  load_lattices(path, lattice_type)[mat_index]
results = load_results(mat_index, d_chosen)

beta= np.linspace(0, np.pi, int(grid_size / 2))
gamma = np.linspace(0, 2*np.pi, int(grid_size / 2))
beta_rs = beta

for k_index in range(len(k_list)):
    k = k_list[k_index]
    qaoa_generic = QAOA_Comp_Generic(d_chosen, k, r)
    rescale = qaoa_generic.scale
    gamma_rs = gamma/rescale

    r_asympt = qaoa_generic.ham_gauss_asympt(beta, gamma)
    r_argmin = np.nanargmin(r_asympt)
    x, y = r_argmin // r_asympt.shape[1], r_argmin % r_asympt.shape[1]
    beta_a = beta_rs[x]
    gamma_a = gamma_rs[y]
    min_a = r_asympt[r_argmin]

    print("k / d:", k, d_chosen)
    print(B)

    qaoa_generic = QAOA_Comp_Generic(d_chosen, k, r)

    # Simulation
    print("Rescale factor: ", rescale)
    i_time = time.process_time()
    r_sim = qaoa_generic.ham_sim(beta_rs, gamma_rs, False, 1, B, offset=True, do_scale=False)
    r_argmin = np.nanargmin(r_sim)
    x, y = r_argmin// r_sim.shape[1], r_argmin % r_sim.shape[1]
    beta_s = beta_rs[x]
    gamma_s = gamma_rs[y]
    min_s = r_asympt[r_argmin]

    print("Sim. evaluation time:", time.process_time() - i_time)
    qaoa_generic.qaoa.set_parameters(B, k, 0)

    print("Cost - sim. / predicted:", min_s, min_a)
    print("Beta / Gamma - sim. / predicted", beta_s, gamma_s, beta_a, gamma_a/qaoa_generic.scales)
    beta_s = results[k]
    results.append(copy.deepcopy([beta_s, gamma_s, beta_a, gamma_a, min_s, min_a]))

if save and user_index!=-1:
    save_results(results, mat_index, d_chosen)
