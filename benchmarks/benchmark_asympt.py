import numpy as np
import os
import sys
import copy
import time

from qiskit.algorithms.optimizers import L_BFGS_B

sys.path.insert(1,  os.path.dirname(sys.path[0]))
from asymptotics.QAOA_comp_generic import QAOA_Comp_Generic

def load_lattices(path, lattice_type):
    with open(path+'/lattices/mat_'+lattice_type+'.npy', 'rb') as f:
        m = np.load(f)

    print("Nb lattices:", len(m))

    return m

def save_results(results, mat_index, d):
    os.makedirs(path+'/asymptotics/results', exist_ok=True)
    with open(path + '/asymptotics/results/results_' + type +'_'+str(nb_mat)+'mat_'+str(d)+'d_'+str(np.abs(mat_index))+'i.npy', 'wb') as f:
        np.save(f, results)

def adjust_format(b,g):
    return b,g
    gp = (-g) % (2*np.pi)
    bp = (-b) % np.pi

    if np.abs(gp)<np.abs(g) and np.abs(bp)<np.abs(b):
        return bp, gp
    return b,g

try:
    user_index = int(sys.argv[1])
    print("User set index:", user_index)
except:
    user_index=-1

# Parameters
d_list=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
k_list=[1,2,3]
nb_mat = 100
r=10
type = 'generic'
grid_size=100
use_opt=True
save=True

print("Nb mat:", nb_mat, "- r:", r)

if user_index !=-1:
    trials=1
    mat_index = user_index % nb_mat
    d_index = int(np.floor(user_index/nb_mat))
    d_chosen = d_list[d_index]

    print("Lattice index / d index / d:", mat_index, d_index, d_chosen)

path = os.path.dirname(sys.path[0])
lattice_type = type+'_'+str(d_chosen)+'d_'+str(r)+'r_1k_un'
B =  load_lattices(path, lattice_type)[mat_index]

optimizer = L_BFGS_B(maxfun=1000)

beta= np.linspace(0, np.pi, int(grid_size / 2))
gamma = np.linspace(0, 2*np.pi, int(grid_size / 2))
beta_rs = beta

t_time = time.process_time()
rng = np.random.default_rng(int(time.time()))
results_k = []

beta_r = 0
gamma_r = 0
min_r = 0

for k in k_list:
    qaoa_generic = QAOA_Comp_Generic(d_chosen, k, r)

    r_asympt = qaoa_generic.ham_gauss_asympt(beta, gamma)
    r_argmin = np.nanargmin(r_asympt)
    x, y = r_argmin // r_asympt.shape[1], r_argmin % r_asympt.shape[1]
    b0, g0 = adjust_format(beta[y], gamma[x])

    if use_opt:
        opt_r = optimizer.minimize(qaoa_generic.ham_gauss_asympt_opt, [b0, g0])
        beta_a = opt_r.x[0]
        gamma_a = opt_r.x[1]
        beta_a, gamma_a = adjust_format(beta_a, gamma_a)
    else:
        beta_a = beta[y]
        gamma_a = gamma[x]
        beta_a, gamma_a = adjust_format(beta_a, gamma_a)
    print(B)
    print("k/d:", k, d_chosen)

    qaoa_generic = QAOA_Comp_Generic(d_chosen, k, r)

    # Simulation
    rescale=1
    print("Rescale:", rescale)
    gamma_rs = np.linspace(0, 2 * np.pi, int(grid_size *qaoa_generic.scales/2))

    i_time = time.process_time()
    r_sim = qaoa_generic.ham_sim(beta_rs, gamma_rs, False, 1, B, offset=True, do_scale=True)
    r_argmin = np.nanargmin(r_sim)
    x, y = r_argmin// r_sim.shape[1], r_argmin % r_sim.shape[1]
    print("Simulation eval time:", time.process_time() - i_time)
    qaoa_generic.qaoa.set_parameters(B, k, 0)

    if use_opt:
        i_time = time.process_time()
        b0, g0 = adjust_format(beta_rs[y], gamma_rs[x])
        opt_r = optimizer.minimize(qaoa_generic.ham_sim_opt, [b0, g0])
        beta_s = opt_r.x[0]
        gamma_s = opt_r.x[1]
        beta_s, gamma_s = adjust_format(opt_r.x[0],opt_r.x[1])

    else:
        beta_s = beta_rs[y]
        gamma_s = gamma_rs[x]
        beta_s, gamma_s = adjust_format(beta_rs[y], gamma_rs[x])

    min_s = qaoa_generic.ham_sim(beta_s, gamma_s, False, 1, B, offset=False, do_scale=True)
    min_a = qaoa_generic.ham_sim(beta_a, gamma_a, False, 1, B, offset=False)

    print("sim/asympt:", min_s, min_a)
    print("beta/gamma - sim/asympt", beta_s, gamma_s, beta_a, gamma_a)

    results_k.append(copy.deepcopy([beta_s, gamma_s, beta_r, gamma_r, beta_a, gamma_a, min_s, min_r, min_a]))

if save and user_index!=-1:
    save_results(results_k, mat_index, d_chosen)

print("Total time:", time.process_time()-t_time, "s")
