import numpy as np
from itertools import product
import copy
from scipy.signal import argrelextrema

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

class QAOA_Simulation:
    def __init__(self, backend, sim_iter=250, range_l=21):
        self.B = np.array([])
        self.k_r=-1
        self.iter_count=0
        self.backend = backend
        self.sim_iter = sim_iter
        self.range_l = range_l
        self.iter_cost = []

        if self.backend.name() == 'statevector_simulator':
            self.evaluate_cost = self.evaluate_cost_sv
        else:
            self.evaluate_cost = self.evaluate_cost_iter

    def omega(self, i):
        raise NotImplementedError

    def alpha(self, i, j):
        raise NotImplementedError

    def set_k(self, k_r):
        raise NotImplementedError

    def pos(self, i, j):
        return i * (self.k + 1) + j

    def h_f(self, i, l):
        val = 0

        for j in range(self.d):
            val += self.G[i, j] * self.omega(j)

        val *= 2 * self.alpha(i, l)

        return val

    def j_f(self, i, l, j, p):
        factor = 2
        if i == j and l == p:
            factor = 1
        return factor * self.G[i, j] * self.alpha(i, l) * self.alpha(j, p)

    def compute_offset(self):
        self.offset = 0

        for i in range(self.d):
            for j in range(i, self.d):
                factor = 2
                if i == j:
                    factor = 1
                self.offset += factor * self.G[i, j] * self.omega(i) * self.omega(j)

        for x in range(self.d):
          for u in range(self.k+1):
              self.offset += self.j_f(x,u,x,u)

    def compute_interactions(self):
        self.h_m = np.zeros(self.N)
        self.j_m = np.zeros((self.N, self.N))
        self.j_ms = np.zeros((self.N, self.N))

        for x in range(self.d):
            for u in range(self.k + 1):
                self.h_m[self.pos(x, u)] = self.h_f(x, u)

                for y in range(self.d):
                    for v in range(self.k + 1):
                        self.j_m[self.pos(x, u), self.pos(y, v)] = self.j_f(x, u, y, v)

                        if self.pos(y, v) > self.pos(x, u):
                            self.j_ms[self.pos(x, u), self.pos(y, v)] = self.j_f(x, u, y, v)

    def init_circ(self):
        init = QuantumCircuit(self.N)

        for i in range(self.N):
            init.h(i)

        return init

    def cost_ham_circ(self, l):
        cost_h = QuantumCircuit(self.N)

        for i in range(self.d):
            for p in range(self.k + 1):
                cost_h.rz(2 * self.h_f(i, p) * self.params[2 * l + 1], self.pos(i, p))

        for i in range(self.d):
            for p in range(self.k + 1):
                for j in range(self.d):
                    for q in range(self.k + 1):
                        if self.pos(j, q) > self.pos(i, p):
                            cost_h.rzz(2 * self.j_f(i, p, j, q) * self.params[2 * l + 1], self.pos(i, p),
                                       self.pos(j, q))

        return cost_h

    def mixer_ham_circ(self, l):
        mixer_h = QuantumCircuit(self.N)

        for i in range(self.N):
            mixer_h.rx(-2 * self.params[2 * l], i)

        return mixer_h

    def set_parameters(self, B, k_r, p, G=[0], force_init=False):
        if force_init or (not np.array_equal(B, self.B)) or k_r != self.k_r:
            self.opt_sol = -1
            self.B = np.array(B)
            if len(np.array(G)) == 1:
                self.G = np.dot(np.transpose(B), B)
            else:
                self.G = G

            self.d = np.shape(B)[0]
            self.k_r = k_r
            self.set_k(k_r)
            self.N = self.d * (self.k + 1)
            self.p = p

            self.compute_interactions()
            self.compute_offset()

            if p == 0:
                return 0

            self.compute_binary_list()
            self.compute_cost_fn()
            self.compute_obj_fn()

            try:
                self.compute_filter_fn()
            except:
                pass
        else:
            self.p=p

        self.iter_count=0
        self.params = []

        for i in range(self.p):
            self.params.append(Parameter('b_' + str(i)))
            self.params.append(Parameter('g_' + str(i)))

        self.qc = QuantumCircuit(self.N)

        self.qc.append(self.init_circ(), [i for i in range(self.N)])

        for l in range(self.p):
            self.qc.append(self.cost_ham_circ(l), [i for i in range(self.N)])
            self.qc.append(self.mixer_ham_circ(l), [i for i in range(self.N)])

        self.qc = self.qc.decompose()

        return self.qc

    def grid_search1(self, g_size):
        if self.p != 1:
            raise Exception("Wrong p value: ", self.p)

        h=0
        b_res_b = -1
        while(h < int(self.range_l*self.N)):
            for i, gamma_value in enumerate(np.linspace(np.pi/(self.range_l*self.N)*h, np.pi/(self.range_l*self.N)*(h+1), int(g_size / 2))):

                b_res = -1
                for j, beta_value in enumerate(np.linspace(0, np.pi, int(g_size / 2))):
                    res = self.evaluate_cost((beta_value, gamma_value))

                    if res <= b_res or b_res == -1:
                        b_res = res
                        b_b_t = beta_value

                if b_res_b == -1 or b_res_b > b_res:
                    b_res_b = copy.deepcopy(b_res)
                    b_g = copy.deepcopy(gamma_value)
                    b_b = copy.deepcopy(b_b_t)

                elif b_res_b < b_res:
                    return [b_b, b_g], b_res_b

            print("Search extension: ", h)
            h+=1

        print('Error, no minimum found')
        raise RuntimeError

    def evaluate_grid1(self, g_size, scale=1, offset=0):
        exp_values = np.empty((int(g_size/2), int(g_size / 2)))
        beta = np.linspace(0, np.pi, int(g_size / 2))
        gamma = np.linspace(0, np.pi/scale+offset, int(g_size / 2))

        for i, gamma_value in enumerate(np.linspace(0, np.pi/scale+offset, int(g_size / 2))):
            for j, beta_value in enumerate(np.linspace(0, np.pi, int(g_size / 2))):
                exp_values[i][j] = self.evaluate_cost((beta_value, gamma_value))

        return exp_values, beta, gamma

    def compute_cost_fn(self):
        cost_z = 0
        cost_zz = 0
        v = (1 - 2 * self.bl)
        cost_z += np.dot(v, self.h_m)
        cost_zz += np.sum(np.dot(v, self.j_ms) * v, axis=1)
        self.cost_m = (cost_z + cost_zz + self.offset)
        self.z_m = (self.cost_m).astype(int) == 0

    def compute_binary_list(self):
        l = np.arange(2 ** self.N)
        self.bl = (l.reshape(-1, 1) & (2 ** np.arange(self.N)) != 0).astype(int)

    def evaluate_cost(self, params_val, filter_enabled=False, obj_enabled=False):
        raise NotImplementedError

    def compute_filter_fn(self):
        raise NotImplementedError

    def compute_obj_fn(self):
        if self.opt_sol == -1:
            self.compute_opt_sol()

        self.obj_m = (np.isclose(self.cost_m,self.opt_sol)).astype(int)

    def evaluate_cost_sv(self, params_val, filter_enabled=False, filter_rate=False, obj_enabled=False, hist=False):
        self.iter_count+=1
        dic = {}

        for i in range(self.p):
            dic[self.params[2 * i]] = params_val[2 * i]
            dic[self.params[2 * i + 1]] = params_val[2 * i + 1]
        qc2 = self.qc.bind_parameters(dic)

        job = self.backend.run(qc2)
        sv = job.result().get_statevector(qc2)
        pr = sv.probabilities()

        if hist:
            return [self.opt_sol, pr]
        if obj_enabled:
            pr_obj = np.dot(pr, self.obj_m)

        if filter_enabled:
            pr_z = np.dot(pr, self.z_m)
            pr_filter = np.dot(pr, self.filter_m)
            cost = np.dot(self.cost_m, pr * self.filter_m)/pr_filter
        else:
            cost = np.dot(self.cost_m, pr)

        self.iter_cost.append(cost)

        if filter_rate:
            if obj_enabled:
                return cost, pr_z, pr_obj
            else:
                return cost, pr_z
        else:
            if obj_enabled:
                return cost, pr_obj
            else:
                return cost

        return cost

    def evaluate_cost_iter(self, params_val, filter_enabled=False, filter_rate=False, obj_enabled=False):
        self.iter_count += 1
        dic = {}

        for i in range(self.p):
            dic[self.params[2 * i]] = params_val[2 * i]
            dic[self.params[2 * i + 1]] = params_val[2 * i + 1]

        qc2 = self.qc.bind_parameters(dic)
        qc2.measure_all()
        job = self.backend.run(qc2, shots=self.sim_iter)
        counts = job.result().get_counts()
        counts_filter = 0
        counts_obj = 0
        cost = 0

        for c in counts.keys():
            cc = c[::-1]

            m = []
            for i in cc:
                m.append(int(i))

            r = self.compute_cost_fn()

            if obj_enabled and self.obj_fn(r):
                counts_obj += counts[c]

            if (filter_enabled and self.filter_fn(r, m)) or not filter_enabled:
                cost += r * counts[c]

                if filter_enabled:
                    counts_filter += counts[c]

        if filter_enabled:
            cost /= counts_filter
        else:
            cost /= self.sim_iter

        if filter_rate:
            if obj_enabled:
                return cost, counts_filter / self.sim_iter, counts_obj / self.sim_iter
        else:
            if obj_enabled:
                return cost, counts_obj / self.sim_iter
            else:
                return cost

    def qudit_value(self, i):
        v = self.omega(i)
        for l in range(self.k + 1):
            v += self.alpha(i, l) * (1 - 2 * self.bl[:,self.pos(i, l)])
        return v

    def get_feasible_ranges(self):
        raise NotImplemented

    def compute_opt_sol(self):
        nz_cost_m = self.cost_m.astype(int)
        nz_cost_m = nz_cost_m[np.nonzero(nz_cost_m)]
        self.opt_sol = np.min(nz_cost_m)
        st = np.where(self.cost_m == self.opt_sol)

    def compute_opt_sol_old(self):
        s_n = -1
        l = self.get_feasible_ranges()
        prod = product(*l)
        for i in prod:
            n = np.dot(i, np.dot(self.G, np.transpose(i)))
            if (s_n == -1 or n < s_n) and n != 0:
                s_n = copy.deepcopy(n)

        self.opt_sol = s_n

class QAOA1_Analytical(QAOA_Simulation):
    def __init__(self, backend, sim_iter=250):
        super().__init__(backend, sim_iter)

    def z(self, i, l):
        return self.z1(i, l) * np.sin(2 * self.beta)

    def z1(self, i, l):
        val = -self.h_m[self.pos(i, l)] * np.sin(2 * self.gamma * self.h_m[self.pos(i, l)])

        for j in range(self.d):
            for p in range(self.k + 1):
                if i != j or l != p:
                    val *= np.cos(2 * self.gamma * self.j_m[self.pos(i, l), self.pos(j, p)])

        return val

    def zz(self, i, l, j, p):
        if i != j or l != p:
            return np.sin(4 * self.beta) * self.zz1(i, l, j, p) + (np.sin(2 * self.beta)) ** 2 * self.zz2(i, l, j, p)
        else:
            return self.j_m[self.pos(i, l), self.pos(j, p)]

    def zz1(self, i, l, j, p):
        val1_1 = np.cos(2 * self.gamma * self.h_m[self.pos(i, l)])
        val1_2 = np.cos(2 * self.gamma * self.h_m[self.pos(j, p)])

        for x in range(self.d):
            for u in range(self.k + 1):
                if (x != i or u != l) and (x != j or u != p):
                    val1_1 *= np.cos(2 * self.gamma * self.j_m[self.pos(i, l), self.pos(x, u)])
                    val1_2 *= np.cos(2 * self.gamma * self.j_m[self.pos(j, p), self.pos(x, u)])

        val1 = -self.j_m[self.pos(i, l), self.pos(j, p)] / 2 * np.sin(
            2 * self.gamma * self.j_m[self.pos(i, l), self.pos(j, p)]) * (val1_1 + val1_2)

        return val1

    def zz2(self, i, l, j, p):
        val2_1 = np.cos(2 * self.gamma * (self.h_m[self.pos(i, l)] + self.h_m[self.pos(j, p)]))
        val2_2 = -np.cos(2 * self.gamma * (self.h_m[self.pos(i, l)] - self.h_m[self.pos(j, p)]))

        for x in range(self.d):
            for u in range(self.k + 1):
                if (x != i or u != l) and (x != j or u != p):
                    val2_1 *= np.cos(2 * self.gamma * (
                                self.j_m[self.pos(i, l), self.pos(x, u)] + self.j_m[self.pos(j, p), self.pos(x, u)]))
                    val2_2 *= np.cos(2 * self.gamma * (
                                self.j_m[self.pos(i, l), self.pos(x, u)] - self.j_m[self.pos(j, p), self.pos(x, u)]))

        val2 = -self.j_m[self.pos(i, l), self.pos(j, p)] / 2 * (val2_1 + val2_2)

        return val2

    def evaluate_cost1(self, beta, gamma, offset=True):
        scalar = False

        if len(np.shape(beta)) > 1:
            self.beta = beta[0, :]
            self.gamma = gamma[:, 0]
        else:
            if len(np.shape(beta))==0:
                scalar = True
            self.beta = beta
            self.gamma = gamma

        cost_z1 = 0
        cost_zz1 = 0
        cost_zz2 = 0

        for x in range(self.d):
            for u in range(self.k + 1):
                cost_z1 += self.z1(x, u)

                for y in range(self.d):
                    for v in range(self.k + 1):

                        if self.pos(y, v) > self.pos(x, u):
                            cost_zz1 += self.zz1(x, u, y, v)
                            cost_zz2 += self.zz2(x, u, y, v)

        cost = np.matrix(cost_z1).T * np.matrix(np.sin(2 * self.beta)) + np.matrix(cost_zz1).T * np.matrix(
            np.sin(4 * self.beta)) + np.matrix(cost_zz2).T * np.matrix((np.sin(2 * self.beta)) ** 2)

        if offset:
            cost+= self.offset

        if scalar:
            return cost[0, 0]
        else:
            return cost

    def evaluate_cost1_c(self, params):
        return self.evaluate_cost1(params[0], params[1])

    def grid_search1(self, g_size):
        if self.p != 1:
            raise Exception("Wrong p value: ", self.p)

        j = 0
        while(j < int(self.range_l*self.N)):
            beta = np.linspace(0, np.pi, int(g_size / 2))
            gamma = np.linspace(np.pi/(self.range_l*self.N)*j, np.pi/(self.range_l*self.N)*(j+1), int(g_size / 2))

            B, G = np.meshgrid(beta, gamma)

            C = self.evaluate_cost1(B, G)

            argmins1 = np.argmin(C, axis=1)

            O1 = []
            for i in range(len(argmins1)):
                O1.append(C[i, argmins1[i]])

            extrma = argrelextrema(np.array(O1), np.less)

            if (len(extrma[0]) >0):
                argmins2 = argrelextrema(np.array(O1), np.less)[0][0]
                return [beta[argmins1[argmins2]], gamma[argmins2]], C[argmins2, argmins1[argmins2]]
            else:
                print("Search extension :", j+1)
                j+=1

        print("Error, no minimum found")
        raise RuntimeError


    def evaluate_grid1(self, g_size, scale=1, offset=0):
        beta = np.linspace(0, np.pi, int(g_size / 2))
        gamma = np.linspace(0, np.pi/scale+offset, int(g_size / 2))

        B, G = np.meshgrid(beta, gamma)
        C = self.evaluate_cost1(B, G)

        return C, beta, gamma

    def evaluate_grid2(self, g_size, scale=1, offset=0):
        return super().evaluate_grid1(g_size, scale)
