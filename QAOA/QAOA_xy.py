import numpy as np
import functools as ft

from qiskit import QuantumCircuit
from qiskit.opflow import I, X, Y
from qiskit.opflow import PauliTrotterEvolution, Suzuki

from QAOA.QAOA_tools import QAOA_Simulation

class QAOA_XY(QAOA_Simulation):
    def __init__(self, backend, sim_iter=250):
        super().__init__(backend, sim_iter)
        self.tau = 1 / 4

    def omega(self, i):
        return 1 / 4

    def alpha(self, i, l):
        if l <= self.k - 2:
            return -2 ** (l - 1)
        elif l == self.k - 1:
            return 2 ** (self.k - 2) + 1 / 4
        else:
            return -1 / 4

    def set_k(self, k_r):
        self.k = k_r + 1

    def compute_offset(self):
        super().compute_offset()

        val = 0
        for i in range(self.d):
            val += self.G[i, i]

        self.offset += (self.tau ** 2) * val

    def init_circ(self):
        qd_s = np.ones(2 ** (self.k))

        s1 = np.kron([1, 0], qd_s)
        s0 = np.kron([0, 1], qd_s)

        k_s = []
        for i in range(self.d):
            list_states = np.tile(s0, (i, 1))
            list_states = np.append(list_states, [s1], axis=0)
            list_states = np.append(list_states, np.tile(s0, (self.d - i - 1, 1)), axis=0)
            list_states = reversed(list_states)
            k_s.append(ft.reduce(np.kron, list_states))

        t_s = np.sum(k_s, axis=0)
        t_s /= np.linalg.norm(t_s)

        qc = QuantumCircuit(self.N)
        qc.initialize(t_s)

        return qc

    def mixer_ham_circ(self, l):
        mixer_h = QuantumCircuit(self.N)

        for i in range(self.d):
            for p in range(self.k):
                mixer_h.rx(-2 * self.params[2 * l], self.pos(i, p))

        # Approx. evolution for d > 2
        if self.d > 2:
            p_s = 0
            for i in range(self.d):
                op = 0
                if i != self.d - 1:
                    op = (I ^ ((self.k + 1) * i + self.k)) ^ X ^ (I ^ (self.k)) ^ X ^ (
                                I ^ (self.N - (self.k + 1) * (i + 2)))
                    op += (I ^ ((self.k + 1) * i + self.k)) ^ Y ^ (I ^ (self.k)) ^ Y ^ (
                                I ^ (self.N - (self.k + 1) * (i + 2)))
                elif i == self.d - 1 and self.d > 2:
                    op = (I ^ self.k) ^ X ^ (I ^ (self.N - (self.k + 2))) ^ X
                    op += (I ^ self.k) ^ Y ^ (I ^ (self.N - (self.k + 2))) ^ Y

                p_s += op

            evolution_op = (-self.params[2 * l] * p_s).exp_i()

            trotterized_op = PauliTrotterEvolution(trotter_mode=Suzuki(order=1)).convert(evolution_op)
            mixer_h.append(trotterized_op, *(mixer_h.qregs))
            mixer_h = mixer_h.decompose().decompose().decompose()

            return mixer_h

        # Exact evolution for d = 2
        else:
            mixer_h.rxx(-2 * self.params[2 * l], self.pos(0, self.k), self.pos(1, self.k))
            mixer_h.ryy(-2 * self.params[2 * l], self.pos(0, self.k), self.pos(1, self.k))

            return mixer_h

    def j2_f(self, i):
        val = 0

        for j in range(self.d):
            val += self.G[i, j]

        return 2 * self.tau * self.omega(i) * val

    def j3_f(self, i, j, l):
        return 2 * self.tau * self.alpha(j, l) * self.G[i, j]

    def j4_f(self, i, j):
        return 2 * self.tau ** 2 * self.G[i, j]

    def cost_ham_circ(self, l):
        qc = super().cost_ham_circ(l)

        for i in range(self.d):
            qc.rz(2 * self.params[2 * l + 1] * self.j3_f(i, i, self.k - 1), self.pos(i, self.k))
            qc.rz(2 * self.params[2 * l + 1] * self.j3_f(i, i, self.k), self.pos(i, self.k - 1))

        for i in range(self.d):
            qc.rzz(2 * self.params[2 * l + 1] * self.j2_f(i), self.pos(i, self.k - 1), self.pos(i, self.k))

        for i in range(self.d):
            for j in range(self.d):
                for p in range(self.k + 1):
                    if (j != i or p != self.k) and (j != i or p != self.k - 1):
                        qc.cnot(self.pos(i, self.k - 1), self.pos(i, self.k))
                        qc.cnot(self.pos(i, self.k), self.pos(j, p))

                        qc.rz(2 * self.params[2 * l + 1] * self.j3_f(i, j, p), self.pos(j, p))

                        qc.cnot(self.pos(i, self.k), self.pos(j, p))
                        qc.cnot(self.pos(i, self.k - 1), self.pos(i, self.k))

        for i in range(self.d):
            for j in range(i + 1, self.d):
                qc.cnot(self.pos(i, self.k - 1), self.pos(i, self.k))
                qc.cnot(self.pos(i, self.k), self.pos(j, self.k - 1))
                qc.cnot(self.pos(j, self.k - 1), self.pos(j, self.k))

                qc.rz(2 * self.params[2 * l + 1] * self.j4_f(i, j), self.pos(j, self.k))

                qc.cnot(self.pos(j, self.k - 1), self.pos(j, self.k))
                qc.cnot(self.pos(i, self.k), self.pos(j, self.k - 1))
                qc.cnot(self.pos(i, self.k - 1), self.pos(i, self.k))

        return qc

    def compute_cost_fn(self):
        super().compute_cost_fn()

        for i in range(self.d):
            self.cost_m += self.j3_f(i, i, self.k - 1) * (1 - 2 * self.bl[:,self.pos(i, self.k)])
            self.cost_m += self.j3_f(i, i, self.k) * (1 - 2 * self.bl[:,self.pos(i, self.k - 1)])

        for i in range(self.d):
            self.cost_m += self.j2_f(i) * (1 - 2 * self.bl[:,self.pos(i, self.k - 1)]) * (1 - 2 * self.bl[:,self.pos(i, self.k)])

        for i in range(self.d):
            for j in range(self.d):
                for p in range(self.k + 1):
                    if (j != i or p != self.k) and (j != i or p != self.k - 1):
                        self.cost_m += self.j3_f(i, j, p) * (1 - 2 * self.bl[:,self.pos(i, self.k - 1)]) * (
                                    1 - 2 * self.bl[:,self.pos(i, self.k)]) * (1 - 2 * self.bl[:,self.pos(j, p)])

        for i in range(self.d):
            for j in range(i + 1, self.d):
                self.cost_m += self.j4_f(i, j) * (1 - 2 *self.bl[:,self.pos(i, self.k - 1)]) * (1 - 2 * self.bl[:,self.pos(i, self.k)]) * (
                            1 - 2 * self.bl[:,self.pos(j, self.k - 1)]) * (1 - 2 * self.bl[:,self.pos(j, self.k)])
        self.z_m = (self.cost_m).astype(int) == 0

    def compute_filter_fn(self):
        self.filter_m = (self.cost_m > 0).astype(int)

    def get_feasible_ranges(self, nz):
        ranges = []
        for i in range(self.d):
            if i != nz:
                ranges.append(np.arange(-2 ** self.k+1, 2 ** self.k+1))
            else:
                r = np.arange(-2 ** self.k, 2 ** self.k + 1)
                r = np.delete(r, 2 ** self.k)
                ranges.append(r)

        return ranges

    def qudit_value(self, i):
        val = super().qudit_value(i)
        val += self.tau * (1 - 2 * self.bl[self.pos(i, self.k - 1)]) * (1 - 2 * self.bl[self.pos(i, self.k)])
        return val
