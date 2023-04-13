import numpy as np
import copy
from itertools import combinations

from QAOA.QAOA_tools import QAOA_Simulation

class QAOA_dPenalty(QAOA_Simulation):
    def __init__(self, backend, sim_iter=250, redundant=False):
        super().__init__(backend, sim_iter)
        self.redundant = redundant

    def omega(self, i):
        return 1 / 2

    def alpha(self, i, l):
        if l <= self.k - 2:
            return -2 ** (l - 1)
        elif l == self.k - 1:
            return 2 ** (self.k - 2) + 1 / 2
        else:
            return -1 / 2

    def set_k(self, k_r):
        self.k = k_r + 1

    def compute_offset(self):
        super().compute_offset()
        self.offset += self.L / (2 ** self.d)

    def compute_cost_fn(self):
        super().compute_cost_fn()
        pen = self.cost_m >= 0

        for i in range(self.d):
            pen *= self.bl[:,self.pos(i, self.k)] == 1

        pen = pen.astype(float)
        pen*=copy.deepcopy(self.L)

        self.cost_m += pen

    def compute_filter_fn(self):
        self.filter_m = self.cost_m > 0

        for i in range(self.d):
            qv = self.qudit_value(i)
            self.filter_m *= qv != 2 ** (self.k - 1)
            self.filter_m *= qv != 2 ** (self.k - 1) + 1

        self.filter_m = self.filter_m.astype(int)

    def cost_ham_circ(self, l):
        cost_h = super().cost_ham_circ(l)

        cand = range(self.d)

        for i in range(1, self.d + 1):
            comb = list(combinations(cand, i))

            for c in comb:
                for j in range(i - 1):
                    cost_h.cnot(self.pos(c[j], self.k), self.pos(c[j + 1], self.k))

                cost_h.rz(self.L * 2 * self.params[2 * l + 1] * (-1) ** i / (2 ** self.d),
                          self.pos(c[i - 1], self.k))

                for j in reversed(range(i - 1)):
                    cost_h.cnot(self.pos(c[j], self.k), self.pos(c[j + 1], self.k))

        return cost_h

    def set_parameters(self, B, k_r, p,G=[0]):
        self.d = np.shape(B)[0]

        if not self.redundant:
            self.L = (np.sqrt(self.d) * abs(np.linalg.det(B)) ** (1 / self.d)) ** 2
            self.sL = copy.deepcopy(self.L)
        else:
            self.sL = 0
            self.L = 0
            self.cost_ham_circ = super().cost_ham_circ
            self.compute_cost_fn = super().compute_cost_fn

        super().set_parameters(B, k_r, p)

    def toggle_penalty(self, enable):
        if enable:
            self.L = copy.deepcopy(self.sL)
        else:
            self.L = 0

        self.compute_interactions()
        self.compute_offset()
        self.compute_cost_fn()
        self.compute_opt_sol()
        self.compute_obj_fn()
        self.compute_filter_fn()

    def get_feasible_ranges(self):
        return [np.arange(-2 ** self.k, 2 ** self.k + 1)] * self.d