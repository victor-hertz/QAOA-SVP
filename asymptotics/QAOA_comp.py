import numpy as np
import os
import sys
from qiskit import Aer

sys.path.insert(1,  os.path.dirname(sys.path[0]))
from QAOA.QAOA_binary import QAOA_Binary

class QAOA_Comp:
    def __init__(self, d, k ,r):
        backend = Aer.get_backend('statevector_simulator')
        self.qaoa = QAOA_Binary(backend)
        self.qaoa.k = k
        self.d = d
        self.k = k
        self.r = r
        self.mu = d*r*(r+1)/3
        self.sigma = np.sqrt(d*(r*(r+1)/3)**2);
        self.sigmap= np.sqrt((r*(r+1)/3)**2);
        self.sigmau = np.sqrt(d*(r*(r+1)*(3*r**2+3*r-1)/15-(r*(r+1)/3)**2));
        self.scale1 = 2 ** (2 * self.k)
        self.scales = self.sigmap * self.d * self.scale1

    def alpha(self,l):
        return -2**(l-1)

    def set_parameters(self, beta, gamma):
        self.beta = beta
        self.gamma = gamma/self.scales
        self.gammap = gamma/self.scale1
        self.qaoa.beta = beta
        self.qaoa.gamma = self.gamma

    def ideal_mat(self, params):
        d = len(params)
        a = []
        for i in range(d):
            a.append(np.roll(params,i))
        return np.array(a)

    def generate_mat(self, ideal):
        raise NotImplemented

    def z1_sim(self, i, p,trials, ideal=False):
        values=0
        for a in range(trials):
            B = self.generate_mat(ideal)

            if ideal:
                qc = self.qaoa.set_parameters(B, self.k, 0,G=B)
            else:
                qc = self.qaoa.set_parameters(B, self.k, 0)

            values +=  self.qaoa.z1(i,p)

        return values/trials

    def zz1_sim(self, i,j,p,l, trials, ideal=False):
        values=0
        for a in range(trials):
            B = self.generate_mat(ideal)

            if ideal:
                qc = self.qaoa.set_parameters(B, self.k, 0,G=B)
            else:
                qc = self.qaoa.set_parameters(B, self.k, 0)

            values += self.qaoa.zz1(i,p,j,l)

        return values/trials

    def zz2_sim(self, i,j,p,l, trials, ideal=False):
        values=0
        for a in range(trials):
            B = self.generate_mat(ideal)

            if ideal:
                qc = self.qaoa.set_parameters(B, self.k, 0,G=B)
            else:
                qc = self.qaoa.set_parameters(B, self.k, 0)

            values += self.qaoa.zz2(i,p,j,l)

        return values/trials

    def z1_gauss(self, p):
        raise NotImplemented

    def zz1_gauss(self, i, j, p, l):
        raise NotImplemented

    def zz2_gauss(self, i, j, p, l):
        raise NotImplemented

    def ham_gauss(self, beta, gamma):
        self.set_parameters(beta, gamma)

        cost_z1 = 0
        cost_zz1 = 0
        cost_zz2 = 0

        for u in range(self.k + 1):
            cost_z1 += self.z1_gauss(u)*self.d

            for v in range(self.k + 1):
                if u < v:
                    cost_zz1 += self.zz1_gauss(u, v, True)*self.d
                    cost_zz2 += self.zz2_gauss(u, v, True)*self.d

                cost_zz1 += self.zz1_gauss(u, v, False)*self.d*(self.d-1)/2
                cost_zz2 += self.zz2_gauss(u, v, False)*self.d*(self.d-1)/2

        cost = np.matrix(cost_z1).T * np.matrix(np.sin(2 * self.beta)) + np.matrix(cost_zz1).T * np.matrix(
        np.sin(4 * self.beta)) + np.matrix(cost_zz2).T * np.matrix((np.sin(2 * self.beta)) ** 2)

        return cost

    def ham_gauss_asympt_opt(self, params):
        return self.ham_gauss_asympt(params[0], params[1])[0, 0]

    def ham_gauss_asympt(self, beta, gamma):
        self.set_parameters(beta, gamma)

        cost_z1 = 0
        cost_zz1 = 0
        cost_zz2 = 0

        for u in range(self.k + 1):
            cost_z1 += self.z1_gauss_asympt(u)

            for v in range(self.k + 1):
                if u < v:
                    cost_zz1 += self.zz1d_gauss_asympt(u, v)
                    cost_zz2 += self.zz2d_gauss_asympt(u, v)

                cost_zz1 += self.zz1_gauss_asympt(u, v)/2
                cost_zz2 += self.zz2_gauss_asympt(u, v)/2

        cost = np.matrix(cost_z1).T * np.matrix(np.sin(2 * self.beta)) + np.matrix(cost_zz1).T * np.matrix(
        np.sin(4 * self.beta)) + np.matrix(cost_zz2).T * np.matrix((np.sin(2 * self.beta)) ** 2)

        return cost

    def ham_sim_opt(self, params):
        return self.qaoa.evaluate_cost1(params[0], params[1]/self.scales, offset=False)

    def ham_sim_opt_noscale(self, params):
        return self.qaoa.evaluate_cost1(params[0], params[1], offset=False)

    def ham_sim(self, beta, gamma, ideal, trials, mat=[], offset=False, do_scale=True):
        cost=0
        if do_scale:
            scale=self.scales
        else:
            scale=1

        for a in range(trials):
            if np.shape(mat)[0] == 0:
                B = self.generate_mat(ideal)
            else:
                B = mat
                trials=1

            if ideal:
                self.qaoa.set_parameters(B, self.k, 0, B, force_init=True)
            else:
                self.qaoa.set_parameters(B, self.k, 0, force_init=True)

            cost+=self.qaoa.evaluate_cost1(beta, gamma/scale, offset)

        return cost/trials

