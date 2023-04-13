import numpy as np
import os
import sys

sys.path.insert(1,  os.path.dirname(sys.path[0]))
from asymptotics.QAOA_comp import QAOA_Comp

class QAOA_Comp_Generic(QAOA_Comp):
    def __init__(self, d, k, r):
        super().__init__(d, k, r)

    def generate_mat(self, ideal=False):
        i = True

        while (i):
            if ideal:
                a = np.random.normal(0, self.sigma, size=(self.d, self.d))
                a -= np.diag(a.diagonal())
                a -= np.tril(a)
                a += a.T
                np.fill_diagonal(a, np.random.normal(self.mu, self.sigmau, size=self.d))

            else:
                a = np.random.randint(-self.r, self.r + 1, size=(self.d, self.d))

            if not ideal:
                i = np.isclose(np.linalg.det(a),0)
            else:
                i=0
            #i = 0
        return a

    def compute_chi(self, p, sigma=-1):
        if sigma == -1:
            sigma = self.alpha(p) * self.sigma

        chi = 0
        param = (sigma * self.gamma) ** 2

        for v in range(2 ** (self.k + 1) + 1):
            if v % 2 != 0:
                chi += np.exp(-2 * param * (v ** 2 + 1)) * np.cosh(4 * param * v)

        chi /= 2 ** (self.k)
        return chi

    def compute_chi_asympt(self, p, sigma=-1):
        if sigma == -1:
            sigma = self.alpha(p) * self.sigma

        chi = 0
        param = (sigma * self.gamma) ** 2

        for v in range(2 ** (self.k + 1) + 1):
            if v % 2 != 0:
                chi += -2 * param * (v ** 2 + 1)

        chi *= self.d / 2 ** (self.k)  # problem
        chi = np.exp(chi)
        return chi

    def compute_chi_prod_asympt(self, p, sigma=-1):
        if sigma == -1:
            sigma = self.alpha(p)

        param = (sigma * self.gammap) ** 2

        return np.exp(-2 * param * 2 / 3 * (1 + 8 * self.alpha(self.k) ** 2))

    def compute_omega(self, p):
        omega = 0
        param = (self.alpha(p) * self.sigma * self.gamma) ** 2

        for v in range(2 ** (self.k + 1) + 1):
            if v % 2 != 0:
                omega += np.exp(-2 * param * (v ** 2 + 1)) * (np.cosh(4 * param * v) - v * np.sinh(4 * param * v))

        omega *= self.sigma ** 2 * self.gamma * self.alpha(p) / (2 ** (self.k - 1))
        return omega

    def compute_omega_asympt(self, p):
        return 2 * self.gammap * self.sigmap* self.alpha(p)

    def check_cond1(self, p, v):
        s = (np.binary_repr(v + 2 ** (self.k + 1) - 1 - 2 ** p))[::-1]
        if s[0] == '0' and (len(s) < p + 2 or s[p + 1] == '0'):
            return True

        return False

    def check_cond2(self, p, l, v):
        s = (np.binary_repr(v + 2 ** (self.k + 1) - 1 - 2 ** p - 2 ** l))[::-1]
        if s[0] == '0' and (len(s) < p + 2 or s[p + 1] == '0') and (len(s) < l + 2 or s[l + 1] == '0'):
            return True

        return False

    def compute_cu(self, p):
        cu = 0
        param = (self.alpha(p) * self.sigmau * self.gamma) ** 2

        for v in range(2 ** (self.k + 1) + 1):
            if self.check_cond1(p, v):
                cu += np.exp(-2 * param * (1 + v) ** 2) * np.cos(2 * self.alpha(p) * self.gamma * (1 + v) * self.mu)
                cu += np.exp(-2 * param * (1 - v) ** 2) * np.cos(2 * self.alpha(p) * self.gamma * (1 - v) * self.mu)

        cu /= 2 ** (self.k)
        return cu*self.alpha(p)

    def compute_cu_asympt(self, p):
        val = self.alpha(p)/(4*self.alpha(self.k))
        val *= np.sin(8*self.gammap*self.alpha(p)*self.alpha(self.k))
        val = np.divide(val, np.tan(2*self.gammap*self.alpha(p))*np.cos(4*self.gammap*self.alpha(p)**2))

        return val

    def compute_su(self, p):
        param = (self.alpha(p) * self.sigmau * self.gamma) ** 2

        t1 = 0
        t2 = 0
        for h in range(2 ** (self.k + 1) + 1):
            if self.check_cond1(p, h):
                t1 += np.exp(-2 * param * (1 + h) ** 2) * (1 + h) * np.cos(
                    2 * self.alpha(p) * self.gamma * (1 + h) * self.mu)
                t1 += np.exp(-2 * param * (1 - h) ** 2) * (1 - h) * np.cos(
                    2 * self.alpha(p) * self.gamma * (1 - h) * self.mu)

                t2 += np.exp(-2 * param * (1 + h) ** 2) * np.sin(2 * self.alpha(p) * self.gamma * (1 + h) * self.mu)
                t2 += np.exp(-2 * param * (1 - h) ** 2) * np.sin(2 * self.alpha(p) * self.gamma * (1 - h) * self.mu)

        t1 *= self.gamma * self.alpha(p) * (self.sigmau) ** 2
        t2 *= self.mu / 2
        su = t1 + t2
        su /= (2 ** (self.k - 1))
        return su*self.alpha(p)

    def compute_su_asympt(self, p):
        val = self.alpha(p)*self.sigmap/(4*self.alpha(self.k))*np.sin(8*self.gammap*self.alpha(p)*self.alpha(self.k))
        val = np.divide(val, (np.cos(4*self.gammap*self.alpha(p)**2)))

        return val

    def compute_zeta1(self, p, l):
        zeta = 0
        param1 = (self.alpha(p) * self.sigma * self.gamma) ** 2

        for v in range(2 ** (self.k + 1) + 1):
            if self.check_cond1(l, v):
                param2 = 2 * self.alpha(l) + v
                param3 = 2 * self.alpha(l) - v
                zeta += np.exp(-2 * param1 * (param2 ** 2 + 1)) * (
                            param2 * np.cosh(4 * param2 * param1) - np.sinh(4 * param2 * param1))
                zeta += np.exp(-2 * param1 * (param3 ** 2 + 1)) * (
                            param3 * np.cosh(4 * param3 * param1) - np.sinh(4 * param3 * param1))


        zeta *= self.gamma * self.alpha(l) * self.alpha(p) * self.sigma ** 2
        zeta /= 2 ** (self.k - 1)
        return zeta

    def compute_zeta1_asympt(self, p, l):
        return 4 * self.alpha(p) * self.alpha(l) ** 2 * self.sigmap* self.gammap

    def z1_gauss(self, p):
        chi = self.compute_chi(p, -1)

        omega = self.compute_omega(p)
        su = self.compute_su(p)
        cu = self.compute_cu(p)

        val = (self.d - 1) * omega * cu

        val += chi * su
        val *= -chi ** (self.d - 2)

        return val

    def z1_gauss_asympt(self, p):
        val = self.alpha(p)/ (4*self.alpha(self.k))
        val *= - np.exp(-4 / 3 * (self.gammap * self.alpha(p)) ** 2 * (1 + 8 * self.alpha(self.k) ** 2))
        val *= (1+ np.divide(2*self.gammap*self.alpha(p),np.tan(2*self.gammap*self.alpha(p))))*np.sin(8*self.gammap*self.alpha(p)*self.alpha(self.k))
        val = np.divide(val,np.cos(4*self.gammap*self.alpha(p)**2))

        return val

    def compute_zeta2(self, p, l):
        val = 0
        a = [-1, 1]

        for v in range(2 ** (self.k + 1) + 1):
            if self.check_cond2(p, l, v):
                for h in a:
                    for k in a:
                        param = 2 * self.gamma * self.alpha(p) * (2 * self.alpha(l) + h * v + k * 1)
                        val += np.exp(- (self.sigmau * param) ** 2 / 2) * (
                                    self.sigmau ** 2 * param * np.cos(param * self.mu) + self.mu * np.sin(
                                param * self.mu))
        val /= 2 ** (self.k)
        val *= self.alpha(p) * self.alpha(l)  # 1/(2^(k-2))*1/4
        return val

    def compute_zeta2_asympt(self, p, l):
        val = self.alpha(p) * self.alpha(l) * self.sigmap / (4*self.alpha(self.k))
        val *= np.tan(4*self.gammap*self.alpha(p)*self.alpha(l))
        val *= np.sin(8*self.gammap*self.alpha(p)*self.alpha(self.k))
        val = np.divide(val, np.cos(4*self.gammap*self.alpha(p)**2)*np.tan(2*self.gammap*self.alpha(p)))

        return val

    def zz1d_gauss(self, p, l):
        chi1 = self.compute_chi(p) ** (self.d - 1)
        chi2 = self.compute_chi(l) ** (self.d - 1)

        val = chi1 * self.compute_zeta2(p, l)
        val += chi2 * self.compute_zeta2(l, p)

        val *= -1

        return val

    def zz1_gauss(self, p, l, diag=False):
        if diag:
            return self.zz1d_gauss(p, l)

        chi1 = self.compute_chi(p) ** (self.d - 2)
        chi2 = self.compute_chi(l) ** (self.d - 2)

        cu1 = self.compute_cu(p)
        zeta1 = self.compute_zeta1(p, l)

        cu2 = self.compute_cu(l)
        zeta2 = self.compute_zeta1(l, p)

        val = chi1 * cu1 * zeta1
        val += chi2 * cu2 * zeta2
        val *= -1

        return val


    def zz1d_gauss_asympt(self, p, l):
        w = [p, l]
        val = 0
        for v in w:
            t_val = np.exp(-4 / 3 * (self.gammap * self.alpha(v)) ** 2 * (1 + 8 * self.alpha(self.k) ** 2))
            t_val *= np.sin(8 * self.gammap * self.alpha(v) * self.alpha(self.k) )
            t_val = np.divide(t_val, np.cos(4*self.gammap*self.alpha(v)**2)*np.tan(2*self.gammap*self.alpha(v)))
            val += t_val

        val *= np.tan(4*self.gammap*self.alpha(p)*self.alpha(l))
        val *= -self.alpha(p) * self.alpha(l) /(4*self.alpha(self.k))
        return val

    def zz1_gauss_asympt(self, p, l):
        w = [p, l]

        val = 0
        for v in w:
            t_val = np.divide(1, np.tan(2*self.gammap*self.alpha(v))*np.cos(4*self.gammap*self.alpha(v)**2))
            t_val *= np.sin(8*self.gammap*self.alpha(self.k)*self.alpha(v))
            t_val *= np.exp( -4 / 3 * (self.gammap * self.alpha(v) ) ** 2 * (1 + 8 * self.alpha(self.k) ** 2))
            val += t_val

        val *= -self.gammap*(self.alpha(l)*self.alpha(p))**2 /self.alpha(self.k)

        return val

    def compute_gamma2(self, p, l, nega=1):
        val = 0

        a = [-1, 1]
        for v in range(2 ** (self.k + 1) + 1):
            if self.check_cond2(p, l, v):
                for h in a:
                    param = 2 * self.gamma * (self.alpha(p) + nega * self.alpha(l)) * (1 + h * v)
                    val += np.exp(-(self.sigmau * param) ** 2 / 2) * (
                                -self.sigmau ** 2 * param * np.sin(param * self.mu) + self.mu * np.cos(param * self.mu))

        val /= nega * 2 ** (self.k - 1)
        val *= self.alpha(p) * self.alpha(l)
        return val

    def compute_gamma2_asympt(self, p, l, t):
        param = 2 * self.gammap * (self.alpha(p) + t * self.alpha(l))
        val = np.sin(4*param*self.alpha(self.k))
        val = np.divide(val, np.cos(2*param*self.alpha(l))*np.cos(2*param*self.alpha(p))*np.tan(param))

        return t * val * self.alpha(p) * self.alpha(l) * self.sigmap / (4*self.alpha(self.k))

    def zz2d_gauss(self, p, l):
        chi1 = self.compute_chi(-1, np.abs(self.sigma * (self.alpha(p) + self.alpha(l)))) ** (self.d - 1)
        chi2 = self.compute_chi(-1, np.abs(self.sigma * (self.alpha(p) - self.alpha(l)))) ** (self.d - 1)

        val = chi1 * self.compute_gamma2(p, l, 1)
        val += chi2 * self.compute_gamma2(p, l, -1)

        val *= -1

        return val

    def zz2d_gauss_asympt(self, p, l):
        w = [-1, 1]

        val = 0
        for t in w:
            param = 2 * self.gammap * (self.alpha(p) + t * self.alpha(l))
            t_val = np.sin(4 * param * self.alpha(self.k))
            t_val = np.divide(t_val, np.cos(2 * param * self.alpha(l)) * np.cos(2 * param * self.alpha(p)) * np.tan(param))
            t_val *= t*np.exp( -4 / 3 * (self.gammap * (self.alpha(p)+t*self.alpha(l)) ) ** 2 * (1 + 8 * self.alpha(self.k) ** 2))
            val += t_val
        return -val * self.alpha(p) * self.alpha(l) /(4 * self.alpha(self.k))

    def zz2_sub(self, p, l, v1, v2, l3=1):
        val = 0
        a = [-1, 1]
        for l1 in a:
            for l2 in a:
                z1 = (l1*v2 + l2) * self.alpha(p) + l3*(v1+l2) * self.alpha(l)
                z2 = (v1+l2)*self.alpha(p)
                z3 = l3*(l1*v2+l2)*self.alpha(l)

                t_val = l3*z1
                t_val *= np.exp(-2*self.gamma**2*((z1*self.sigma)**2+(z2**2+z3**2)*self.sigmau**2))
                t_val *= np.sin(2*self.gamma*self.mu*(z2+z3))
                val += t_val

        return val

    def compute_gamma1(self, p, l):
        val = 0
        for v1 in range(2 ** (self.k + 1) + 1):
            if self.check_cond1(p, v1):
                for v2 in range(2 ** (self.k + 1) + 1):
                    if self.check_cond1(l, v2):
                        val += self.zz2_sub(p, l, v1, v2, 1)
                        val += self.zz2_sub(p, l, v1, v2, -1)

        return val * self.alpha(p) * self.alpha(l) * self.gamma * self.sigma ** 2 / (2 ** (2 * self.k - 1))

    def pf(self, p,l):
        t_val = np.sin(8*self.gammap*self.alpha(self.k)*self.alpha(p)) *(-1 + self.alpha(p)*np.sin(4*self.gammap*self.alpha(p))*np.tan(4*self.gammap*self.alpha(p)**2))
        t_val += 2*self.alpha(self.k)*np.cos(8*self.gammap*self.alpha(self.k)*self.alpha(p))*np.sin(4*self.gammap*self.alpha(p))
        val = np.sin(8*self.gammap*self.alpha(l)*self.alpha(self.k))*t_val
        val /= 4*self.alpha(self.k)**2*np.tan(2*self.gammap*self.alpha(l))*np.sin(2*self.gammap*self.alpha(p))**2*np.cos(4*self.gammap*self.alpha(l)**2)*np.cos(4*self.gammap*self.alpha(p)**2)
        val *= -self.sigmap*self.gammap*self.alpha(l)**2*self.alpha(p)
        return val

    def compute_gamma1_asympt(self, p, l):
        val = self.pf(p,l)+self.pf(l,p)
        return val

    def zz2_gauss_asympt(self, p, l):
        val = np.exp( -4 / 3 * self.gammap**2 * (self.alpha(p)**2+self.alpha(l)**2)* (1 + 8 * self.alpha(self.k) ** 2))
        return val*self.compute_gamma1_asympt(p,l)/self.sigmap

    def zz2_gauss(self, p, l, diag=False):
        if diag:
            return self.zz2d_gauss(p, l)

        chi = self.compute_chi(-1, self.sigma * np.sqrt((self.alpha(p) ** 2 + self.alpha(l) ** 2))) ** (self.d - 2)
        gamma1 = self.compute_gamma1(p, l)

        return gamma1 * chi
