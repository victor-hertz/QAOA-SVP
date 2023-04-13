import numpy as np

from QAOA.QAOA_tools import QAOA1_Analytical

class QAOA_NZ(QAOA1_Analytical):
  def __init__(self, backend, sim_iter=250):
    super().__init__(backend, sim_iter)
    self.nz_qudit=-1

  def omega(self, i):
    if i != self.nz_qudit:
      return -1/2
    else:
      return 0

  def alpha(self,i,l):
    if i != self.nz_qudit or l <= self.k-1:
      return -2**(l-1)
    else:
      return 2**(self.k-1)+1/2

  def set_k(self, k_r):
    self.k = k_r

  def compute_filter_fn(self):
    self.filter_m = (self.qudit_value(self.nz_qudit) != 2**self.k).astype(int)

  def set_parameters(self, B, k_r, p, nz_qudit):
    init = nz_qudit != self.nz_qudit
    self.nz_qudit = nz_qudit
    super().set_parameters(B, k_r, p, force_init=init)

  def get_feasible_ranges(self, nz):
    ranges = []
    for i in range(self.d):
      if i != nz:
        ranges.append(np.arange(-2**self.k, 2**self.k))
      else:
        r = np.arange(-2**self.k, 2**self.k+1)
        r = np.delete(r, 2**self.k)
        ranges.append(r)

    return ranges

  def get_feasible_ranges(self):
    ranges = []
    for i in range(self.d):
      if i != self.nz_qudit:
        ranges.append(np.arange(-2**self.k, 2**self.k))
      else:
        r = np.arange(-2**self.k, 2**self.k+1)
        r = np.delete(r, 2**self.k)
        ranges.append(r)

    return ranges

  def compute_glob_opt_sol(self):
    t_nz_i = self.nz_qudit
    opt_nz_qudit = 0
    b_val = -1
    for i in range(self.d):
      self.nz_qudit = i
      self.compute_interactions()
      self.compute_offset()
      self.compute_cost_fn()
      self.compute_opt_sol()

      if b_val == -1 or self.opt_sol<b_val:
        opt_nz_qudit = i
        b_val = self.opt_sol

    self.nz_qudit = t_nz_i
    self.compute_interactions()
    self.compute_offset()
    self.compute_cost_fn()
    self.compute_opt_sol()

    return opt_nz_qudit

