import numpy as np
import sys
import os

sys.path.insert(1,  os.path.dirname(os.path.realpath(__file__)))
from QAOA.QAOA_tools import QAOA1_Analytical

class QAOA_Binary(QAOA1_Analytical):
  def __init__(self, backend, sim_iter=250):
    super().__init__(backend, sim_iter)

  def omega(self, i):
    return -1/2

  def alpha(self,i,l):
    return -2**(l-1)

  def set_k(self, k_r):
    self.k = k_r

  def compute_filter_fn(self):
    self.filter_m = (self.cost_m).astype(int) > 0

  def get_feasible_ranges(self):
    return [np.arange(-2**self.k, 2**self.k)]*self.d