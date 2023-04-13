# The Quantum Approximate Optimization Algorithm (QAOA) for the Shortest Vector Problem (SVP)

This repository contains the code to reproduce the numerical results from my master's thesis at Imperial College London. Note that this project has dependencies on Qiskit and is intended to be run on a HPC cluster.

* The "QAOA" directory contains the implementation of the standard [[1](https://arxiv.org/pdf/2105.13106.pdf)], 2-penalty / d-penalty [[2](https://arxiv.org/pdf/2202.06757.pdf)], XY and no-zero encoding (introduced in the thesis). Furthermore, when possible, the analytical first-depth expectation is also provided.

* The "lattices" directory includes a file to generate random and cyclic lattices.

* The "asymptotics" directory includes tools to evaluate the analytical results for the first-depth QAOA over infinite-dimensional random lattices for the standard encoding.

* The "benchmarks" directory includes tools to assess the performance of these encodings over specificed lattices and to evaluate the use of the asymptotical predictions.

## Sample results

### Benchmarks for random (left) and cyclic (right) lattices given $(d,k)=(5, 1)$

* Average probability of sampling the zero and shortest vector

<p align="middle">
    <img src="/plots/light/l01_generic_5d_10r_1k_un.png#gh-light-mode-only" height="200" />
  <img src="/plots/light/l01_cyclic_5d_10r_1k_un.png#gh-light-mode-only" height="200" /> 
  <img src="/plots/dark/l01_generic_5d_10r_1k_un.png#gh-dark-mode-only" height="200" />
  <img src="/plots/dark/l01_cyclic_5d_10r_1k_un.png#gh-dark-mode-only" height="200" /> 
</p>

* Cumulative histograms for the first-depth approximation factor $\delta$

<p align="middle">
  <img src="/plots/light/hist_5d_1p_0s.png#gh-light-mode-only" height="200" />
  <img src="/plots/light/hist_5d_1p_1s.png#gh-light-mode-only" height="200" /> 
  <img src="/plots/dark/hist_5d_1p_0s.png#gh-dark-mode-only" height="200" />
  <img src="/plots/dark/hist_5d_1p_1s.png#gh-dark-mode-only" height="200" /> 
</p>

### First-depth predictions for random lattices using the standard encoding

* Predicted (left) and simulated (right) average energy landscape given $(d, k, q)=(20, 3, 10)$ 

<p align="middle">
  <img src="/plots/light/surface_asympt.png#gh-light-mode-only" height="200" />
  <img src="/plots/light/surface_sim.png#gh-light-mode-only" height="200" /> 
  <img src="/plots/dark/surface_asympt.png#gh-dark-mode-only" height="200" />
  <img src="/plots/dark/surface_sim.png#gh-dark-mode-only" height="200" /> 
</p>

* Predicted (dotted lines) and simulated (box plots) optimal parameters given $(d, k)=( \\{ 1,\ldots, 20 \\} , \\{ 1,2,3 \\} )$

<p align="middle">
  <img src="/plots/light/prediction_generic_gamma.png#gh-light-mode-only" height="200" />
  <img src="/plots/light/prediction_generic_beta.png#gh-light-mode-only" height="200"" /> 
  <img src="/plots/dark/prediction_generic_gamma.png#gh-dark-mode-only" height="200" />
  <img src="/plots/dark/prediction_generic_beta.png#gh-dark-mode-only" height="200"" /> 
</p>
