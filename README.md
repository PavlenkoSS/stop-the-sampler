# Stop the Sampler! Classifier-Based Adaptive Stopping for Sampling Kernels

Official code for the paper [Stop the Sampler! Classifier-Based Adaptive Stopping for Sampling Kernels](https://arxiv.org/abs/2606.16073)

Kirill Korolev, Nikita Morozov, Stepan Pavlenko, Esmeralda S. Whitammer, Sergey Samsonov 

This repository is based on the [code](https://github.com/DenisBless/variational_sampling_methods) of Blessing et al. (2024) from their paper "[Beyond ELBOs: A Large-Scale Evaluation of Variational Methods for Sampling](https://arxiv.org/abs/2406.07423)".

## Abstract
Sampling from complex, unnormalized probability densities is a fundamental challenge in Bayesian inference and probabilistic modeling. While Markov chain Monte Carlo (MCMC) methods provide asymptotic guarantees, they often suffer from slow mixing and high computational costs due to fixed or manually tuned trajectory lengths. In this work, we propose a novel framework that treats trajectory termination as a learnable component of the sampling dynamics. By framing MCMC within the theory of non-acyclic generative flow networks (GFlowNets), we train state-dependent neural classifiers to decide when a trajectory has reached a high-density region and should terminate. We theoretically establish the connection between optimal classifiers and the target density via detailed balance conditions and introduce a multilevel training scheme to facilitate exploration in complex geometries. Experimental results across various benchmark densities demonstrate that our approach significantly reduces average trajectory lengths while improving mode coverage and mixing compared to standard MCMC baselines.

## Installation
- python 3.10.14
- jax 0.6.2

We recommend using the conda (or mamba) environment to install the dependencies.
```bash
conda create -n stop-the-sampler python=3.10.14
conda activate stop-the-sampler
```

Install the jax and jaxlib with the appropriate CUDA version or TPU support, e.g., cuda12
```bash
pip install -U "jax[cuda12]==0.6.2"
```

Install the other dependencies.
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python run.py algorithm=<algorithm_name> target=<target_name>
```

`<algorithm_name>` can be one of the following:
- `gfn_non_acyclic_baseline` (for ULA)
- `gfn_non_acyclic` (for the main algorithm (section 3.3))
- `gfn_non_acyclic_ml` (for the multilevel scheme (section 3.5))

`target_name` can be one of the following:
- `gaussian_mixture9`
- `funnel`
- `many_well`
- `nice_digits`

### Optimizer

Every parameter group is trained with Adam by default. AdamO -- Adam plus a decoupled isometry regularizer ([Rosseau, Müller & Nowé](https://arxiv.org/abs/2606.09762), ported from `plasticity-umap-reset`) -- is selected with the `optimizer` config group:
```bash
python run.py algorithm=gfn_non_acyclic target=funnel algorithm/optimizer=adamo algorithm.optimizer.lam=1e-3
```
It adds `-iso_lr * lam * grad R_iso(W)` outside Adam's adaptive scaling, where `R_iso(W) = sum_i (sigma_i(W)^2 - 1)^2` over every `kernel` of the networks; biases, `logZ` and the time phases get plain Adam. `iso_lr: null` means each group's own step size, with its schedule. `scope: all` (the paper's) covers every kernel, output layers included; `algorithm.optimizer.scope=hidden` leaves the output layers -- the kernels the networks initialise to a constant -- to plain Adam. The summed `R_iso` is logged to Comet as `iso_penalty` for every optimizer.

The paper starts AdamO from an orthogonal init. `algorithm.model.hidden_init=orthogonal` draws every hidden `Dense` kernel semi-orthogonal (gain 1), so `R_iso` of the hidden layers is zero at init; the default `lecun_normal` is Flax's default and draws exactly what earlier runs did. The output layers keep their near-zero init under either law, so the sampler still starts from the reference process:
```bash
python run.py algorithm=gfn_non_acyclic target=funnel algorithm/optimizer=adamo algorithm.model.hidden_init=orthogonal
```

Tests: `python -m pytest tests`
