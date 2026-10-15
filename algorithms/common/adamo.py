"""AdamO -- Adam with a decoupled isometry regularizer.

From *Preserving Plasticity in Continual Learning via Dynamical Isometry*
(Rosseau, Müller & Nowé, arXiv:2606.09762), ported from
``plasticity.optim.adamo`` (PavlenkoSS/plasticity-umap-reset) to Flax parameter
trees.  The optimizer keeps Adam's moments on the **task gradient only** and adds
the regularizer's gradient outside the adaptive scaling, exactly as AdamW
decouples weight decay:

    m_t = b1 m_{t-1} + (1 - b1) g_t
    v_t = b2 v_{t-1} + (1 - b2) g_t (*) g_t
    theta_{t+1} = theta_t - eta_t * mhat_t / (sqrt(vhat_t) + eps) - eta_iso_t * lam * r_t

with ``r_t = grad_theta sum_l R_iso(W_l)``.  Feeding ``r_t`` through the moments
instead would let ``v_t`` rescale the isometry force per coordinate, which is the
thing the decoupling is for.

The regularizer is the Gram deviation on the smaller side of the matrix::

    R_iso(W) = ||W^T W - I||_F^2  or  ||W W^T - I||_F^2,  whichever Gram is smaller

Both expand to ``sum_i (sigma_i(W)^2 - 1)^2``, so the penalty is a function of the
singular values alone and does not depend on Flax's ``(fan_in, units)`` kernel
layout versus PyTorch's ``(units, fan_in)``.

Differences from the original, both forced by this repository:

* a **linear map** is a rank-2-or-more leaf named ``kernel`` (Flax's name for a
  ``Dense``/``Conv`` weight) rather than ``weight`` (Equinox's).  Rank alone is not
  enough: ``timestep_phase`` and ``level_phase`` are ``(1, num_hid)`` and are not
  maps, and pushing them towards unit norm is not a defined thing to want.
* ``learning_rate`` and ``iso_lr`` may be optax schedules, since every trainer
  here builds its rates through ``lr_schedule``.  With constant rates the chain is
  the original's term for term.
"""
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax
from jax import Array


# --------------------------------------------------------------------------- #
# which leaves are linear maps
# --------------------------------------------------------------------------- #
def as_matrix(w: Array) -> Array:
    """A Flax kernel as ``(fan_in, units)``: identity on a ``Dense`` kernel, and a
    ``Conv`` kernel ``(kh, kw, C_in, C_out)`` becomes ``(kh * kw * C_in, C_out)``."""
    return w.reshape(-1, w.shape[-1]) if jnp.ndim(w) > 2 else w


def is_kernel_path(path) -> bool:
    """Does this leaf's path end in ``kernel``?"""
    return bool(path) and getattr(path[-1], "key", None) == "kernel"


def linear_maps(params):
    """A bool pytree over ``params``: which leaves are linear maps.

    Rank 2 or more *and* named ``kernel``.  Biases, ``logZ``, ``betas`` and the
    ``(1, num_hid)`` time/level phases are not.  Leaves that ``optax.masked`` has
    replaced by ``MaskedNode`` carry no arrays and are skipped by the flattening.
    """
    leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
    return jax.tree_util.tree_unflatten(
        treedef, [jnp.ndim(v) >= 2 and is_kernel_path(path) for path, v in leaves]
    )


# --------------------------------------------------------------------------- #
# the regularizer
# --------------------------------------------------------------------------- #
def r_iso(w: Array) -> Array:
    """``R_iso(W)`` for one kernel: the squared Gram deviation from identity."""
    w = as_matrix(w)
    rows, cols = w.shape
    gram = w.T @ w if rows >= cols else w @ w.T
    return jnp.sum((gram - jnp.eye(gram.shape[0], dtype=w.dtype)) ** 2)


def grad_r_iso(w: Array) -> Array:
    """``dR_iso/dW`` in closed form: ``4 W (W^T W - I)`` or ``4 (W W^T - I) W``.

    Written out rather than left to autodiff because it is *identically zero* at
    orthogonality; ``tests/test_adamo.py`` checks it against ``jax.grad``.
    """
    shape = w.shape
    w = as_matrix(w)
    rows, cols = w.shape
    if rows >= cols:
        return (4.0 * w @ (w.T @ w - jnp.eye(cols, dtype=w.dtype))).reshape(shape)
    return (4.0 * (w @ w.T - jnp.eye(rows, dtype=w.dtype)) @ w).reshape(shape)


@jax.jit
def iso_penalty(params) -> Array:
    """Total ``R_iso`` over every linear map in ``params``.

    Jitted because the trainers log it eagerly every iteration.
    """
    leaves = [
        w
        for w, k in zip(
            jax.tree_util.tree_leaves(params),
            jax.tree_util.tree_leaves(linear_maps(params)),
        )
        if k
    ]
    if not leaves:
        return jnp.asarray(0.0)
    return sum(r_iso(w) for w in leaves)


def iso_grads(params):
    """``r_t`` as a pytree matching ``params``; zero for anything not a linear map."""
    return jax.tree_util.tree_map(
        lambda w, k: grad_r_iso(w) if k else jnp.zeros_like(w),
        params,
        linear_maps(params),
    )


# --------------------------------------------------------------------------- #
# the optax transformation
# --------------------------------------------------------------------------- #
class AddIsoState(NamedTuple):
    count: Array  # steps taken, for a scheduled ``iso_lr``


def add_iso_gradient(
    iso_lr: optax.ScalarOrSchedule, lam: float
) -> optax.GradientTransformation:
    """Subtract ``eta_iso * lam * r_t``, outside the adaptive scaling.

    Placed *after* the Adam scaling and the learning rate in the chain, so the term
    it adds is divided by neither ``sqrt(v)`` nor anything else.  ``updates`` are
    already the negative step at this point, hence the minus sign here.
    """

    def init(params):
        del params
        return AddIsoState(count=jnp.zeros([], jnp.int32))

    def update(updates, state, params=None):
        if params is None:
            raise ValueError("add_iso_gradient needs params; pass them to opt.update")
        step = iso_lr(state.count) if callable(iso_lr) else iso_lr
        r = iso_grads(params)
        updates = jax.tree_util.tree_map(
            lambda u, ri: u - step * lam * ri, updates, r
        )
        return updates, AddIsoState(count=optax.safe_int32_increment(state.count))

    return optax.GradientTransformation(init, update)


def adamo(
    learning_rate: optax.ScalarOrSchedule,
    lam: float = 1e-3,
    iso_lr: Optional[optax.ScalarOrSchedule] = None,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    """Adam on the task gradient, plus a decoupled isometry step.

    ``lam = 1e-3`` and ``iso_lr = learning_rate`` are the paper's defaults.
    ``lam = 0`` recovers ``optax.adam`` exactly.  On a parameter group with no
    kernel (``logZ``, ``betas``) the isometry term is zero and this is Adam.
    """
    iso_lr = learning_rate if iso_lr is None else iso_lr
    return optax.chain(
        optax.scale_by_adam(b1=b1, b2=b2, eps=eps),
        optax.scale_by_learning_rate(learning_rate),
        add_iso_gradient(iso_lr, lam),
    )
