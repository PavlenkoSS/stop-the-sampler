"""AdamO: is it Adam plus a decoupled isometry step, and is R_iso what it claims?

The R_iso checks are the ones ``plasticity.optim.adamo`` is held to in
PavlenkoSS/plasticity-umap-reset; the rest check what the port adds here: Flax
kernels, the parameter groups of ``optax.multi_transform``, scheduled rates, the
Hydra config group, and the orthogonal init the paper starts AdamO from.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from omegaconf import OmegaConf

from algorithms.common.adamo import (
    adamo,
    grad_r_iso,
    iso_grads,
    iso_penalty,
    linear_maps,
    output_kernels,
    r_iso,
)
from algorithms.common.diffusion_related.init_model import (
    init_model,
    init_model_non_acyclic,
    init_model_non_acyclic_ml,
    init_model_pisgrad_net_clf,
)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")


def _w(*shape, seed=0):
    return jax.random.normal(jax.random.PRNGKey(seed), shape)


def _orthogonal(d, seed=0):
    return jax.nn.initializers.orthogonal()(jax.random.PRNGKey(seed), (d, d))


def _alg_cfg(optimizer, **overrides):
    cfg = {
        "name": "gfn_non_acyclic",
        "batch_size": 4,
        "iters": 10,
        "step_size": 1e-3,
        "bwd_step_size": 3e-3,
        "logZ_step_size": 1e-1,
        "init_logZ": 0.0,
        "grad_clip": 0.0,
        "weight_decay": 0.0,
        "model": {"dim": 2, "num_layers": 3, "num_hid": 16},
        "lr_schedule": {"type": "constant"},
        "optimizer": optimizer,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


ADAMO = {"name": "adamo", "lam": 1e-3, "iso_lr": None}


# --------------------------------------------------------------------------- #
# the regularizer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(8, 8), (12, 5), (5, 12), (3, 3, 4, 6)])
def test_the_closed_form_gradient_matches_autodiff(shape):
    """The hand-written 4W(WtW - I) cannot be allowed to drift from the definition."""
    w = _w(*shape)
    assert np.allclose(
        np.asarray(grad_r_iso(w)), np.asarray(jax.grad(r_iso)(w)), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("d", [4, 16, 64])
def test_r_iso_and_its_gradient_vanish_at_orthogonality(d):
    q = _orthogonal(d)
    assert float(r_iso(q)) == pytest.approx(0.0, abs=1e-8)
    assert float(jnp.abs(grad_r_iso(q)).max()) == pytest.approx(0.0, abs=1e-5)


def test_a_rotation_of_an_orthogonal_matrix_is_still_invisible():
    q, rot = _orthogonal(32, seed=0), _orthogonal(32, seed=1)
    assert float(r_iso(rot @ q)) == pytest.approx(0.0, abs=1e-6)
    assert float(jnp.abs(grad_r_iso(rot @ q)).max()) == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("shape", [(10, 10), (12, 5), (5, 12)])
def test_r_iso_is_a_function_of_the_singular_values_only(shape):
    """sum_i (sigma_i^2 - 1)^2 -- so Flax's (fan_in, units) layout gives the
    same number as PyTorch's (units, fan_in)."""
    w = _w(*shape, seed=1)
    s = jnp.linalg.svd(w, compute_uv=False)
    expected = float(jnp.sum((s**2 - 1.0) ** 2))
    assert float(r_iso(w)) == pytest.approx(expected, rel=1e-4)
    assert float(r_iso(w.T)) == pytest.approx(expected, rel=1e-4)


# --------------------------------------------------------------------------- #
# which leaves it acts on
# --------------------------------------------------------------------------- #
def _flat_mask(params):
    flat, _ = jax.tree_util.tree_flatten_with_path(linear_maps(params))
    return {jax.tree_util.keystr(path): bool(k) for path, k in flat}


def test_the_isometry_force_is_on_the_kernels_of_the_samplers_networks_only():
    """Biases, logZ and the (1, num_hid) time phase are not linear maps,
    however rank-2 the phase looks."""
    state = init_model(jax.random.PRNGKey(0), 2, _alg_cfg(ADAMO, name="pis"))
    mask = _flat_mask(state.params)
    assert any(mask.values())
    for path, is_map in mask.items():
        assert is_map == path.endswith("['kernel']"), path
    assert any("timestep_phase" in p for p in mask)

    state = init_model_non_acyclic(jax.random.PRNGKey(0), 2, _alg_cfg(ADAMO))
    mask = _flat_mask(state.params)
    for path, is_map in mask.items():
        assert is_map == path.endswith("['kernel']"), path
    assert not mask["['params']['logZ']"]

    g = iso_grads(state.params)
    for path, leaf in jax.tree_util.tree_flatten_with_path(g)[0]:
        if not jax.tree_util.keystr(path).endswith("['kernel']"):
            assert float(jnp.abs(leaf).max()) == 0.0


def test_iso_penalty_sums_r_iso_over_the_kernels():
    params = {"a": {"kernel": _w(6, 4), "bias": _w(4)}, "phase": _w(1, 4, seed=2)}
    assert float(iso_penalty(params)) == pytest.approx(float(r_iso(params["a"]["kernel"])))


# --------------------------------------------------------------------------- #
# the optimizer
# --------------------------------------------------------------------------- #
def _dense_params():
    return {
        "Dense_0": {"kernel": _w(6, 8, seed=3), "bias": _w(8, seed=4)},
        "Dense_1": {"kernel": _w(8, 3, seed=5), "bias": _w(3, seed=6)},
    }


def test_lambda_zero_reproduces_adam_exactly():
    params = _dense_params()
    grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.1, params)
    a, b = adamo(1e-3, lam=0.0), optax.adam(1e-3)
    ua, _ = a.update(grads, a.init(params), params)
    ub, _ = b.update(grads, b.init(params), params)
    for x, y in zip(jax.tree_util.tree_leaves(ua), jax.tree_util.tree_leaves(ub)):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_the_isometry_term_is_decoupled_from_the_adaptive_scaling():
    """AdamO's update minus Adam's is exactly -eta_iso * lam * r_t, whatever the
    task gradient's magnitude: r_t does not pass through sqrt(v)."""
    params = _dense_params()
    lam, lr = 1e-3, 1e-3
    r = iso_grads(params)
    for scale in (0.1, 100.0):
        grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * scale, params)
        o, n = adamo(lr, lam=lam), optax.adam(lr)
        uo, _ = o.update(grads, o.init(params), params)
        un, _ = n.update(grads, n.init(params), params)
        diff = jax.tree_util.tree_map(lambda x, y: x - y, uo, un)
        for du, ri in zip(jax.tree_util.tree_leaves(diff), jax.tree_util.tree_leaves(r)):
            assert np.allclose(np.asarray(du), -lr * lam * np.asarray(ri), atol=1e-9)


def test_adamo_drives_a_perturbed_matrix_back_toward_orthogonality():
    params = {"kernel": _orthogonal(16) * 1.5}  # singular values all 1.5
    opt = adamo(1e-2, lam=1.0)
    state = opt.init(params)
    before = float(r_iso(params["kernel"]))
    for _ in range(200):
        zero = jax.tree_util.tree_map(jnp.zeros_like, params)  # no task gradient
        upd, state = opt.update(zero, state, params)
        params = optax.apply_updates(params, upd)
    assert float(r_iso(params["kernel"])) < 0.01 * before


def test_a_scheduled_iso_lr_follows_its_schedule():
    params = {"kernel": _w(5, 5)}
    schedule = optax.piecewise_constant_schedule(1e-2, {2: 0.1})
    opt = adamo(schedule, lam=1.0)
    state = opt.init(params)
    zero = {"kernel": jnp.zeros((5, 5))}
    r = np.asarray(grad_r_iso(params["kernel"]))
    for step in range(4):
        upd, state = opt.update(zero, state, params)
        eta = 1e-2 if step < 2 else 1e-3
        assert np.allclose(np.asarray(upd["kernel"]), -eta * r, rtol=1e-5, atol=1e-9)


# --------------------------------------------------------------------------- #
# wired into the trainers' TrainState
# --------------------------------------------------------------------------- #
def test_each_parameter_group_takes_its_own_rate_and_logz_is_plain_adam():
    """Through multi_transform: the forward and backward networks' kernels move by
    -lr_group * lam * r_t under a zero task gradient, and nothing else moves."""
    lam = 1e-2
    cfg = _alg_cfg({"name": "adamo", "lam": lam, "iso_lr": None})
    state = init_model_non_acyclic(jax.random.PRNGKey(0), 2, cfg)
    zero = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    updates, _ = state.tx.update(zero, state.opt_state, state.params)

    r = iso_grads(state.params)
    moved = 0
    for (path, upd), (_, ri) in zip(
        *(jax.tree_util.tree_flatten_with_path(t)[0] for t in (updates, r))
    ):
        key = jax.tree_util.keystr(path)
        delta = np.asarray(upd)
        if not key.endswith("['kernel']"):
            assert np.array_equal(delta, np.zeros_like(delta)), key
            continue
        lr = cfg.bwd_step_size if "bwd_state_net" in key else cfg.step_size
        assert np.allclose(delta, -lr * lam * np.asarray(ri), rtol=1e-4, atol=1e-10), key
        moved += bool(np.abs(delta).max() > 0)
    assert moved > 0


def test_adam_config_builds_the_same_optimizer_as_before():
    """``optimizer: adam`` -- and a config with no optimizer at all -- train
    exactly as the hard-coded optax.adam did."""
    key = jax.random.PRNGKey(0)
    grads_key = jax.random.PRNGKey(1)
    states = [
        init_model_non_acyclic(key, 2, _alg_cfg({"name": "adam"})),
        init_model_non_acyclic(key, 2, _alg_cfg(None)),
        init_model_non_acyclic(key, 2, _alg_cfg({"name": "adamo", "lam": 0.0, "iso_lr": None})),
    ]
    leaves, treedef = jax.tree_util.tree_flatten(states[0].params)
    keys = jax.random.split(grads_key, len(leaves))
    grads = treedef.unflatten(
        [jax.random.normal(k, x.shape) for k, x in zip(keys, leaves)]
    )
    out = [s.apply_gradients(grads=grads).params for s in states]
    for other in out[1:]:
        for x, y in zip(jax.tree_util.tree_leaves(out[0]), jax.tree_util.tree_leaves(other)):
            assert np.array_equal(np.asarray(x), np.asarray(y))


def test_an_unknown_optimizer_is_an_error():
    with pytest.raises(ValueError, match="Invalid optimizer"):
        init_model_non_acyclic(jax.random.PRNGKey(0), 2, _alg_cfg({"name": "adamw"}))


@pytest.mark.parametrize(
    "algorithm",
    [
        "gfn_non_acyclic",
        "gfn_non_acyclic_ml",
        "gfn_non_acyclic_baseline",
        "diff_sampler_clf",
        "pis",
        "dis",
        "dds",
    ],
)
def test_the_config_group_selects_adamo(algorithm):
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        default = hydra.compose("base_conf", overrides=[f"algorithm={algorithm}"])
        chosen = hydra.compose(
            "base_conf",
            overrides=[
                f"algorithm={algorithm}",
                "algorithm/optimizer=adamo",
                "algorithm.optimizer.lam=1e-2",
                "algorithm.model.hidden_init=orthogonal",
                "algorithm.optimizer.scope=hidden",
            ],
        )
    assert default.algorithm.optimizer.name == "adam"
    assert default.algorithm.model.hidden_init == "lecun_normal"
    assert chosen.algorithm.optimizer.name == "adamo"
    assert chosen.algorithm.optimizer.lam == pytest.approx(1e-2)
    assert chosen.algorithm.optimizer.iso_lr is None
    assert chosen.algorithm.model.hidden_init == "orthogonal"
    assert chosen.algorithm.optimizer.scope == "hidden"
    with hydra.initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        adamo_default = hydra.compose(
            "base_conf", overrides=[f"algorithm={algorithm}", "algorithm/optimizer=adamo"]
        )
    assert adamo_default.algorithm.optimizer.scope == "all"


# --------------------------------------------------------------------------- #
# orthogonal init of the hidden layers
# --------------------------------------------------------------------------- #
MODEL = {"dim": 2, "num_layers": 3, "num_hid": 16}
BUILDS = {
    "pisgrad_net": (init_model, {"name": "pis"}),
    "non_acyclic_net": (init_model_non_acyclic, {}),
    "non_acyclic_net_shared": (
        init_model_non_acyclic,
        {"model": {**MODEL, "shared_model": True}},
    ),
    "non_acyclic_net_ml": (init_model_non_acyclic_ml, {"num_levels": 2}),
    "pisgrad_net_clf": (init_model_pisgrad_net_clf, {}),
}


def _params(build, hidden_init=None):
    init_fn, overrides = BUILDS[build]
    cfg = _alg_cfg(ADAMO, **overrides)
    if hidden_init is not None:
        cfg.model.hidden_init = hidden_init
    return init_fn(jax.random.PRNGKey(0), 2, cfg).params


def _kernels(params):
    flat, _ = jax.tree_util.tree_flatten_with_path(params)
    return {
        jax.tree_util.keystr(p): w
        for p, w in flat
        if jax.tree_util.keystr(p).endswith("['kernel']")
    }


@pytest.mark.parametrize("build", list(BUILDS))
def test_orthogonal_init_makes_every_hidden_kernel_isometric(build):
    params = _params(build, "orthogonal")
    kernels, out = _kernels(params), output_kernels(params)
    heads = [w for p, w in kernels.items() if p in out]
    hidden = {p: w for p, w in kernels.items() if p not in out}
    assert heads and hidden
    for path, w in hidden.items():
        assert float(r_iso(w)) == pytest.approx(0.0, abs=1e-4), path
    # so all that is left of R_iso at init is the near-zero heads
    assert float(iso_penalty(params)) == pytest.approx(
        float(sum(r_iso(w) for w in heads)), rel=1e-5
    )


@pytest.mark.parametrize("build", list(BUILDS))
def test_orthogonal_init_redraws_the_hidden_kernels_and_nothing_else(build):
    """Same tree, same heads, biases, phases and logZ: the output layers keep
    their near-zero init, so the sampler still starts from the reference process."""
    lecun, ortho = _params(build, "lecun_normal"), _params(build, "orthogonal")
    assert jax.tree.structure(lecun) == jax.tree.structure(ortho)
    out = output_kernels(lecun)
    flat, _ = jax.tree_util.tree_flatten_with_path(lecun)
    for (path, a), b in zip(flat, jax.tree.leaves(ortho)):
        key = jax.tree_util.keystr(path)
        if key.endswith("['kernel']") and key not in out:
            assert not np.array_equal(np.asarray(a), np.asarray(b)), key
        else:
            assert np.array_equal(np.asarray(a), np.asarray(b)), key


def test_lecun_normal_is_the_default():
    implicit, explicit = _params("non_acyclic_net"), _params("non_acyclic_net", "lecun_normal")
    for a, b in zip(jax.tree.leaves(implicit), jax.tree.leaves(explicit)):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_an_unknown_hidden_init_is_an_error():
    with pytest.raises(ValueError, match="hidden_init"):
        _params("non_acyclic_net", "xavier")


# --------------------------------------------------------------------------- #
# scope: which kernels the isometry force covers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hidden_init", ["lecun_normal", "orthogonal"])
@pytest.mark.parametrize("build", list(BUILDS))
def test_output_kernels_finds_exactly_the_output_layers(build, hidden_init):
    """Cross-checked against a second rule that holds at these sizes: every hidden
    layer is num_hid = 16 wide, and no output layer is."""
    params = _params(build, hidden_init)
    kernels = _kernels(params)
    expected = {p for p, w in kernels.items() if w.shape[-1] != MODEL["num_hid"]}
    assert expected
    assert output_kernels(params) == expected


def _iso_updates(scope, lam=1e-2):
    """The update each leaf gets from a zero task gradient: the isometry step alone."""
    cfg = _alg_cfg({"name": "adamo", "lam": lam, "iso_lr": None, "scope": scope})
    state = init_model_non_acyclic(jax.random.PRNGKey(0), 2, cfg)
    zero = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    updates, _ = state.tx.update(zero, state.opt_state, state.params)
    flat, _ = jax.tree_util.tree_flatten_with_path(updates)
    upd = {jax.tree_util.keystr(p): np.asarray(u) for p, u in flat}
    r = {
        jax.tree_util.keystr(p): np.asarray(g)
        for p, g in jax.tree_util.tree_flatten_with_path(iso_grads(state.params))[0]
    }
    lr = {k: cfg.bwd_step_size if "bwd_state_net" in k else cfg.step_size for k in upd}
    return upd, r, lr, output_kernels(state.params), lam


def test_scope_hidden_leaves_the_output_layers_to_plain_adam():
    upd, r, lr, heads, lam = _iso_updates("hidden")
    assert heads
    for key, u in upd.items():
        if key in heads or not key.endswith("['kernel']"):
            assert np.array_equal(u, np.zeros_like(u)), key
        else:
            assert np.allclose(u, -lr[key] * lam * r[key], rtol=1e-4, atol=1e-10), key
            assert np.abs(u).max() > 0, key


def test_scope_all_covers_the_output_layers_too():
    """The paper's: the sum over layers includes the output layer."""
    upd, r, lr, heads, lam = _iso_updates("all")
    for key in heads:
        assert np.abs(upd[key]).max() > 0, key
        assert np.allclose(upd[key], -lr[key] * lam * r[key], rtol=1e-4, atol=0), key


def test_an_unknown_scope_is_an_error():
    with pytest.raises(ValueError, match="scope"):
        _iso_updates("heads")
