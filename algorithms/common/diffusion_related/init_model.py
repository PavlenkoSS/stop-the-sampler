import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.traverse_util import path_aware_map

from algorithms.common.adamo import SCOPES, adamo, output_kernels
from algorithms.common.models.pisgrad_net import PISGRADNet
from algorithms.common.models.non_acyclic_net import NonAcyclicNet
from algorithms.common.models.pisgrad_net_clf import PISGRADNetClf
from algorithms.common.models.non_acyclic_net_ml import NonAcyclicNetML


def pisgrad_net_label_map(path, _):
    if (
        "flow_state_time_net" in path
        or "flow_time_coder_state" in path
        or "flow_timestep_phase" in path
    ):
        return "logflow_optim"
    elif "logZ" in path:
        return "logZ_optim"
    elif "betas" in path:
        return "betas_optim"
    elif "bwd_state_net" in path:
        return "network_bwd_optim"
    else:
        return "network_optim"


def build_optimizer(alg_cfg, base_lr, build_lr_schedule, heads=frozenset()):
    """The optimizer one parameter group is trained with, as ``alg_cfg.optimizer``
    names it: ``adam`` (the default, and the only choice before) or ``adamo``.

    Every group goes through here, so ``adamo`` is AdamO on every group; on one
    without kernels (``logZ``, ``betas``) its isometry term is zero and it is Adam.
    ``heads`` are the output layers' kernel paths (``output_kernels`` of the fresh
    params), which ``scope: hidden`` leaves to plain Adam.
    """
    opt_cfg = getattr(alg_cfg, "optimizer", None)
    name = "adam" if opt_cfg is None else opt_cfg.name
    match name:
        case "adam":
            return optax.adam(learning_rate=build_lr_schedule(base_lr))
        case "adamo":
            # iso_lr follows the same schedule as the step size; null means
            # equal to this group's step size, as in the paper
            iso_lr = getattr(opt_cfg, "iso_lr", None)
            scope = getattr(opt_cfg, "scope", "all")
            if scope not in SCOPES:
                raise ValueError(
                    f"optimizer.scope must be one of {SCOPES}, got {scope!r}"
                )
            if scope == "hidden" and not heads:
                raise ValueError(
                    "optimizer.scope=hidden, but no output layer was found"
                )
            return adamo(
                learning_rate=build_lr_schedule(base_lr),
                lam=opt_cfg.lam,
                iso_lr=None if iso_lr is None else build_lr_schedule(iso_lr),
                exclude=heads if scope == "hidden" else frozenset(),
            )
        case _:
            raise ValueError(f"Invalid optimizer: {name}")


def init_model(key, dim, alg_cfg) -> TrainState:
    def build_lr_schedule(base_lr):
        sched_cfg = getattr(alg_cfg, "lr_schedule", None)
        if sched_cfg is None:
            return lambda step: base_lr

        sched_type = getattr(sched_cfg, "type", "constant")
        match sched_type:
            case "constant":
                return lambda step: base_lr
            case "multistep":
                milestones_arr = (
                    jnp.array(sched_cfg.milestones, dtype=jnp.int32)
                    if len(sched_cfg.milestones) > 0
                    else None
                )

                def multistep_fn(step):
                    if milestones_arr is None:
                        num_decays = 0
                    else:
                        num_decays = jnp.sum(step >= milestones_arr)
                    return base_lr * (sched_cfg.gamma**num_decays)

                return multistep_fn
            case "cosine":
                decay_steps = max(alg_cfg.iters, 1)
                return optax.cosine_decay_schedule(
                    init_value=base_lr,
                    decay_steps=decay_steps,
                    alpha=sched_cfg.end_factor,
                )
            case _:
                raise ValueError(f"Invalid learning rate scheduler type: {sched_type}")

    model = PISGRADNet(**alg_cfg.model)
    # model = LangevinNetwork(**alg_cfg.model)

    key, key_gen = jax.random.split(key)
    params = model.init(
        key_gen,
        jnp.ones([alg_cfg.batch_size, dim]),
        jnp.ones([alg_cfg.batch_size, 1]),
        jnp.ones([alg_cfg.batch_size, dim]),
    )

    heads = output_kernels(params)
    if "gfn" not in alg_cfg.name:
        optimizer = optax.chain(
            optax.zero_nans(),
            (
                optax.clip_by_global_norm(alg_cfg.grad_clip)
                if alg_cfg.grad_clip > 0
                else optax.identity()
            ),
            build_optimizer(alg_cfg, alg_cfg.step_size, build_lr_schedule, heads),
        )
    else:  # "gfn" in alg_cfg.name
        optimizers_map = {
            "network_optim": build_optimizer(
                alg_cfg, alg_cfg.step_size, build_lr_schedule, heads
            )
        }
        if "gfn_subtb" in alg_cfg.name:
            optimizers_map["logflow_optim"] = build_optimizer(
                alg_cfg, alg_cfg.logflow_step_size, build_lr_schedule, heads
            )
            if alg_cfg.beta_schedule == "learnt":
                params["params"]["betas"] = jnp.ones((alg_cfg.num_steps,))
                optimizers_map["betas_optim"] = build_optimizer(
                    alg_cfg, alg_cfg.beta_step_size, build_lr_schedule, heads
                )

        if not (
            "gfn_tb" in alg_cfg.name and alg_cfg.loss_type == "lv"
        ):  # for lv, logZ is not used
            params["params"]["logZ"] = jnp.array((alg_cfg.init_logZ,))
            optimizers_map["logZ_optim"] = build_optimizer(
                alg_cfg, alg_cfg.logZ_step_size, build_lr_schedule, heads
            )

        param_labels = path_aware_map(pisgrad_net_label_map, params)
        partitioned_optimizer = optax.multi_transform(optimizers_map, param_labels)

        optimizer = optax.chain(
            optax.zero_nans(),
            (
                optax.clip_by_global_norm(alg_cfg.grad_clip)
                if alg_cfg.grad_clip > 0
                else optax.identity()
            ),
            partitioned_optimizer,
        )

    model_state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    return model_state


def init_model_non_acyclic(key, dim, alg_cfg) -> TrainState:
    def build_lr_schedule(base_lr):
        sched_cfg = getattr(alg_cfg, "lr_schedule", None)
        if sched_cfg is None:
            return lambda step: base_lr

        sched_type = getattr(sched_cfg, "type", "constant")
        match sched_type:
            case "constant":
                return lambda step: base_lr
            case "multistep":
                milestones_arr = (
                    jnp.array(sched_cfg.milestones, dtype=jnp.int32)
                    if len(sched_cfg.milestones) > 0
                    else None
                )

                def multistep_fn(step):
                    if milestones_arr is None:
                        num_decays = 0
                    else:
                        num_decays = jnp.sum(step >= milestones_arr)
                    return base_lr * (sched_cfg.gamma**num_decays)

                return multistep_fn
            case "cosine":
                decay_steps = max(alg_cfg.iters, 1)
                return optax.cosine_decay_schedule(
                    init_value=base_lr,
                    decay_steps=decay_steps,
                    alpha=sched_cfg.end_factor,
                )
            case _:
                raise ValueError(f"Invalid learning rate scheduler type: {sched_type}")

    # Define the model
    model = NonAcyclicNet(**alg_cfg.model)

    key, key_gen = jax.random.split(key)
    # fmt: off
    params = model.init(
        key_gen,
        jnp.ones([alg_cfg.batch_size, dim]),
        jnp.ones([alg_cfg.batch_size,]),
        jnp.ones([alg_cfg.batch_size, dim]),
        predict_fwd=True,
    )
    if not model.shared_model:
        key, key_gen = jax.random.split(key)
        params_bwd = model.init(
            key_gen,
            jnp.ones([alg_cfg.batch_size, dim]),
            predict_fwd=False,
        )
        params["params"] = {**params["params"], **params_bwd["params"]}

    heads = output_kernels(params)
    optimizers_map = {
        "network_optim": build_optimizer(
            alg_cfg, alg_cfg.step_size, build_lr_schedule, heads
        )
    }

    if not model.shared_model:
        optimizers_map["network_bwd_optim"] = build_optimizer(
            alg_cfg, alg_cfg.bwd_step_size, build_lr_schedule, heads
        )

    params["params"]["logZ"] = jnp.array((alg_cfg.init_logZ,))
    optimizers_map["logZ_optim"] = build_optimizer(
        alg_cfg, alg_cfg.logZ_step_size, build_lr_schedule, heads
    )

    param_labels = path_aware_map(pisgrad_net_label_map, params)
    partitioned_optimizer = optax.multi_transform(optimizers_map, param_labels)

    mask_fn = lambda p: jax.tree.map(lambda x: x.ndim != 1, p)

    optimizer = optax.chain(
        optax.zero_nans(),
        (
            optax.clip_by_global_norm(alg_cfg.grad_clip)
            if alg_cfg.grad_clip > 0
            else optax.identity()
        ),
        (
            optax.masked(optax.add_decayed_weights(alg_cfg.weight_decay), mask_fn)
            if alg_cfg.weight_decay > 0
            else optax.identity()
        ),
        partitioned_optimizer,
    )

    model_state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    return model_state


def init_model_pisgrad_net_clf(key, dim, alg_cfg) -> TrainState:
    def build_lr_schedule(base_lr):
        sched_cfg = getattr(alg_cfg, "lr_schedule", None)
        if sched_cfg is None:
            return lambda step: base_lr

        sched_type = getattr(sched_cfg, "type", "constant")
        match sched_type:
            case "constant":
                return lambda step: base_lr
            case "multistep":
                milestones_arr = (
                    jnp.array(sched_cfg.milestones, dtype=jnp.int32)
                    if len(sched_cfg.milestones) > 0
                    else None
                )

                def multistep_fn(step):
                    if milestones_arr is None:
                        num_decays = 0
                    else:
                        num_decays = jnp.sum(step >= milestones_arr)
                    return base_lr * (sched_cfg.gamma**num_decays)

                return multistep_fn
            case "cosine":
                decay_steps = max(alg_cfg.iters, 1)
                return optax.cosine_decay_schedule(
                    init_value=base_lr,
                    decay_steps=decay_steps,
                    alpha=sched_cfg.end_factor,
                )
            case _:
                raise ValueError(f"Invalid learning rate scheduler type: {sched_type}")

    # Define the model
    model = PISGRADNetClf(**alg_cfg.model)

    key, key_gen = jax.random.split(key)
    # fmt: off
    params = model.init(
        key_gen,
        jnp.ones([alg_cfg.batch_size, dim]),
        jnp.ones([alg_cfg.batch_size, 1]),
        jnp.ones([alg_cfg.batch_size,]),
        jnp.ones([alg_cfg.batch_size, dim]),
        predict_fwd=True,
    )
    if not model.shared_model:
        key, key_gen = jax.random.split(key)
        params_bwd = model.init(
            key_gen,
            jnp.ones([alg_cfg.batch_size, dim]),
            jnp.ones([alg_cfg.batch_size, 1]),
            predict_fwd=False,
        )
        params["params"] = {**params["params"], **params_bwd["params"]}

    heads = output_kernels(params)
    optimizers_map = {
        "network_optim": build_optimizer(
            alg_cfg, alg_cfg.step_size, build_lr_schedule, heads
        )
    }

    if not model.shared_model:
        optimizers_map["network_bwd_optim"] = build_optimizer(
            alg_cfg, alg_cfg.bwd_step_size, build_lr_schedule, heads
        )

    params["params"]["logZ"] = jnp.array((alg_cfg.init_logZ,))
    optimizers_map["logZ_optim"] = build_optimizer(
        alg_cfg, alg_cfg.logZ_step_size, build_lr_schedule, heads
    )

    param_labels = path_aware_map(pisgrad_net_label_map, params)
    partitioned_optimizer = optax.multi_transform(optimizers_map, param_labels)

    mask_fn = lambda p: jax.tree.map(lambda x: x.ndim != 1, p)

    optimizer = optax.chain(
        optax.zero_nans(),
        (
            optax.clip_by_global_norm(alg_cfg.grad_clip)
            if alg_cfg.grad_clip > 0
            else optax.identity()
        ),
        (
            optax.masked(optax.add_decayed_weights(alg_cfg.weight_decay), mask_fn)
            if alg_cfg.weight_decay > 0
            else optax.identity()
        ),
        partitioned_optimizer,
    )

    model_state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    return model_state


def init_model_non_acyclic_ml(key, dim, alg_cfg) -> TrainState:
    def build_lr_schedule(base_lr):
        sched_cfg = getattr(alg_cfg, "lr_schedule", None)
        if sched_cfg is None:
            return lambda step: base_lr

        sched_type = getattr(sched_cfg, "type", "constant")
        match sched_type:
            case "constant":
                return lambda step: base_lr
            case "multistep":
                milestones_arr = (
                    jnp.array(sched_cfg.milestones, dtype=jnp.int32)
                    if len(sched_cfg.milestones) > 0
                    else None
                )

                def multistep_fn(step):
                    if milestones_arr is None:
                        num_decays = 0
                    else:
                        num_decays = jnp.sum(step >= milestones_arr)
                    return base_lr * (sched_cfg.gamma**num_decays)

                return multistep_fn
            case "cosine":
                decay_steps = max(alg_cfg.iters, 1)
                return optax.cosine_decay_schedule(
                    init_value=base_lr,
                    decay_steps=decay_steps,
                    alpha=sched_cfg.end_factor,
                )
            case _:
                raise ValueError(f"Invalid learning rate scheduler type: {sched_type}")

    # Define the model
    model = NonAcyclicNetML(**alg_cfg.model)

    key, key_gen = jax.random.split(key)
    # fmt: off
    params = model.init(
        key_gen,
        jnp.ones([alg_cfg.batch_size, dim]),
        jnp.ones([alg_cfg.batch_size, 1]),
        jnp.ones([alg_cfg.batch_size,]),
        jnp.ones([alg_cfg.batch_size, dim]),
        predict_fwd=True,
    )
    key, key_gen = jax.random.split(key)
    params_bwd = model.init(
        key_gen,
        jnp.ones([alg_cfg.batch_size, dim]),
        jnp.ones([alg_cfg.batch_size, 1]),
        predict_fwd=False,
    )
    params["params"] = {**params["params"], **params_bwd["params"]}

    heads = output_kernels(params)
    optimizers_map = {
        "network_optim": build_optimizer(
            alg_cfg, alg_cfg.step_size, build_lr_schedule, heads
        )
    }

    optimizers_map["network_bwd_optim"] = build_optimizer(
        alg_cfg, alg_cfg.bwd_step_size, build_lr_schedule, heads
    )
    params["params"]["logZ"] = jnp.full((alg_cfg.num_levels,), alg_cfg.init_logZ)
    optimizers_map["logZ_optim"] = build_optimizer(
        alg_cfg, alg_cfg.logZ_step_size, build_lr_schedule, heads
    )

    param_labels = path_aware_map(pisgrad_net_label_map, params)
    partitioned_optimizer = optax.multi_transform(optimizers_map, param_labels)

    mask_fn = lambda p: jax.tree.map(lambda x: x.ndim != 1, p)

    optimizer = optax.chain(
        optax.zero_nans(),
        (
            optax.clip_by_global_norm(alg_cfg.grad_clip)
            if alg_cfg.grad_clip > 0
            else optax.identity()
        ),
        (
            optax.masked(optax.add_decayed_weights(alg_cfg.weight_decay), mask_fn)
            if alg_cfg.weight_decay > 0
            else optax.identity()
        ),
        partitioned_optimizer,
    )

    model_state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    return model_state
