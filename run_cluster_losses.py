"""
Cluster launcher for Table 1 (arXiv:2606.16073): ULA w/ clf and ULA w/ clf & corr
with TB / SubTB / DB losses on GMM9 (d=2), Funnel (d=2), GMM9 (d=40), ManyWell (d=40).

Hyperparameters follow Appendix A.3 / Table 3.
"""

import argparse
import os
from itertools import product

import yaml


# Table 3 + A.3 (init_std: 1 low-d, 5 high-d)
TARGET_CONFIGS = {
    "gmm9_2d": {
        "target": "gaussian_mixture9",
        "target.dim": 2,
        "algorithm.step_size": "1e-3",
        "algorithm.bwd_step_size": "1e-3",
        "algorithm.reg_coef": "1e-1",
        "algorithm.model.gamma": "1e-1",
        "algorithm.num_steps": 100,
        "algorithm.grad_clip": "1.0",
        "algorithm.init_std": "1.0",
    },
    "funnel_2d": {
        "target": "funnel",
        "target.dim": 2,
        "algorithm.step_size": "1e-4",
        "algorithm.bwd_step_size": "1e-4",
        "algorithm.reg_coef": "1e-2",
        "algorithm.model.gamma": "1e-2",
        "algorithm.num_steps": 100,
        "algorithm.grad_clip": "1e-3",
        "algorithm.init_std": "1.0",
    },
    "gmm9_40d": {
        "target": "gaussian_mixture9",
        "target.dim": 40,
        "algorithm.step_size": "1e-4",
        "algorithm.bwd_step_size": "1e-4",
        "algorithm.reg_coef": "1.0",
        "algorithm.model.gamma": "1e-1",
        "algorithm.num_steps": 100,
        "algorithm.grad_clip": "1e-3",
        "algorithm.init_std": "5.0",
    },
    "many_well_40d": {
        "target": "many_well",
        "target.dim": 40,
        "algorithm.step_size": "1e-5",
        "algorithm.bwd_step_size": "1e-5",
        "algorithm.reg_coef": "1e-1",
        "algorithm.model.gamma": "1e-2",
        "algorithm.num_steps": 100,
        "algorithm.grad_clip": "1e-5",
        "algorithm.init_std": "5.0",
    },
}

# ULA w/ clf vs ULA w/ clf & corr
VARIANTS = {
    "ula_clf": {"algorithm.model.learn_fwd_corrections": False},
    "ula_clf_corr": {"algorithm.model.learn_fwd_corrections": True},
}

LOSSES = ("subtb", "db")
SEEDS = (0,)


def _hydra_bool(value):
    return "true" if value else "false"


def _load_cometml_secret(repo_dir):
    path = os.path.join(repo_dir, "configs", "cometml_secret.yaml")
    with open(path) as f:
        data = yaml.safe_load(f)
    comet = data["cometml"]
    api_key = comet.get("api_key")
    workspace = comet.get("workspace")
    project_name = comet.get("project_name")
    assert api_key, f"Missing cometml.api_key in {path}"
    assert workspace, f"Missing cometml.workspace in {path}"
    return api_key, workspace, project_name


def create_argument_combinations(args):
    """Build run.py commands for the Table 1 grid."""
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    api_key = workspace = project_name = None
    if not args.no_comet:
        api_key, workspace, project_name = _load_cometml_secret(repo_dir)

    targets = args.targets if args.targets else list(TARGET_CONFIGS.keys())
    variants = args.variants if args.variants else list(VARIANTS.keys())
    losses = args.losses if args.losses else list(LOSSES)
    seeds = args.seeds if args.seeds is not None else list(SEEDS)

    for t in targets:
        assert t in TARGET_CONFIGS, f"Unknown target {t}; choose from {list(TARGET_CONFIGS)}"
    for v in variants:
        assert v in VARIANTS, f"Unknown variant {v}; choose from {list(VARIANTS)}"
    for loss in losses:
        assert loss in LOSSES, f"Unknown loss {loss}; choose from {LOSSES}"

    combinations = []

    for target_name, variant_name, loss, seed in product(targets, variants, losses, seeds):
        params = {
            "algorithm": "gfn_non_acyclic",
            "algorithm.loss_type": loss,
            "seed": seed,
            "use_cometml": _hydra_bool(not args.no_comet),
            "compute_forward_metrics": "true",
            **TARGET_CONFIGS[target_name],
            **VARIANTS[variant_name],
        }
        if project_name:
            params["cometml.project_name"] = project_name
        # Keep n_chunks aligned with num_steps for SubTB
        params["algorithm.n_chunks"] = params["algorithm.num_steps"]

        cmd = [
            f"cd '{repo_dir}' &&",
            f"source $(conda info --base)/etc/profile.d/conda.sh &&",
            f"conda activate {args.conda_env} &&",
        ]
        if not args.no_comet:
            cmd.append(
                f"COMET_API_KEY='{api_key}' COMET_WORKSPACE='{workspace}' "
                f"COMETML_API_KEY='{api_key}' COMETML_WORKSPACE='{workspace}'"
            )
        cmd.extend(["python", "run.py"])
        for name, value in params.items():
            if isinstance(value, bool):
                value = _hydra_bool(value)
            cmd.append(f"{name}={value}")

        combinations.append(" ".join(cmd))

    return combinations


def main(args):
    python_commands = create_argument_combinations(args)
    version = args.log_version
    # constraint = args.node
    name = "plasticity"
    log_dir = f"sbatch_logs_{version}/{name}"
    os.makedirs(log_dir, exist_ok=True)

    print(f"Submitting {len(python_commands)} jobs (dry_run={args.dry_run})")
    for python_command in python_commands:
        if args.gpus == 0:
            python_command = f"JAX_PLATFORMS=cpu {python_command}"

        sbatch_command = (
            f"sbatch -A proj_1752 -c {args.cpu_cores} -G {args.gpus} "
            f"--job-name={name} "
            f"--error={log_dir}/%j.err --output={log_dir}/%j.log "
            f'--partition="{args.partition}" --time={args.time}'
        )
        print(python_command, "\n")
        if not args.dry_run:
            # Escape double quotes inside --wrap
            wrapped = python_command.replace('"', '\\"')
            os.system(f'{sbatch_command} --wrap="{wrapped}"')


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Launch Table 1 stop-the-sampler runs on the cluster."
    )
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--cpu_cores", type=int, default=12)
    p.add_argument("--gpus", type=int, default=1)
    # p.add_argument("--node", type=str, default="type_g")
    p.add_argument("--partition", type=str, default="rocky")
    p.add_argument("--time", type=str, default="0-12:00:00")
    p.add_argument("--log_version", type=int, default=1)
    p.add_argument(
        "--conda_env",
        type=str,
        default="rocky-gfn-smc-jax",
        help="Conda env name (override if your cluster env differs).",
    )
    p.add_argument(
        "--targets",
        nargs="+",
        default=None,
        choices=list(TARGET_CONFIGS.keys()),
        help="Subset of targets (default: all four Table 1 targets).",
    )
    p.add_argument(
        "--variants",
        nargs="+",
        default=None,
        choices=list(VARIANTS.keys()),
        help="ula_clf and/or ula_clf_corr (default: both).",
    )
    p.add_argument(
        "--losses",
        nargs="+",
        default=None,
        choices=list(LOSSES),
        help="tb / subtb / db (default: all three).",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seeds (default: None).",
    )
    p.add_argument(
        "--no_comet",
        action="store_true",
        help="Disable Comet logging (skips configs/cometml_secret.yaml).",
    )
    args = p.parse_args()
    main(args)
