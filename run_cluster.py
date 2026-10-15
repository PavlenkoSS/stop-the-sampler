"""Cluster launcher for gfn_non_acyclic on credit (plast): DB and SubTB."""

import argparse
import os
from itertools import product

import yaml


LOSSES = ("db", "subtb")
GAMMAS = ("1e-3", "1e-4")
REG_COEFS = ("1e-2", "1e-3")

RUN_PARAMS = {
    "algorithm": "gfn_non_acyclic",
    "target": "credit",
    "target.gfn_non_acyclic.grad_clip": "1e-7",
    "target.gfn_non_acyclic.step_size": "1e-5",
    "target.gfn_non_acyclic.bwd_step_size": "1e-5",
    "target.all.mean_clip": "5",
    "algorithm.only_clf_reg": True,
    "target.all.use_lp": True,
    "target.all.outer_clip": "1.0",
    "target.all.inner_clip": "5.0",
    "target.all.fwd_log_var_range": "2",
    "target.all.bwd_log_var_range": "2",
    "algorithm.use_weights": True,
    "target.gfn_non_acyclic.init_std": "0.1",
}


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


def create_python_commands(args):
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    api_key = workspace = project_name = None
    if not args.no_comet:
        api_key, workspace, project_name = _load_cometml_secret(repo_dir)

    losses = args.losses if args.losses else list(LOSSES)
    gammas = args.gammas if args.gammas else list(GAMMAS)
    reg_coefs = args.reg_coefs if args.reg_coefs else list(REG_COEFS)

    combinations = []
    for loss, gamma, reg_coef in product(losses, gammas, reg_coefs):
        params = {
            **RUN_PARAMS,
            "algorithm.loss_type": loss,
            "target.gfn_non_acyclic.gamma": gamma,
            "target.gfn_non_acyclic.reg_coef": reg_coef,
            "use_cometml": _hydra_bool(not args.no_comet),
        }
        if project_name:
            params["cometml.project_name"] = project_name

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
    python_commands = create_python_commands(args)
    version = args.log_version
    name = "plast"
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
            wrapped = python_command.replace('"', '\\"')
            os.system(f'{sbatch_command} --wrap="{wrapped}"')


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Launch gfn_non_acyclic credit (plast) DB/SubTB on the cluster."
    )
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--cpu_cores", type=int, default=12)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--partition", type=str, default="rocky")
    p.add_argument("--time", type=str, default="0-04:00:00")
    p.add_argument("--log_version", type=int, default=1)
    p.add_argument(
        "--conda_env",
        type=str,
        default="rocky-gfn-smc-jax",
        help="Conda env name (override if your cluster env differs).",
    )
    p.add_argument(
        "--losses",
        nargs="+",
        default=None,
        choices=list(LOSSES),
        help="Loss types (default: db subtb).",
    )
    p.add_argument(
        "--gammas",
        nargs="+",
        default=None,
        help="Grid of target.gfn_non_acyclic.gamma (default: 1e-3).",
    )
    p.add_argument(
        "--reg_coefs",
        nargs="+",
        default=None,
        help="Grid of target.gfn_non_acyclic.reg_coef (default: 1e-2).",
    )
    p.add_argument(
        "--no_comet",
        action="store_true",
        help="Disable Comet logging (skips configs/cometml_secret.yaml).",
    )
    args = p.parse_args()
    main(args)
