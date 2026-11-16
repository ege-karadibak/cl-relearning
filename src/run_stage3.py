"""Stage 3 for one arm, seed and subset: relearn CIFAR-10 from 5k images with a fresh head.

    python src/run_stage3.py --arm A --seed 1 --subset a_seen          real run
    python src/run_stage3.py --arm A --seed 1 --subset never_seen
    python src/run_stage3.py --arm A --seed 1 --subset never_seen --calibrate --lr 0.03
                                                                       calibration: one learning rate
    python src/run_stage3.py --arm A --seed 1 --subset a_seen --smoke  short test on the smoke checkpoints
    add --device cuda:1 (or mps) to use another device than the one in configs/base.yaml

Starts from {arm}_stage2.pt (both subsets of an arm start from the same model). Adds a fresh
"relearn" head, then: warmup, constant learning rate, `steps` steps on the 5k subset.
Measures CIFAR-10 test accuracy and loss every 25 steps early on, then every 100 steps
(the validation split instead, during calibration).

Writes logs/seed{seed}/{arm}_stage3_{subset}.jsonl: these curves are the main result.
No checkpoint is saved: nothing comes after stage 3.

Calibration: run each learning rate in lr_choices (--calibrate --lr X), compare the validation
curves, then write the chosen lr in configs/stage3.yaml and set calibrated: true.
"""

import argparse                              # command-line options

import data
from data import stage3_data, val_data
from evaluate import evaluate
from train import train
from utils import CHECKPOINTS, LOGS, Logger, checkpoint_path, get_device, load_checkpoint, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3 (relearning) for one arm, seed and subset.")
    parser.add_argument("--arm", required=True, choices=["A", "B", "C"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--subset", required=True, choices=["a_seen", "never_seen"])
    parser.add_argument("--device", default=None, help="cuda:0, cuda:1 or mps (default: configs/base.yaml)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--calibrate", action="store_true", help="score on validation; use with --lr")
    mode.add_argument("--smoke", action="store_true", help="short test on the smoke checkpoints")
    parser.add_argument("--lr", type=float, default=None, help="learning rate to try (calibration only)")
    return parser.parse_args()


def measure_steps(total, dense_until, dense_every, sparse_every):
    """Steps to measure at: often at the start (where curves differ most), then less often."""
    dense = range(0, min(dense_until, total) + 1, dense_every)
    sparse = range(dense_until, total + 1, sparse_every)
    return set(dense) | set(sparse) | {total}


def main():
    args = parse_args()
    arm, seed, subset = args.arm, args.seed, args.subset
    cfg = load_config("stage3")
    if args.lr is not None and not args.calibrate:
        raise SystemExit("--lr is only for calibration; real runs use the lr in configs/stage3.yaml.")

    run_name = f"{arm}_stage3_{subset}"
    if args.smoke:
        cfg.update(steps=60, warmup_steps=10, dense_until=20, dense_every=10, sparse_every=20)
        start_path = CHECKPOINTS / "smoke" / f"seed{seed}" / f"{arm}_stage2.pt"
        log_folder = LOGS / "smoke"
        for old_file in (log_folder / f"seed{seed}" / f"{run_name}.jsonl",
                         log_folder / f"seed{seed}" / f"{run_name}.meta.json"):
            old_file.unlink(missing_ok=True)     # smoke outputs are throwaway: start clean
    elif args.calibrate:
        if args.lr is None:
            raise SystemExit("--calibrate needs --lr (one of lr_choices in configs/stage3.yaml).")
        cfg["lr"] = args.lr
        run_name += f"_lr{args.lr:g}"
        start_path, log_folder = checkpoint_path(arm, 2, seed), LOGS / "calibration"
    else:
        if not cfg["calibrated"]:
            raise SystemExit("configs/stage3.yaml is not calibrated yet: choose lr with --calibrate, "
                             "then set calibrated: true.")
        start_path, log_folder = checkpoint_path(arm, 2, seed), LOGS
    resume = log_folder / f"seed{seed}" / f"{run_name}.resume.pt"

    device = get_device(args.device or cfg["device"], cfg["gpu_memory_fraction"])
    cfg["device"] = str(device)
    data.configure(cfg["num_workers"], pin_memory=device.type == "cuda")

    net, meta = load_checkpoint(start_path, device)
    set_seed(seed)                           # same new head (and data order) for every arm within a seed
    net.add_head("relearn")                  # fresh CIFAR-10 head for every arm: only backbones are compared
    train_loader, test_loader = stage3_data(subset, seed)
    score_loader, prefix = (val_data(), "val") if args.calibrate else (test_loader, "test")

    total = cfg["steps"]
    measure_at = measure_steps(total, cfg["dense_until"], cfg["dense_every"], cfg["sparse_every"])

    def measure(step):
        r = evaluate(net, score_loader, device)
        return {f"{prefix}_acc": r["acc"], f"{prefix}_loss": r["loss"]}

    logger = Logger(run_name, seed, config={**cfg, "subset": subset}, folder=log_folder,
                    resume=resume.exists(), device=device)
    mode = f" (calibration, lr {cfg['lr']:g})" if args.calibrate else " (smoke test)" if args.smoke else ""
    print(f"arm {arm}, seed {seed}, subset {subset}: {total} steps from {start_path.name} "
          f"on {device.type}{mode}")

    summary = train(net, train_loader, device, cfg, total, measure_at, measure, logger,
                    warmup_steps=cfg["warmup_steps"], resume_path=resume, resume_every=500)
    print("final:", {k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()})


if __name__ == "__main__":                   # needed: the data loaders start worker processes
    main()
